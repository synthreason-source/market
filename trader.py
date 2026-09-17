from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import pystray

# ============================================================
# LIVE TRADING SAFETY WARNING
# ============================================================
# This script executes real orders using financial capital via API keys.
# Ensure your API keys have correct IP restrictions and sizing controls.
# ============================================================

REFRESH_INTERVAL_SECONDS = 30       
HISTORY_DAYS = 90                   
RETRAIN_EVERY_N_CYCLES = 45         

FORECAST_HORIZON_SECONDS = 24 * 60 * 60
FORECAST_STEPS = max(1, FORECAST_HORIZON_SECONDS // REFRESH_INTERVAL_SECONDS)

COINAPI_KEY = os.environ.get("COINAPI_KEY", "YOUR_COINAPI_KEY_HERE")
EMS_BASE_URL = "https://ems.coinapi.io/v1"

# Live trading configuration
LIVE_TRADING_ENABLED = False        # Flip to True to unlock live execution
ORDER_AMOUNT_USD = 10.0             # Real order size in quote currency

MOVE_THRESHOLDS = [
    ("strong_down", -1.5),
    ("down", -0.3),
    ("flat", 0.3),
    ("up", 1.5),
    ("strong_up", math.inf),
]
MOVE_MIDPOINTS = {
    "strong_down": -2.5, "down": -0.8, "flat": 0.0, "up": 0.8, "strong_up": 2.5,
}
MOVE_COLOR_BUCKET = {
    "strong_down": "down", "down": "down",
    "flat": "flat",
    "up": "up", "strong_up": "up",
}
COLORS = {
    "up": (46, 160, 67),      
    "down": (219, 68, 55),    
    "flat": (128, 128, 128),  
    "unknown": (100, 100, 100),
}

random.seed()

@dataclass(frozen=True)
class AssetSpec:
    key: str                
    label: str              
    kind: str               
    coinapi_symbol: str     
    exchange_id: str        # Target exchange ID for EMS routing (e.g., "BITSTAMP", "COINBASE")
    base_asset: str         # e.g., "BTC"
    quote_asset: str        # e.g., "USD"


ASSETS: List[AssetSpec] = [
    AssetSpec("btc", "BTC/USD", "crypto", "BITSTAMP_SPOT_BTC_USD", "BITSTAMP", "BTC", "USD"),
    AssetSpec("eth", "ETH/USD", "crypto", "BITSTAMP_SPOT_ETH_USD", "BITSTAMP", "ETH", "USD"),
    AssetSpec("sol", "SOL/USD", "crypto", "COINBASE_SPOT_SOL_USD", "COINBASE", "SOL", "USD"),
]
ASSETS_BY_KEY = {a.key: a for a in ASSETS}

# ============================================================
# CoinAPI EMS Live Execution Engine
# ============================================================

@dataclass
class LiveTrader:
    active_orders: Dict[str, dict] = field(default_factory=dict)
    execution_history: List[dict] = field(default_factory=list)

    def evaluate_signal(self, spec: AssetSpec, current_price: float, predicted_token: str) -> None:
        if not LIVE_TRADING_ENABLED:
            return
        
        current_pos = self.active_orders.get(spec.key)

        # Flip logic: Close existing position if signal reverses or flattens
        if current_pos is not None:
            should_exit = False
            if current_pos["side"] == "BUY" and predicted_token in ("down", "strong_down"):
                should_exit = True
            elif current_pos["side"] == "SELL" and predicted_token in ("up", "strong_up"):
                should_exit = True
            elif predicted_token == "flat":
                should_exit = True

            if should_exit:
                self._submit_order(spec, "SELL" if current_pos["side"] == "BUY" else "BUY", current_pos["amount"])
                self.active_orders.pop(spec.key, None)

        # Open entry logic
        if spec.key not in self.active_orders:
            if predicted_token in ("up", "strong_up"):
                qty = ORDER_AMOUNT_USD / current_price
                self._submit_order(spec, "BUY", qty)
                self.active_orders[spec.key] = {"side": "BUY", "amount": qty, "entry": current_price}
            elif predicted_token in ("down", "strong_down"):
                qty = ORDER_AMOUNT_USD / current_price
                self._submit_order(spec, "SELL", qty)
                self.active_orders[spec.key] = {"side": "SELL", "amount": qty, "entry": current_price}

    def _submit_order(self, spec: AssetSpec, side: str, quantity: float) -> Optional[dict]:
        url = f"{EMS_BASE_URL}/orders"
        payload = {
            "exchange_id": spec.exchange_id,
            "client_order_id": f"ngram-{int(time.time() * 1000)}",
            "symbol_id": spec.coinapi_symbol,
            "amount": round(quantity, 6),
            "price": 0.0,  # Market order execution
            "side": side,
            "order_type": "MARKET",
            "time_in_force": "IOC"
        }
        headers = {
            "X-CoinAPI-Key": COINAPI_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode("utf-8"), 
                headers=headers, 
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.execution_history.append(data)
                return data
        except Exception as e:
            self.execution_history.append({"error": str(e), "spec": spec.key})
            return None


live_portfolio = LiveTrader()

# ============================================================
# Market Data REST Fetching
# ============================================================

def _http_get(url: str, timeout: int = 10) -> str:
    headers = {
        "X-CoinAPI-Key": COINAPI_KEY,
        "Accept": "application/json",
        "User-Agent": "CoinAPILiveTrader/1.0",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_coinapi_history(symbol_id: str, days: int = HISTORY_DAYS) -> List[float]:
    try:
        end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        start_time = time.strftime(
            "%Y-%m-%dT%H:%M:%S", 
            time.gmtime(time.time() - (days * 24 * 60 * 60))
        )
        url = (
            f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"
            f"?period_id=1DAY&time_start={start_time}&time_end={end_time}"
        )
        data = json.loads(_http_get(url))
        if not isinstance(data, list) or len(data) < 5:
            raise ValueError("Invalid history")
        prices = [float(item["price_close"]) for item in data if "price_close" in item]
        return prices if len(prices) >= 5 else _synthetic_prices(days)
    except Exception:
        return _synthetic_prices(days)


def fetch_coinapi_price(symbol_id: str) -> Tuple[Optional[float], Optional[str]]:
    url = f"https://rest.coinapi.io/v1/trades/{symbol_id}/latest?limit=1"
    try:
        data = json.loads(_http_get(url))
        if isinstance(data, list) and len(data) > 0:
            return float(data[0]["price"]), None
        elif isinstance(data, dict) and "price" in data:
            return float(data["price"]), None
        return None, f"No price payload for {symbol_id}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _synthetic_prices(days: int) -> List[float]:
    price = 100.0
    out = []
    for _ in range(days):
        pct = random.gauss(0, 1.5) / 100.0
        price = max(price * (1 + pct), 1.0)
        out.append(price)
    return out


# ============================================================
# Tokenization & N-Gram Core
# ============================================================

def pct_change(prev: float, curr: float) -> float:
    return 0.0 if prev == 0 else (curr - prev) / prev * 100.0


def movement_token(change_pct: float) -> str:
    for label, upper in MOVE_THRESHOLDS:
        if change_pct < upper:
            return label
    return MOVE_THRESHOLDS[-1][0]


def prices_to_tokens(prices: List[float]) -> List[str]:
    return [movement_token(pct_change(a, b)) for a, b in zip(prices, prices[1:])]


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = 1
    influence_tau: float = 0.5
    curve_k: float = 8.0
    curve_midpoint: float = 0.5

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    def ingest_tokens(self, tokens: List[str]) -> None:
        if not tokens:
            return
        sequence = ["<bos>", "<bos>"] + tokens + [self.eos_token]
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)
        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count
        self.lexical_vectors = {t: {ctx: c / (sum(cnts.values()) or 1) for ctx, c in cnts.items()} 
                                for t, cnts in token_contexts.items()}
        self.influence_vectors = {}
        for source in self.vocabulary:
            src_vec = self.lexical_vectors.get(source, {})
            self.influence_vectors[source] = {
                target: sim for target in self.vocabulary if source != target
                and (sim := cosine_similarity(src_vec, self.lexical_vectors.get(target, {}))) >= self.influence_tau
            }
        self.finalized = True

    def sample_next(self, prev: str, prev_prev: Optional[str], temperature: float = 0.8, top_k: int = 5) -> str:
        if not self.finalized:
            self.finalize()
        counts = self.trigram.get(f"{prev_prev}\t{prev}") if prev_prev else None
        if not counts:
            counts = self.bigram.get(prev)
        if not counts:
            counts = self.unigram
        total = sum(counts.values())
        if not total:
            return self.eos_token
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]


@dataclass
class AssetRuntime:
    spec: AssetSpec
    price_history: List[float] = field(default_factory=list)
    model_holder: dict = field(default_factory=dict)
    current_price: Optional[float] = None
    predicted_token: Optional[str] = None
    status: str = "starting..."
    cycle_count: int = 0


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    assets: Dict[str, AssetRuntime] = field(default_factory=dict)


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()
BTC_KEY = "btc"


def run_cycle(rt: AssetRuntime) -> None:
    rt.cycle_count += 1
    current_price, error = fetch_coinapi_price(rt.spec.coinapi_symbol)
    with state.lock:
        if current_price is None:
            rt.status = f"fetch failed: {error}"
            return
        rt.status = "ok"
        rt.price_history.append(current_price)
        tokens = prices_to_tokens(rt.price_history)

        if rt.cycle_count % RETRAIN_EVERY_N_CYCLES == 0 or rt.model_holder.get("model") is None:
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            rt.model_holder["model"] = model

        model = rt.model_holder["model"]
        prev = tokens[-1] if tokens else "<bos>"
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        next_token = model.sample_next(prev, prev_prev)
        if next_token in ("<eos>", "<unk>"):
            next_token = "flat"

        rt.predicted_token = next_token
        rt.current_price = current_price

        # Fire live EMS order evaluation
        live_portfolio.evaluate_signal(rt.spec, current_price, next_token)


def background_loop() -> None:
    with state.lock:
        for spec in ASSETS:
            state.assets[spec.key] = AssetRuntime(spec=spec, status="fetching history...")
    for rt in state.assets.values():
        rt.price_history = fetch_coinapi_history(rt.spec.coinapi_symbol)
        rt.status = "starting..."

    while not stop_event.is_set():
        for rt in list(state.assets.values()):
            if stop_event.is_set():
                break
            run_cycle(rt)
        wake_event.wait(REFRESH_INTERVAL_SECONDS)
        wake_event.clear()


def make_icon_image(bucket: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = COLORS.get(bucket, COLORS["unknown"])
    if bucket == "up":
        draw.polygon([(32, 12), (52, 46), (12, 46)], fill=color)
    elif bucket == "down":
        draw.polygon([(12, 18), (52, 18), (32, 52)], fill=color)
    else:
        draw.rounded_rectangle([14, 26, 50, 38], radius=6, fill=color)
    return img


def menu_mode_text(icon: pystray.Icon) -> str:
    mode = "LIVE TRADING ON" if LIVE_TRADING_ENABLED else "LIVE TRADING DISABLED"
    return f"Mode: {mode}"


def menu_orders_text(icon: pystray.Icon) -> str:
    active = len(live_portfolio.active_orders)
    return f"Active EMS Positions: {active}"


def _make_line(key: str):
    def _getter(icon: pystray.Icon) -> str:
        with state.lock:
            rt = state.assets.get(key)
            if not rt or not rt.current_price:
                return f"{key}: loading..."
            pos = live_portfolio.active_orders.get(key)
            pos_tag = f" [{pos['side']} live]" if pos else ""
            return f"{rt.spec.label}: ${rt.current_price:,.2f} | next: {(rt.predicted_token or '--').upper()}{pos_tag}"
    return _getter


def main() -> None:
    icon = pystray.Icon(
        "coinapi_live_trader",
        make_icon_image("flat"),
        "CoinAPI Live Trader",
        menu=pystray.Menu(
            pystray.MenuItem(menu_mode_text, None, enabled=False),
            pystray.MenuItem(menu_orders_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            *([pystray.MenuItem(_make_line(spec.key), None, enabled=False) for spec in ASSETS]),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda i, item: (stop_event.set(), wake_event.set(), i.stop()))
        )
    )
    threading.Thread(target=background_loop, daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()COINAPI_KEY = os.environ.get("COINAPI_KEY", "YOUR_COINAPI_KEY_HERE")

MOVE_THRESHOLDS = [
    ("strong_down", -1.5),
    ("down", -0.3),
    ("flat", 0.3),
    ("up", 1.5),
    ("strong_up", math.inf),
]
MOVE_MIDPOINTS = {
    "strong_down": -2.5, "down": -0.8, "flat": 0.0, "up": 0.8, "strong_up": 2.5,
}
MOVE_COLOR_BUCKET = {
    "strong_down": "down", "down": "down",
    "flat": "flat",
    "up": "up", "strong_up": "up",
}
COLORS = {
    "up": (46, 160, 67),      # green
    "down": (219, 68, 55),    # red
    "flat": (128, 128, 128),  # gray
    "unknown": (100, 100, 100),
}

random.seed()

# ============================================================
# Asset universe using CoinAPI symbol IDs (v1/symbols)
# ============================================================

@dataclass(frozen=True)
class AssetSpec:
    key: str                # internal id, e.g. "btc", "eth"
    label: str              # display name, e.g. "BTC/USD", "ETH/USD"
    kind: str               # "crypto" | "forex"
    coinapi_symbol: str     # CoinAPI specific symbol ID (e.g., BITSTAMP_SPOT_BTC_USD)


ASSETS: List[AssetSpec] = [
    AssetSpec("btc", "BTC/USD", "crypto", "BITSTAMP_SPOT_BTC_USD"),
    AssetSpec("eth", "ETH/USD", "crypto", "BITSTAMP_SPOT_ETH_USD"),
    AssetSpec("sol", "SOL/USD", "crypto", "COINBASE_SPOT_SOL_USD"),
    AssetSpec("ada", "ADA/USD", "crypto", "COINBASE_SPOT_ADA_USD"),
    AssetSpec("dot", "DOT/USD", "crypto", "BINANCE_SPOT_DOT_USDT"),
    AssetSpec("link", "LINK/USD", "crypto", "BINANCE_SPOT_LINK_USDT"),
    AssetSpec("avax", "AVAX/USD", "crypto", "BINANCE_SPOT_AVAX_USDT"),
    AssetSpec("eurusd", "EUR/USD", "forex", "OANDA_FX_EUR_USD"),
    AssetSpec("gbpusd", "GBP/USD", "forex", "OANDA_FX_GBP_USD"),
    AssetSpec("usdjpy", "USD/JPY", "forex", "OANDA_FX_USD_JPY"),
    AssetSpec("audusd", "AUD/USD", "forex", "OANDA_FX_AUD_USD"),
]
ASSETS_BY_KEY = {a.key: a for a in ASSETS}

# ============================================================
# Data fetching via CoinAPI REST v1
# ============================================================

def _http_get(url: str, timeout: int = 10) -> str:
    headers = {
        "X-CoinAPI-Key": COINAPI_KEY,
        "Accept": "application/json",
        "User-Agent": "CoinAPITraderToy/1.0",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_coinapi_history(symbol_id: symbol_str, days: int = HISTORY_DAYS) -> List[float]:
    """Fetches historical OHLCV daily data from CoinAPI v1/ohlcv/{symbol_id}/history."""
    try:
        # Calculate start time dynamically
        end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        start_time = time.strftime(
            "%Y-%m-%dT%H:%M:%S", 
            time.gmtime(time.time() - (days * 24 * 60 * 60))
        )
        url = (
            f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history"
            f"?period_id=1DAY&time_start={start_time}&time_end={end_time}"
        )
        data = json.loads(_http_get(url))
        if not isinstance(data, list) or len(data) < 5:
            raise ValueError("Invalid or insufficient history returned from CoinAPI")
        
        prices = [float(item["price_close"]) for item in data if "price_close" in item]
        if len(prices) < 5:
            raise ValueError("Too few valid closing prices")
        return prices
    except Exception:
        return _synthetic_prices(days)


def fetch_coinapi_price(symbol_id: str) -> Tuple[Optional[float], Optional[str]]:
    """Polls latest trade or quote price via CoinAPI v1/trades/{symbol_id}/latest."""
    url = f"https://rest.coinapi.io/v1/trades/{symbol_id}/latest?limit=1"
    try:
        data = json.loads(_http_get(url))
        if isinstance(data, list) and len(data) > 0:
            return float(data[0]["price"]), None
        elif isinstance(data, dict) and "price" in data:
            return float(data["price"]), None
        return None, f"No price payload found for {symbol_id}"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} ({url})"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def fetch_latest_price(spec: AssetSpec) -> Tuple[Optional[float], Optional[str]]:
    return fetch_coinapi_price(spec.coinapi_symbol)


def fetch_initial_history(spec: AssetSpec) -> List[float]:
    return fetch_coinapi_history(spec.coinapi_symbol)


def _synthetic_prices(days: int) -> List[float]:
    price = 100.0
    out = []
    for _ in range(days):
        pct = random.gauss(0, 1.5) / 100.0
        price = max(price * (1 + pct), 1.0)
        out.append(price)
    return out


# ============================================================
# Tokenization & N-Gram Core
# ============================================================

def pct_change(prev: float, curr: float) -> float:
    if prev == 0:
        return 0.0
    return (curr - prev) / prev * 100.0


def movement_token(change_pct: float) -> str:
    for label, upper in MOVE_THRESHOLDS:
        if change_pct < upper:
            return label
    return MOVE_THRESHOLDS[-1][0]


def prices_to_tokens(prices: List[float]) -> List[str]:
    return [movement_token(pct_change(a, b)) for a, b in zip(prices, prices[1:])]


def bag_of_tokens(tokens: Iterable[str]) -> Counter:
    return Counter(t for t in tokens if t not in IGNORED_TOKENS)


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = 1
    influence_tau: float = 0.5
    curve_k: float = 8.0
    curve_midpoint: float = 0.5

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    def ingest_tokens(self, tokens: List[str]) -> None:
        if not tokens:
            return
        sequence = ["<bos>", "<bos>"] + tokens + [self.eos_token]
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in self.vocabulary:
            source_vec = self.lexical_vectors.get(source, {})
            scores = {}
            for target in self.vocabulary:
                if source == target:
                    continue
                sim = cosine_similarity(source_vec, self.lexical_vectors.get(target, {}))
                if sim >= self.influence_tau:
                    scores[target] = sim
            self.influence_vectors[source] = scores
        self.finalized = True

    def _backoff_distribution(self, prev: str, prev_prev: Optional[str]) -> Dict[str, float]:
        if prev_prev is not None:
            counts = self.trigram.get(f"{prev_prev}\t{prev}")
            if counts:
                return self._normalize(counts)
        counts = self.bigram.get(prev)
        if counts:
            return self._normalize(counts)
        return self._normalize(self.unigram)

    @staticmethod
    def _normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def _curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))
        z = self.curve_k * (p_eos - self.curve_midpoint)
        return 1.0 / (1.0 + math.exp(z))

    def _score_next_token(self, prev: str, prev_prev: Optional[str], candidate_limit: int = 64) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}
        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        curve = self._curve_weight(base.get(self.eos_token, 0.0))
        scores = {}
        for token in candidates:
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            scores[token] = safe_log(base[token]) + curve * 0.35 * similarity + curve * 0.65 * influence
        return scores

    def _probabilities(self, prev: str, prev_prev: Optional[str], temperature: float, candidate_limit: int) -> Dict[str, float]:
        scores = self._score_next_token(prev, prev_prev, candidate_limit)
        if not scores:
            return {}
        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(self, prev: str, prev_prev: Optional[str], temperature: float = 0.8, top_k: int = 5) -> str:
        probs = self._probabilities(prev, prev_prev, temperature, max(top_k, 1))
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]


FORECAST_DISPLAY_CAP_PCT = 30.0

def horizon_forecast(
    model: NGramModel,
    prev: str,
    prev_prev: Optional[str],
    current_price: float,
    steps: int = FORECAST_STEPS,
    temperature: float = 0.9,
    top_k: int = 5,
) -> Tuple[float, float, str]:
    probs = model._probabilities(prev, prev_prev, temperature, max(top_k, 1))
    if not probs:
        return current_price, 0.0, "flat"
    expected_pct_per_step = sum(MOVE_MIDPOINTS.get(t, 0.0) * p for t, p in probs.items())
    cumulative_pct = expected_pct_per_step * steps
    clamped_pct = max(-FORECAST_DISPLAY_CAP_PCT, min(FORECAST_DISPLAY_CAP_PCT, cumulative_pct))
    forecast_price = current_price * (1 + clamped_pct / 100.0)
    return forecast_price, clamped_pct, movement_token(clamped_pct)


@dataclass
class AssetRuntime:
    spec: AssetSpec
    price_history: List[float] = field(default_factory=list)
    model_holder: dict = field(default_factory=dict)
    current_price: Optional[float] = None
    predicted_token: Optional[str] = None
    predicted_price: Optional[float] = None
    forecast_24h_price: Optional[float] = None
    forecast_24h_pct: Optional[float] = None
    forecast_24h_token: Optional[str] = None
    correct_count: int = 0
    total_count: int = 0
    last_updated: Optional[str] = None
    status: str = "starting..."
    cycle_count: int = 0


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    assets: Dict[str, AssetRuntime] = field(default_factory=dict)


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()

BTC_KEY = "btc"


def run_cycle(rt: AssetRuntime) -> None:
    rt.cycle_count += 1
    current_price, error = fetch_latest_price(rt.spec)
    with state.lock:
        if current_price is None:
            rt.status = f"fetch failed: {error or 'unknown error'}"
            return
        rt.status = "ok"

        previous_price = rt.price_history[-1] if rt.price_history else None
        if previous_price is not None and rt.predicted_token is not None:
            rt.total_count += 1
            predicted_dir = rt.predicted_price - previous_price
            actual_dir = current_price - previous_price
            if predicted_dir * actual_dir >= 0:
                rt.correct_count += 1

        rt.price_history.append(current_price)
        tokens = prices_to_tokens(rt.price_history)

        if rt.cycle_count % RETRAIN_EXT_N_CYCLES == 0 or rt.model_holder.get("model") is None:
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            rt.model_holder["model"] = model

        model = rt.model_holder["model"]

        prev = tokens[-1] if tokens else "<bos>"
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        next_token = model.sample_next(prev, prev_prev)
        if next_token in ("<eos>", "<unk>"):
            next_token = "flat"

        rt.predicted_token = next_token
        rt.predicted_price = current_price * (1 + MOVE_MIDPOINTS.get(next_token, 0.0) / 100.0)
        rt.current_price = current_price
        rt.last_updated = time.strftime("%H:%M:%S")

        avg_price, pct, token = horizon_forecast(model, prev, prev_prev, current_price)
        rt.forecast_24h_price = avg_price
        rt.forecast_24h_pct = pct
        rt.forecast_24h_token = token


def background_loop() -> None:
    with state.lock:
        for spec in ASSETS:
            state.assets[spec.key] = AssetRuntime(spec=spec, status="fetching history...")

    for key, rt in state.assets.items():
        history = fetch_initial_history(rt.spec)
        with state.lock:
            rt.price_history = history
            rt.status = "starting..."

    while not stop_event.is_set():
        for rt in list(state.assets.values()):
            if stop_event.is_set():
                break
            run_cycle(rt)
        wake_event.wait(REFRESH_INTERVAL_SECONDS)
        wake_event.clear()


def make_icon_image(bucket: str) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = COLORS.get(bucket, COLORS["unknown"])

    cx, cy = size // 2, size // 2
    if bucket == "up":
        draw.polygon([(cx, 12), (52, 46), (12, 46)], fill=color)
    elif bucket == "down":
        draw.polygon([(12, 18), (52, 18), (cx, 52)], fill=color)
    else:
        draw.rounded_rectangle([14, 26, 50, 38], radius=6, fill=color)
    return img


def _format_price(price: float, kind: str) -> str:
    if kind == "forex":
        return f"{price:,.4f}"
    if price < 10000:
        return f"${price:,.2f}"
    return f"${price:,.0f}"


def _forecast_str(rt: AssetRuntime) -> str:
    if rt.forecast_24h_price is None:
        return "24h lean: n/a"
    price_str = _format_price(rt.forecast_24h_price, rt.spec.kind)
    sign = "+" if rt.forecast_24h_pct >= 0 else ""
    return f"24h lean: {price_str} ({sign}{rt.forecast_24h_pct:.1f}%, {rt.forecast_24h_token.upper()})"


def _asset_line(rt: AssetRuntime) -> str:
    if rt.current_price is None:
        return f"{rt.spec.label}: {rt.status}"
    price_str = _format_price(rt.current_price, rt.spec.kind)
    token = (rt.predicted_token or "--").upper()
    acc_str = "n/a" if rt.total_count == 0 else f"{100.0 * rt.correct_count / rt.total_count:.0f}%"
    return f"{rt.spec.label}: {price_str} | next: {token} | {_forecast_str(rt)} | acc: {acc_str}"


def menu_price_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        if rt is None or rt.current_price is None:
            return f"BTC: {rt.status if rt else 'starting...'}"
        return f"BTC: ${rt.current_price:,.0f}"


def menu_prediction_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        if rt is None or rt.predicted_token is None:
            return "Next move: --"
        return f"Next move: {rt.predicted_token.upper()} (toy model)"


def menu_forecast_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        return _forecast_str(rt) if rt else "24h lean: n/a"


def menu_accuracy_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        if not rt or rt.total_count == 0:
            return "Accuracy: n/a"
        acc = 100.0 * rt.correct_count / rt.total_count
        return f"Accuracy: {acc:.0f}% (n={rt.total_count})"


def menu_updated_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        return f"Updated: {(rt.last_updated if rt else None) or '--'}"


def _make_asset_line_getter(key: str):
    def _getter(icon: pystray.Icon) -> str:
        with state.lock:
            rt = state.assets.get(key)
            return _asset_line(rt) if rt else f"{key}: n/a"
    return _getter


def _build_flat_rows_for_kinds(kinds: Tuple[str, ...]) -> List[pystray.MenuItem]:
    return [
        pystray.MenuItem(_make_asset_line_getter(spec.key), None, enabled=False)
        for spec in ASSETS
        if spec.kind in kinds
    ]


def build_crypto_submenu() -> List[pystray.MenuItem]:
    return _build_flat_rows_for_kinds(("crypto",))


def build_forex_submenu() -> List[pystray.MenuItem]:
    return _build_flat_rows_for_kinds(("forex",))


def on_refresh(icon: pystray.Icon, item) -> None:
    wake_event.set()


def on_quit(icon: pystray.Icon, item) -> None:
    stop_event.set()
    wake_event.set()
    icon.visible = False
    icon.stop()


def tray_update_loop(icon: pystray.Icon) -> None:
    last_bucket = None
    while not stop_event.is_set():
        with state.lock:
            rt = state.assets.get(BTC_KEY)
            token = rt.predicted_token if rt else None
            price = rt.current_price if rt else None
        bucket = MOVE_COLOR_BUCKET.get(token, "flat") if token else "flat"
        if bucket != last_bucket:
            icon.icon = make_icon_image(bucket)
            last_bucket = bucket
        if price is not None:
            icon.title = f"BTC ${price:,.0f} — next: {(token or '?').upper()} (CoinAPI)"
        icon.update_menu()
        time.sleep(2)


def main() -> None:
    icon = pystray.Icon(
        "coinapi_predictor",
        make_icon_image("flat"),
        "CoinAPI Trader Toy (starting...)",
        menu=pystray.Menu(
            pystray.MenuItem(menu_price_text, None, enabled=False),
            pystray.MenuItem(menu_prediction_text, None, enabled=False),
            pystray.MenuItem(menu_forecast_text, None, enabled=False),
            pystray.MenuItem(menu_accuracy_text, None, enabled=False),
            pystray.MenuItem(menu_updated_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("-- Altcoins & Crypto --", None, enabled=False),
            *build_crypto_submenu(),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("-- Forex Pairs --", None, enabled=False),
            *build_forex_submenu(),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh now", on_refresh),
            pystray.MenuItem("Quit", on_quit),
        ),
    )

    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()

    ui_thread = threading.Thread(target=tray_update_loop, args=(icon,), daemon=True)
    ui_thread.start()

    icon.run()


if __name__ == "__main__":
    main()
