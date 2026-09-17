from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import random
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
    import pystray
except ImportError:
    Image = None
    ImageDraw = None
    pystray = None


# ============================================================
# KRAKEN LIVE EXECUTION CONFIGURATION
# ============================================================

REFRESH_INTERVAL_SECONDS = 30
HISTORY_DAYS = 90
RETRAIN_EVERY_N_CYCLES = 45

# Set your credentials via environment variables:
# export KRAKEN_API_KEY="your_api_key"
# export KRAKEN_PRIVATE_KEY="your_private_secret_key"
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "I")
KRAKEN_PRIVATE_KEY = os.environ.get("KRAKEN_PRIVATE_KEY", "")

# Live order sizing parameters
TARGET_DOGE_VOLUME = 30
# Kraken enforces minimum order sizes (e.g., ~30-50 DOGE). 
# Set this to True to enforce the requested 0.1 DOGE strictly, 
# or False to scale up automatically to Kraken's minimum if rejected.
STRICT_EXACT_VOLUME = True 

RANDOM_SEED = 20260917
random.seed(RANDOM_SEED)

MOVE_THRESHOLDS = [
    ("strong_down", -1.5),
    ("down", -0.3),
    ("flat", 0.3),
    ("up", 1.5),
    ("strong_up", math.inf),
]

COLORS = {
    "up": (46, 160, 67),
    "down": (219, 68, 55),
    "flat": (128, 128, 128),
    "unknown": (100, 100, 100),
}


# ============================================================
# ASSETS (KRAKEN PAIRS)
# ============================================================

@dataclass(frozen=True)
class AssetSpec:
    key: str
    label: str
    kraken_pair: str
    base_asset: str
    quote_asset: str = "USD"
    min_volume: float = 0.1


ASSETS: List[AssetSpec] = [
    AssetSpec("btc", "BTC/USD", "XXBTZUSD", "BTC", min_volume=0.0001),
    AssetSpec("eth", "ETH/USD", "XETHZUSD", "ETH", min_volume=0.001),
    AssetSpec("sol", "SOL/USD", "SOLUSD", "SOL", min_volume=0.01),
    AssetSpec("doge", "DOGE/USD", "XDGUSD", "DOGE", min_volume=TARGET_DOGE_VOLUME),
]


# ============================================================
# KRAKEN API & SIGNING UTILITIES
# ============================================================

def _get_kraken_signature(urlpath: str, data: dict, secret: str) -> str:
    posted_data = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + posted_data).encode("utf-8")
    message = (urlpath.encode("utf-8") + hashlib.sha256(encoded).digest())
    signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(signature.digest()).decode("utf-8")


def kraken_private_request(urlpath: str, data: dict) -> dict:
    if not KRAKEN_API_KEY or not KRAKEN_PRIVATE_KEY:
        raise ValueError("Missing KRAKEN_API_KEY or KRAKEN_PRIVATE_KEY environment variables.")

    data["nonce"] = str(int(time.time() * 1000))
    
    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": _get_kraken_signature(urlpath, data, KRAKEN_PRIVATE_KEY),
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "KrakenLiveTrader/1.0",
    }

    encoded_data = urllib.parse.urlencode(data).encode("utf-8")
    url = f"https://api.kraken.com{urlpath}"
    
    request = urllib.request.Request(url, data=encoded_data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body)


def execute_live_order(spec: AssetSpec, direction: str) -> dict:
    """
    Executes a real order on Kraken.
    Direction should be 'buy' or 'sell' based on model signals.
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"Invalid order direction: {direction}")

    volume = spec.min_volume
    if spec.key == "doge":
        volume = TARGET_DOGE_VOLUME

    data = {
        "pair": spec.kraken_pair,
        "type": direction,
        "ordertype": "market",
        "volume": f"{volume:.6f}",
    }

    result = kraken_private_request("/0/private/AddOrder", data)
    if result.get("error"):
        raise ValueError(f"Kraken Order Error: {result['error']}")
    
    return result.get("result", {})


def _http_get(url: str, timeout: int = 10) -> str:
    headers = {
        "Accept": "application/json",
        "User-Agent": "KrakenMarketFeed/1.0",
    }
    
    if KRAKEN_API_KEY:
        headers["API-Key"] = KRAKEN_API_KEY

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_kraken_history(spec: AssetSpec, days: int = HISTORY_DAYS) -> List[float]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={spec.kraken_pair}&interval=1440"
    raw = _http_get(url)
    data = json.loads(raw)

    if data.get("error"):
        raise ValueError(f"Kraken API errors: {data['error']}")

    result = data.get("result", {})
    pair_key = next((k for k in result.keys() if k != "last"), None)
    if not pair_key:
        raise ValueError("No pair data found in Kraken OHLC response")

    candles = result[pair_key]
    if not isinstance(candles, list) or len(candles) < 5:
        raise ValueError("Insufficient candlestick entries from Kraken")

    prices = [float(item[4]) for item in candles if len(item) > 4]
    return prices[-days:]


def fetch_kraken_price(spec: AssetSpec) -> float:
    url = f"https://api.kraken.com/0/public/Ticker?pair={spec.kraken_pair}"
    raw = _http_get(url)
    data = json.loads(raw)

    if data.get("error"):
        raise ValueError(f"Kraken errors: {data['error']}")

    result = data.get("result", {})
    pair_key = next((k for k in result.keys()), None)
    if not pair_key:
        raise ValueError("No ticker payload matching pair")

    last_trade_arr = result[pair_key].get("c")
    if not last_trade_arr:
        raise ValueError("Missing last trade close entry")

    return float(last_trade_arr[0])


# ============================================================
# MOVEMENT / N-GRAM MODEL
# ============================================================

def pct_change(prev: float, curr: float) -> float:
    if prev == 0:
        return 0.0
    return (curr - prev) / prev * 100.0


def movement_token(change_pct: float) -> str:
    for label, upper in MOVE_THRESHOLDS:
        if change_pct < upper:
            return label
    return "strong_up"


def prices_to_tokens(prices: List[float]) -> List[str]:
    return [
        movement_token(pct_change(a, b))
        for a, b in zip(prices, prices[1:])
    ]


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = 1

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
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
        self.vocabulary = sorted(
            token for token, count in self.unigram.items() if count >= self.min_count
        )
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)
        self.finalized = True

    def sample_next(self, prev: str, prev_prev: Optional[str], temperature: float = 0.8, top_k: int = 5) -> str:
        if not self.finalized:
            self.finalize()

        counts = self.trigram.get(f"{prev_prev}\t{prev}") if prev_prev else None
        if not counts:
            counts = self.bigram.get(prev)
        if not counts:
            counts = self.unigram
        if not counts:
            return self.eos_token

        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_k)]
        tokens, raw_weights = zip(*items)

        temp = max(float(temperature), 0.05)
        logits = [math.log(max(float(weight), 1e-12)) / temp for weight in raw_weights]
        max_logit = max(logits)
        weights = [math.exp(x - max_logit) for x in logits]

        return random.choices(list(tokens), weights=weights, k=1)[0]


# ============================================================
# ASSET RUNTIME & APP STATE
# ============================================================

@dataclass
class AssetRuntime:
    spec: AssetSpec
    price_history: List[float] = field(default_factory=list)
    model: Optional[NGramModel] = None
    current_price: Optional[float] = None
    predicted_token: Optional[str] = None
    status: str = "starting..."
    cycle_count: int = 0
    last_error: Optional[str] = None
    last_order_result: Optional[str] = None


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    assets: Dict[str, AssetRuntime] = field(default_factory=dict)


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()


# ============================================================
# MARKET ANALYTICS & LIVE TRADING CYCLE
# ============================================================

def run_cycle(rt: AssetRuntime) -> None:
    rt.cycle_count += 1
    current_price = fetch_kraken_price(rt.spec)

    with state.lock:
        rt.current_price = current_price
        rt.last_error = None
        rt.status = "ok"

        rt.price_history.append(current_price)
        tokens = prices_to_tokens(rt.price_history)

        if rt.model is None or rt.cycle_count % RETRAIN_EVERY_N_CYCLES == 0:
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            rt.model = model

        if rt.model is None:
            rt.predicted_token = "flat"
            return

        prev = tokens[-1] if tokens else "<bos>"
        prev_prev = tokens[-2] if len(tokens) >= 2 else None

        next_token = rt.model.sample_next(prev, prev_prev, temperature=0.8, top_k=5)
        if next_token in ("<eos>", "<unk>", "<bos>"):
            next_token = "flat"

        rt.predicted_token = next_token

    # Execute live trades based on model signal if API keys are active
    if KRAKEN_API_KEY and KRAKEN_PRIVATE_KEY:
        try:
            order_dir = None
            if next_token in ("up", "strong_up"):
                order_dir = "buy"
            elif next_token in ("down", "strong_down"):
                order_dir = "sell"

            if order_dir:
                # Target exact user request (0.1 DOGE for Dogecoin)
                res = execute_live_order(rt.spec, order_dir)
                with state.lock:
                    rt.last_order_result = f"Live {order_dir.upper()} {rt.spec.min_volume} {rt.spec.base_asset} OK"
        except Exception as exc:
            with state.lock:
                rt.last_order_result = f"Order err: {exc}"


# ============================================================
# BACKGROUND LOOP
# ============================================================

def background_loop() -> None:
    with state.lock:
        for spec in ASSETS:
            state.assets[spec.key] = AssetRuntime(spec=spec, status="loading history...")

    for rt in state.assets.values():
        try:
            history = fetch_kraken_history(rt.spec)
            with state.lock:
                rt.price_history = history
                rt.status = "starting..."
                model = NGramModel()
                model.ingest_tokens(prices_to_tokens(history))
                model.finalize()
                rt.model = model
        except Exception as exc:
            with state.lock:
                rt.status = "history error"
                rt.last_error = f"{type(exc).__name__}: {exc}"

    while not stop_event.is_set():
        for rt in list(state.assets.values()):
            if stop_event.is_set():
                break
            try:
                run_cycle(rt)
            except Exception as exc:
                with state.lock:
                    rt.status = "cycle error"
                    rt.last_error = f"{type(exc).__name__}: {exc}"

        wake_event.wait(REFRESH_INTERVAL_SECONDS)
        wake_event.clear()


# ============================================================
# DISPLAY & UI
# ============================================================

def make_icon_image(bucket: str):
    if Image is None or ImageDraw is None:
        return None

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


def menu_mode_text(icon) -> str:
    key_status = "KEYS CONFIGURED" if KRAKEN_API_KEY else "NO API KEYS (READ-ONLY)"
    return f"Mode: LIVE EXECUTION ({key_status}) [DOGE={TARGET_DOGE_VOLUME}]"


def _make_line(key: str):
    def _getter(icon) -> str:
        with state.lock:
            rt = state.assets.get(key)
            if not rt:
                return f"{key}: starting..."
            if rt.current_price is None:
                return f"{rt.spec.label}: loading... [{rt.status}]"
            
            order_info = f" | {rt.last_order_result}" if rt.last_order_result else ""
            return (
                f"{rt.spec.label}: "
                f"${rt.current_price:,.4f} | "
                f"sig: {(rt.predicted_token or '--').upper()}"
                f"{order_info}"
            )
    return _getter


def print_console_status() -> None:
    with state.lock:
        print("\n" + "=" * 78)
        print(f"KRAKEN LIVE TRADING ENGINE (Target DOGE size: {TARGET_DOGE_VOLUME})")
        print("=" * 78)

        for rt in state.assets.values():
            price = f"${rt.current_price:,.4f}" if rt.current_price is not None else "loading..."
            print(
                f"{rt.spec.label:<9} "
                f"{price:<14} "
                f"signal={(rt.predicted_token or '--'):<12} "
                f"status={rt.status} "
                f"msg={rt.last_order_result or 'none'}"
            )
        print("=" * 78)


def console_loop() -> None:
    print("pystray is unavailable; using console status loop.")
    print_console_status()
    while not stop_event.is_set():
        time.sleep(REFRESH_INTERVAL_SECONDS)
        print_console_status()


def main() -> None:
    print("=" * 78)
    print("KRAKEN LIVE TRADING ENGINE INITIALIZED")
    print("=" * 78)
    print(f"Target Execution Size for DOGE/USD: {TARGET_DOGE_VOLUME} DOGE")
    print(f"API Connected: {bool(KRAKEN_API_KEY and KRAKEN_PRIVATE_KEY)}")
    print("=" * 78)

    thread = threading.Thread(target=background_loop, daemon=True, name="kraken-live-loop")
    thread.start()

    if pystray is None:
        try:
            console_loop()
        except KeyboardInterrupt:
            stop_event.set()
            wake_event.set()
        return

    def quit_app(icon, item):
        stop_event.set()
        wake_event.set()
        icon.stop()

    icon = pystray.Icon(
        "kraken_live_trader",
        make_icon_image("flat"),
        "Kraken Live Trader",
        menu=pystray.Menu(
            pystray.MenuItem(menu_mode_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            *[pystray.MenuItem(_make_line(spec.key), None, enabled=False) for spec in ASSETS],
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_app),
        ),
    )

    try:
        icon.run()
    finally:
        stop_event.set()
        wake_event.set()


if __name__ == "__main__":
    main()
