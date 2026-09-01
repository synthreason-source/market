from __future__ import annotations

import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw
import pystray

# ============================================================
# DISCLAIMER
# ============================================================
# Toy statistical exercise, not a trading system. No genuine
# predictive edge on BTC, stocks, or index prices. The "model"
# is an n-gram toy over discretized percent-change tokens.
# ============================================================

REFRESH_INTERVAL_SECONDS = 1 * 30   # how often to poll price + re-predict
HISTORY_DAYS = 90                    # initial training window
RETRAIN_EVERY_N_CYCLES = 45          # retrain the model every N refreshes

FORECAST_HORIZON_SECONDS = 24 * 60 * 60
FORECAST_STEPS = max(1, FORECAST_HORIZON_SECONDS // REFRESH_INTERVAL_SECONDS)  # 2,880 at 30s cycles

IGNORED_TOKENS = {"<bos>", "<eos>", "<unk>"}

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
# Asset universe: BTC (via CoinGecko) + stocks/indices/forex
# (via Yahoo Finance's public chart endpoint)
# ============================================================
# Yahoo's endpoint needs no signup/API key, unlike Stooq (which now
# gates its CSV feed behind a manually-issued apikey) or most other
# free-tier providers (Alpha Vantage, Twelve Data, Finnhub), which
# all require registering for a key. It's still an unofficial,
# scraped endpoint though, so it can rate-limit or break without
# notice — the code falls back to synthetic data if it does.

@dataclass(frozen=True)
class AssetSpec:
    key: str            # internal id, e.g. "btc", "aapl"
    label: str          # display name, e.g. "BTC/USD", "Apple (AAPL)"
    kind: str           # "crypto" | "stock" | "index" | "forex"
    stooq_symbol: str = ""  # Yahoo Finance ticker (name kept for minimal diff; unused for crypto)


ASSETS: List[AssetSpec] = [
    AssetSpec("btc", "BTC/USD", "crypto"),
    # Individual stocks
    AssetSpec("aapl", "Apple (AAPL)", "stock", "AAPL"),
    AssetSpec("msft", "Microsoft (MSFT)", "stock", "MSFT"),
    AssetSpec("googl", "Alphabet (GOOGL)", "stock", "GOOGL"),
    AssetSpec("amzn", "Amazon (AMZN)", "stock", "AMZN"),
    AssetSpec("nvda", "Nvidia (NVDA)", "stock", "NVDA"),
    AssetSpec("tsla", "Tesla (TSLA)", "stock", "TSLA"),
    # Indices
    AssetSpec("spx", "S&P 500", "index", "^GSPC"),
    AssetSpec("dji", "Dow Jones", "index", "^DJI"),
    AssetSpec("ndq", "Nasdaq Composite", "index", "^IXIC"),
    # Currency pairs (forex)
    AssetSpec("eurusd", "EUR/USD", "forex", "EURUSD=X"),
    AssetSpec("gbpusd", "GBP/USD", "forex", "GBPUSD=X"),
    AssetSpec("usdjpy", "USD/JPY", "forex", "USDJPY=X"),
    AssetSpec("audusd", "AUD/USD", "forex", "AUDUSD=X"),
    AssetSpec("usdcad", "USD/CAD", "forex", "USDCAD=X"),
    AssetSpec("usdchf", "USD/CHF", "forex", "USDCHF=X"),
]
ASSETS_BY_KEY = {a.key: a for a in ASSETS}

# ============================================================
# Data fetching
# ============================================================

HTTP_HEADERS = {
    # Both CoinGecko and Stooq will 403 a bare urllib request with no
    # User-Agent — this is the #1 cause of silent "fetch failed" errors.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}


def _http_get(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_btc_history(days: int = HISTORY_DAYS, vs_currency: str = "usd") -> List[float]:
    """Initial training history for BTC. Falls back to a synthetic walk offline."""
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            f"?vs_currency={vs_currency}&days={days}&interval=daily"
        )
        data = json.loads(_http_get(url))
        prices = [float(p) for _, p in data["prices"]]
        if len(prices) < 10:
            raise ValueError("too few points")
        return prices
    except Exception:
        return _synthetic_prices(days)


def fetch_btc_price(vs_currency: str = "usd") -> Tuple[Optional[float], Optional[str]]:
    """Cheap single-price poll for BTC. Returns (price, error_message)."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={vs_currency}"
    try:
        data = json.loads(_http_get(url))
        return float(data["bitcoin"][vs_currency]), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} ({url})"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def fetch_yahoo_json(symbol: str, params: str) -> dict:
    encoded = quote(symbol, safe="")  # "^" in ^GSPC etc. and "=" in EURUSD=X must be encoded
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?{params}"
    return json.loads(_http_get(url)), url


def fetch_yahoo_history(yahoo_symbol: str, days: int = HISTORY_DAYS) -> List[float]:
    """Initial daily-close training history for a stock/index/forex pair via Yahoo Finance."""
    try:
        data, _ = fetch_yahoo_json(yahoo_symbol, "range=6mo&interval=1d")
        result = data["chart"]["result"][0]
        closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
        if len(closes) < 10:
            raise ValueError("too few points")
        return closes[-days:]
    except Exception:
        return _synthetic_prices(days)


def fetch_yahoo_price(yahoo_symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Cheap latest-quote poll via Yahoo Finance. Returns (price, error_message)."""
    url = ""
    try:
        data, url = fetch_yahoo_json(yahoo_symbol, "range=1d&interval=1m")
        result = data["chart"]["result"][0]
        price = result.get("meta", {}).get("regularMarketPrice")
        if price is None:
            # fall back to the last non-null close in today's intraday series
            closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
            if not closes:
                return None, f"no price in response (symbol: {yahoo_symbol})"
            price = closes[-1]
        return float(price), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} for symbol '{yahoo_symbol}' ({url})"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def fetch_latest_price(spec: AssetSpec) -> Tuple[Optional[float], Optional[str]]:
    if spec.kind == "crypto":
        return fetch_btc_price()
    return fetch_yahoo_price(spec.stooq_symbol)


def fetch_initial_history(spec: AssetSpec) -> List[float]:
    if spec.kind == "crypto":
        return fetch_btc_history()
    return fetch_yahoo_history(spec.stooq_symbol)


def _synthetic_prices(days: int) -> List[float]:
    price = 100.0
    out = []
    for _ in range(days):
        pct = random.gauss(0, 1.5) / 100.0
        price = max(price * (1 + pct), 1.0)
        out.append(price)
    return out


# ============================================================
# Tokenization
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


# ============================================================
# N-gram model over movement tokens
# ============================================================

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


FORECAST_DISPLAY_CAP_PCT = 30.0  # clamp for legibility only — see horizon_forecast()

def horizon_forecast(
    model: NGramModel,
    prev: str,
    prev_prev: Optional[str],
    current_price: float,
    steps: int = FORECAST_STEPS,
    temperature: float = 0.9,
    top_k: int = 5,
) -> Tuple[float, float, str]:
    """Project `steps` cycles ahead using the model's *expected* per-cycle
    move, scaled linearly, rather than literally simulating each step.

    Why not just chain sample_next() 2,880 times and average the endpoints?
    Tried that first — it doesn't work. Compounding thousands of small
    multiplicative percent moves has a built-in downward drift even when the
    moves are symmetric (Jensen's inequality / "volatility drag": +10% then
    -10% nets -1%, not 0%, and that erosion compounds over thousands of
    steps). Every simulated path decays this way, so no amount of averaging
    or taking the median across runs fixes it — it's baked into each path,
    not an artifact of aggregation. A test run reliably forecast BTC at
    -97% over 24h regardless of the model's actual bias, which is wrong.

    This version instead computes E[pct move per cycle] directly from the
    next-token probability distribution (the same one sample_next() draws
    from) and multiplies by the step count — no compounding artifact.

    But this trade fixes one problem and surfaces another: linearly
    stretching a 30-second-scale signal across 2,880 steps means any tiny
    per-cycle lean gets amplified enormously — test runs on synthetic data
    swung from +970% to -3699% depending on the random seed, purely from
    noise in a single cycle's probability estimate. That's not a usable "24h
    price target." The result is clamped to +/-FORECAST_DISPLAY_CAP_PCT for
    legibility, and callers should treat it as a *directional lean*
    (which way, roughly how strongly) rather than a literal forecast price —
    it's labeled "24h lean" in the UI for exactly this reason.
    """
    probs = model._probabilities(prev, prev_prev, temperature, max(top_k, 1))
    if not probs:
        return current_price, 0.0, "flat"
    expected_pct_per_step = sum(MOVE_MIDPOINTS.get(t, 0.0) * p for t, p in probs.items())
    cumulative_pct = expected_pct_per_step * steps
    clamped_pct = max(-FORECAST_DISPLAY_CAP_PCT, min(FORECAST_DISPLAY_CAP_PCT, cumulative_pct))
    forecast_price = current_price * (1 + clamped_pct / 100.0)
    return forecast_price, clamped_pct, movement_token(clamped_pct)


# ============================================================
# Per-asset runtime state
# ============================================================

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


# ============================================================
# Shared app state (written by the background thread, read by
# the tray icon's menu callbacks — guarded by a lock)
# ============================================================

@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    assets: Dict[str, AssetRuntime] = field(default_factory=dict)


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()  # set to trigger an immediate refresh

BTC_KEY = "btc"  # the asset that drives the tray icon itself


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

        retrained = False
        if rt.cycle_count % RETRAIN_EVERY_N_CYCLES == 0 or rt.model_holder.get("model") is None:
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            rt.model_holder["model"] = model
            retrained = True
        else:
            pass  # reuse the existing model between retrains

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

        # The horizon forecast is a closed-form expectation (see
        # horizon_forecast's docstring for why), so it's cheap enough to
        # recompute every cycle rather than only at retrain time.
        avg_price, pct, token = horizon_forecast(model, prev, prev_prev, current_price)
        rt.forecast_24h_price = avg_price
        rt.forecast_24h_pct = pct
        rt.forecast_24h_token = token


def background_loop() -> None:
    # Set up runtime + initial history for every tracked asset.
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


# ============================================================
# Tray icon
# ============================================================

def make_icon_image(bucket: str) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = COLORS.get(bucket, COLORS["unknown"])

    cx, cy = size // 2, size // 2
    if bucket == "up":
        draw.polygon([(cx, 12), (52, 46), (12, 46)], fill=color)      # up triangle
    elif bucket == "down":
        draw.polygon([(12, 18), (52, 18), (cx, 52)], fill=color)      # down triangle
    else:
        draw.rounded_rectangle([14, 26, 50, 38], radius=6, fill=color)  # flat bar
    return img


def _format_price(price: float, kind: str) -> str:
    if kind == "forex":
        return f"{price:,.4f}"  # e.g. 1.0842, 151.3200 — no $ sign, it's a pair rate
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
    """Single-line summary: price, next-cycle move, 24h forecast, and running
    accuracy — all shown directly on the menu row, no submenu click required."""
    if rt.current_price is None:
        return f"{rt.spec.label}: {rt.status}"
    price_str = _format_price(rt.current_price, rt.spec.kind)
    token = (rt.predicted_token or "--").upper()
    if rt.total_count == 0:
        acc_str = "n/a"
    else:
        acc_str = f"{100.0 * rt.correct_count / rt.total_count:.0f}%"
    return f"{rt.spec.label}: {price_str} | next: {token} | {_forecast_str(rt)} | acc: {acc_str}"


def _asset_accuracy_line(rt: AssetRuntime) -> str:
    if rt.total_count == 0:
        return "Accuracy: n/a"
    acc = 100.0 * rt.correct_count / rt.total_count
    return f"Accuracy: {acc:.0f}% (n={rt.total_count})"


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
        if rt is None:
            return "24h lean: n/a"
        return _forecast_str(rt)


def menu_accuracy_text(icon: pystray.Icon) -> str:
    with state.lock:
        rt = state.assets.get(BTC_KEY)
        return _asset_accuracy_line(rt) if rt else "Accuracy: n/a"


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
    """Flat, always-visible menu rows (no submenu) for the given asset kinds."""
    return [
        pystray.MenuItem(_make_asset_line_getter(spec.key), None, enabled=False)
        for spec in ASSETS
        if spec.kind in kinds
    ]


def build_stock_submenu() -> List[pystray.MenuItem]:
    return _build_flat_rows_for_kinds(("stock", "index"))


def build_currency_submenu() -> List[pystray.MenuItem]:
    return _build_flat_rows_for_kinds(("forex",))


def on_refresh(icon: pystray.Icon, item) -> None:
    wake_event.set()


def on_quit(icon: pystray.Icon, item) -> None:
    stop_event.set()
    wake_event.set()
    icon.visible = False
    icon.stop()


def tray_update_loop(icon: pystray.Icon) -> None:
    """Keeps the icon image/tooltip in sync with BTC's state."""
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
            icon.title = f"BTC ${price:,.0f} — next: {(token or '?').upper()} (not financial advice)"
        icon.update_menu()
        time.sleep(2)


def main() -> None:
    icon = pystray.Icon(
        "btc_predictor",
        make_icon_image("flat"),
        "BTC + Stocks predictor (starting...)",
        menu=pystray.Menu(
            pystray.MenuItem(menu_price_text, None, enabled=False),
            pystray.MenuItem(menu_prediction_text, None, enabled=False),
            pystray.MenuItem(menu_forecast_text, None, enabled=False),
            pystray.MenuItem(menu_accuracy_text, None, enabled=False),
            pystray.MenuItem(menu_updated_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("-- Stocks & Indices --", None, enabled=False),
            *build_stock_submenu(),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("-- Currencies --", None, enabled=False),
            *build_currency_submenu(),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh now", on_refresh),
            pystray.MenuItem("Quit", on_quit),
        ),
    )

    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()

    ui_thread = threading.Thread(target=tray_update_loop, args=(icon,), daemon=True)
    ui_thread.start()

    icon.run()  # blocks; must run on the main thread


if __name__ == "__main__":
    main()
