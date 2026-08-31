from __future__ import annotations

import json
import math
import random
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw
import pystray

# ============================================================
# DISCLAIMER
# ============================================================
# Toy statistical exercise, not a trading system. See the
# original desktop script's disclaimer — same caveats apply:
# no genuine predictive edge on BTC price.
# ============================================================

REFRESH_INTERVAL_SECONDS = 1 * 60   # how often to poll price + re-predict
HISTORY_DAYS = 90                    # initial training window
RETRAIN_EVERY_N_CYCLES = 30          # retrain the model every N refreshes

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
# Which tray-icon color bucket each token belongs to.
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
# Data fetching
# ============================================================

def fetch_btc_history(days: int = HISTORY_DAYS, vs_currency: str = "usd") -> List[float]:
    """Initial training history. Falls back to a synthetic walk offline."""
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            f"?vs_currency={vs_currency}&days={days}&interval=daily"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prices = [float(p) for _, p in data["prices"]]
        if len(prices) < 10:
            raise ValueError("too few points")
        return prices
    except Exception:
        return _synthetic_prices(days)


def fetch_latest_price(vs_currency: str = "usd") -> Optional[float]:
    """Cheap single-price poll, used every refresh cycle."""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={vs_currency}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return float(data["bitcoin"][vs_currency])
    except Exception:
        return None


def _synthetic_prices(days: int) -> List[float]:
    price = 60000.0
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
# N-gram model over movement tokens (same core as desktop script)
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


# ============================================================
# Shared app state (written by the background thread, read by
# the tray icon's menu callbacks — guarded by a lock)
# ============================================================

@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_price: Optional[float] = None
    predicted_token: Optional[str] = None
    predicted_price: Optional[float] = None
    correct_count: int = 0
    total_count: int = 0
    last_updated: Optional[str] = None
    status: str = "starting..."


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()  # set to trigger an immediate refresh


def run_cycle(
    price_history: List[float],
    model_holder: dict,
    cycle_count: int,
) -> None:
    current_price = fetch_latest_price()
    with state.lock:
        if current_price is None:
            state.status = "fetch failed, retrying next cycle"
            return
        state.status = "ok"

        previous_price = price_history[-1] if price_history else None
        if previous_price is not None and state.predicted_token is not None:
            state.total_count += 1
            predicted_dir = state.predicted_price - previous_price
            actual_dir = current_price - previous_price
            if predicted_dir * actual_dir >= 0:
                state.correct_count += 1

        price_history.append(current_price)
        tokens = prices_to_tokens(price_history)

        # Retrain only every RETRAIN_EVERY_N_CYCLES
        if cycle_count % RETRAIN_EVERY_N_CYCLES == 0 or model_holder.get("model") is None:
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            model_holder["model"] = model
        else:
            # Optionally, you could update some online stats here if desired.
            # For now, we just reuse the existing model.
            pass

        model = model_holder["model"]

        prev = tokens[-1] if tokens else "<bos>"
        prev_prev = tokens[-2] if len(tokens) >= 2 else None
        next_token = model.sample_next(prev, prev_prev)
        if next_token in ("<eos>", "<unk>"):
            next_token = "flat"

        state.predicted_token = next_token
        state.predicted_price = current_price * (1 + MOVE_MIDPOINTS.get(next_token, 0.0) / 100.0)
        state.current_price = current_price
        state.last_updated = time.strftime("%H:%M:%S")


def background_loop() -> None:
    with state.lock:
        state.status = "fetching history..."
    price_history = fetch_btc_history()
    model_holder: dict = {}
    cycle_count = 0

    while not stop_event.is_set():
        cycle_count += 1
        run_cycle(price_history, model_holder, cycle_count)
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


def menu_price_text(icon: pystray.Icon) -> str:
    with state.lock:
        if state.current_price is None:
            return f"BTC: {state.status}"
        return f"BTC: ${state.current_price:,.0f}"


def menu_prediction_text(icon: pystray.Icon) -> str:
    with state.lock:
        if state.predicted_token is None:
            return "Next move: --"
        return f"Next move: {state.predicted_token.upper()} (toy model)"


def menu_accuracy_text(icon: pystray.Icon) -> str:
    with state.lock:
        if state.total_count == 0:
            return "Accuracy: n/a"
        acc = 100.0 * state.correct_count / state.total_count
        return f"Accuracy: {acc:.0f}% (n={state.total_count})"


def menu_updated_text(icon: pystray.Icon) -> str:
    with state.lock:
        return f"Updated: {state.last_updated or '--'}"


def on_refresh(icon: pystray.Icon, item) -> None:
    wake_event.set()


def on_quit(icon: pystray.Icon, item) -> None:
    stop_event.set()
    wake_event.set()
    icon.visible = False
    icon.stop()


def tray_update_loop(icon: pystray.Icon) -> None:
    """Keeps the icon image/tooltip in sync with app state."""
    last_bucket = None
    while not stop_event.is_set():
        with state.lock:
            token = state.predicted_token
            price = state.current_price
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
        "BTC predictor (starting...)",
        menu=pystray.Menu(
            pystray.MenuItem(menu_price_text, None, enabled=False),
            pystray.MenuItem(menu_prediction_text, None, enabled=False),
            pystray.MenuItem(menu_accuracy_text, None, enabled=False),
            pystray.MenuItem(menu_updated_text, None, enabled=False),
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
