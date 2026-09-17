from __future__ import annotations

import json
import math
import os
import random
import threading
import time
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
# PAPER / LOCAL MONEY SIMULATION
# ============================================================
# This version NEVER submits real orders.
#
# It can use CoinAPI market data when COINAPI_KEY is present,
# but every trade is simulated locally with virtual money.
#
# Default starting capital: $1,000 USD.
# ============================================================

REFRESH_INTERVAL_SECONDS = 30
HISTORY_DAYS = 90
RETRAIN_EVERY_N_CYCLES = 45

COINAPI_KEY = os.environ.get("COINAPI_KEY", "")

# Real execution is deliberately unavailable in this simulation.
LIVE_TRADING_ENABLED = False

INITIAL_LOCAL_CASH = 1000.00
SIMULATED_ORDER_USD = 10.00
SIMULATED_FEE_RATE = 0.001
ALLOW_SHORTS = True

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
# ASSETS
# ============================================================

@dataclass(frozen=True)
class AssetSpec:
    key: str
    label: str
    coinapi_symbol: str
    base_asset: str
    quote_asset: str = "USD"


ASSETS: List[AssetSpec] = [
    AssetSpec("btc", "BTC/USD", "BITSTAMP_SPOT_BTC_USD", "BTC"),
    AssetSpec("eth", "ETH/USD", "BITSTAMP_SPOT_ETH_USD", "ETH"),
    AssetSpec("sol", "SOL/USD", "COINBASE_SPOT_SOL_USD", "SOL"),
]


# ============================================================
# LOCAL PAPER PORTFOLIO
# ============================================================

@dataclass
class SimPosition:
    side: str
    amount: float
    entry_price: float
    entry_value: float
    opened_at: float = field(default_factory=time.time)


@dataclass
class SimulatedPortfolio:
    starting_cash: float = INITIAL_LOCAL_CASH
    cash: float = INITIAL_LOCAL_CASH
    positions: Dict[str, SimPosition] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)

    def open_position(
        self,
        spec: AssetSpec,
        side: str,
        price: float,
        notional: float = SIMULATED_ORDER_USD,
    ) -> bool:
        if price <= 0 or notional <= 0:
            return False

        if spec.key in self.positions:
            return False

        fee = notional * SIMULATED_FEE_RATE

        if side == "BUY":
            total_cost = notional + fee
            if self.cash < total_cost:
                return False
            self.cash -= total_cost
        elif side == "SELL":
            if not ALLOW_SHORTS:
                return False
            # A simulated short receives the sale proceeds but reserves
            # the same notional as collateral, so cash does not become
            # artificially inflated.
            if self.cash < fee:
                return False
            self.cash -= fee
        else:
            return False

        amount = notional / price

        self.positions[spec.key] = SimPosition(
            side=side,
            amount=amount,
            entry_price=price,
            entry_value=notional,
        )

        self.history.append({
            "time": time.time(),
            "asset": spec.label,
            "action": "OPEN",
            "side": side,
            "price": price,
            "amount": amount,
            "notional": notional,
            "fee": fee,
        })
        return True

    def close_position(self, spec: AssetSpec, price: float) -> Optional[float]:
        position = self.positions.pop(spec.key, None)
        if position is None or price <= 0:
            return None

        exit_value = position.amount * price
        entry_value = position.entry_value
        fee = exit_value * SIMULATED_FEE_RATE

        if position.side == "BUY":
            pnl = exit_value - entry_value - fee
            self.cash += exit_value - fee
        else:
            pnl = entry_value - exit_value - fee
            self.cash += pnl

        self.history.append({
            "time": time.time(),
            "asset": spec.label,
            "action": "CLOSE",
            "side": position.side,
            "price": price,
            "amount": position.amount,
            "entry_price": position.entry_price,
            "entry_value": entry_value,
            "exit_value": exit_value,
            "fee": fee,
            "pnl": pnl,
        })
        return pnl

    def position_value(self, spec: AssetSpec, price: float) -> float:
        position = self.positions.get(spec.key)
        if position is None:
            return 0.0

        if position.side == "BUY":
            return position.amount * price
        return position.entry_value - (position.amount * price - position.entry_value)

    def unrealized_pnl(self, spec: AssetSpec, price: float) -> float:
        position = self.positions.get(spec.key)
        if position is None:
            return 0.0

        if position.side == "BUY":
            return position.amount * price - position.entry_value

        return position.entry_value - position.amount * price

    def total_equity(self, prices: Dict[str, float]) -> float:
        equity = self.cash

        for key, position in self.positions.items():
            price = prices.get(key)
            if price is None:
                continue

            if position.side == "BUY":
                equity += position.amount * price
            else:
                # Short equity = cash + unrealized short P/L.
                equity += self.unrealized_pnl(
                    AssetSpec(key, key.upper(), "", key),
                    price,
                )

        return equity

    def summary(self, prices: Dict[str, float]) -> str:
        equity = self.total_equity(prices)
        pnl = equity - self.starting_cash
        pct = (pnl / self.starting_cash * 100.0) if self.starting_cash else 0.0

        return (
            f"LOCAL PAPER MONEY  |  "
            f"cash=${self.cash:,.2f}  |  "
            f"equity=${equity:,.2f}  |  "
            f"P/L=${pnl:+,.2f} ({pct:+.2f}%)  |  "
            f"positions={len(self.positions)}"
        )


paper_portfolio = SimulatedPortfolio()


# ============================================================
# MARKET DATA
# ============================================================

def _http_get(url: str, timeout: int = 10) -> str:
    if not COINAPI_KEY:
        raise RuntimeError("COINAPI_KEY is not set")

    request = urllib.request.Request(
        url,
        headers={
            "X-CoinAPI-Key": COINAPI_KEY,
            "Accept": "application/json",
            "User-Agent": "LocalPaperTrader/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def synthetic_price_history(
    symbol: str,
    days: int = HISTORY_DAYS,
) -> List[float]:
    """
    Deterministic local market simulation.
    Each asset gets a different starting price and volatility.
    """
    seeds = {
        "BITSTAMP_SPOT_BTC_USD": (100000.0, 2.0),
        "BITSTAMP_SPOT_ETH_USD": (4000.0, 2.4),
        "COINBASE_SPOT_SOL_USD": (180.0, 3.0),
    }

    price, volatility = seeds.get(symbol, (100.0, 2.0))
    local_rng = random.Random(
        RANDOM_SEED + sum(ord(c) for c in symbol)
    )

    out = []

    # Add a slow regime component so the simulation isn't simply flat noise.
    trend = local_rng.uniform(-0.15, 0.15)

    for i in range(days):
        if i % 20 == 0:
            trend = local_rng.uniform(-0.20, 0.20)

        pct = trend + local_rng.gauss(0.0, volatility)
        price = max(price * (1.0 + pct / 100.0), 0.01)
        out.append(price)

    return out


def fetch_coinapi_history(
    spec: AssetSpec,
    days: int = HISTORY_DAYS,
) -> Tuple[List[float], str]:
    if not COINAPI_KEY:
        return synthetic_price_history(spec.coinapi_symbol, days), "SIMULATED"

    try:
        end_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        start_time = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.gmtime(time.time() - days * 86400),
        )

        url = (
            f"https://rest.coinapi.io/v1/ohlcv/{spec.coinapi_symbol}/history"
            f"?period_id=1DAY"
            f"&time_start={start_time}"
            f"&time_end={end_time}"
        )

        data = json.loads(_http_get(url))

        if not isinstance(data, list):
            raise ValueError("CoinAPI history was not a list")

        prices = [
            float(item["price_close"])
            for item in data
            if "price_close" in item
        ]

        if len(prices) < 5:
            raise ValueError("Insufficient historical prices")

        return prices, "COINAPI"

    except Exception:
        return synthetic_price_history(spec.coinapi_symbol, days), "SIMULATED"


def fetch_coinapi_price(
    spec: AssetSpec,
    fallback_price: Optional[float] = None,
) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Returns:
        price,
        source: COINAPI / SIMULATED,
        error
    """
    if not COINAPI_KEY:
        return (
            fallback_price if fallback_price is not None else 100.0,
            "SIMULATED",
            "COINAPI_KEY not configured",
        )

    url = (
        f"https://rest.coinapi.io/v1/trades/"
        f"{spec.coinapi_symbol}/latest?limit=1"
    )

    try:
        data = json.loads(_http_get(url))

        if isinstance(data, list) and data:
            return float(data[0]["price"]), "COINAPI", None

        if isinstance(data, dict) and "price" in data:
            return float(data["price"]), "COINAPI", None

        raise ValueError("No price payload")

    except Exception as exc:
        if fallback_price is not None:
            return (
                fallback_price,
                "SIMULATED",
                f"live fetch failed: {type(exc).__name__}: {exc}",
            )

        return None, "ERROR", f"{type(exc).__name__}: {exc}"


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


def cosine_similarity(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = 1

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    trigram: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    lexical_vectors: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
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

        for a, b, c in zip(
            sequence,
            sequence[1:],
            sequence[2:],
        ):
            self.trigram[f"{a}\t{b}"][c] += 1

        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(
            token
            for token, count in self.unigram.items()
            if count >= self.min_count
        )

        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)

        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {
            token: {
                ctx: count / (sum(counts.values()) or 1)
                for ctx, count in counts.items()
            }
            for token, counts in token_contexts.items()
        }

        self.finalized = True

    def sample_next(
        self,
        prev: str,
        prev_prev: Optional[str],
        temperature: float = 0.8,
        top_k: int = 5,
    ) -> str:
        if not self.finalized:
            self.finalize()

        counts = (
            self.trigram.get(f"{prev_prev}\t{prev}")
            if prev_prev
            else None
        )

        if not counts:
            counts = self.bigram.get(prev)

        if not counts:
            counts = self.unigram

        if not counts:
            return self.eos_token

        items = sorted(
            counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:max(1, top_k)]

        tokens, raw_weights = zip(*items)

        # Temperature-aware sampling.
        temp = max(float(temperature), 0.05)
        logits = [
            math.log(max(float(weight), 1e-12)) / temp
            for weight in raw_weights
        ]

        max_logit = max(logits)
        weights = [
            math.exp(x - max_logit)
            for x in logits
        ]

        return random.choices(
            list(tokens),
            weights=weights,
            k=1,
        )[0]


# ============================================================
# ASSET RUNTIME
# ============================================================

@dataclass
class AssetRuntime:
    spec: AssetSpec
    price_history: List[float] = field(default_factory=list)
    model: Optional[NGramModel] = None
    current_price: Optional[float] = None
    predicted_token: Optional[str] = None
    status: str = "starting..."
    data_source: str = "SIMULATED"
    cycle_count: int = 0
    last_error: Optional[str] = None
    last_pnl: Optional[float] = None


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    assets: Dict[str, AssetRuntime] = field(default_factory=dict)


state = AppState()
stop_event = threading.Event()
wake_event = threading.Event()


# ============================================================
# SIMULATION ENGINE
# ============================================================

def simulate_signal(
    rt: AssetRuntime,
    predicted_token: str,
) -> None:
    """
    Local paper execution only.

    UP / STRONG_UP -> open BUY
    DOWN / STRONG_DOWN -> open SELL
    FLAT -> close any position

    A position is never sent to an exchange.
    """
    price = rt.current_price
    if price is None:
        return

    spec = rt.spec
    existing = paper_portfolio.positions.get(spec.key)

    if existing is not None:
        reverse = (
            existing.side == "BUY"
            and predicted_token in ("down", "strong_down")
        ) or (
            existing.side == "SELL"
            and predicted_token in ("up", "strong_up")
        )

        if reverse or predicted_token == "flat":
            rt.last_pnl = paper_portfolio.close_position(spec, price)
            existing = None

    if existing is None:
        if predicted_token in ("up", "strong_up"):
            paper_portfolio.open_position(
                spec,
                "BUY",
                price,
                SIMULATED_ORDER_USD,
            )

        elif predicted_token in ("down", "strong_down"):
            paper_portfolio.open_position(
                spec,
                "SELL",
                price,
                SIMULATED_ORDER_USD,
            )


def run_cycle(rt: AssetRuntime) -> None:
    rt.cycle_count += 1

    fallback = (
        rt.price_history[-1]
        if rt.price_history
        else None
    )

    current_price, source, error = fetch_coinapi_price(
        rt.spec,
        fallback_price=fallback,
    )

    with state.lock:
        if current_price is None:
            rt.status = "price unavailable"
            rt.last_error = error
            return

        rt.current_price = current_price
        rt.data_source = source
        rt.last_error = error
        rt.status = "ok" if source == "COINAPI" else "simulation"

        rt.price_history.append(current_price)

        tokens = prices_to_tokens(rt.price_history)

        if (
            rt.model is None
            or rt.cycle_count % RETRAIN_EVERY_N_CYCLES == 0
        ):
            model = NGramModel()
            model.ingest_tokens(tokens)
            model.finalize()
            rt.model = model

        if rt.model is None:
            rt.predicted_token = "flat"
            return

        prev = tokens[-1] if tokens else "<bos>"
        prev_prev = tokens[-2] if len(tokens) >= 2 else None

        next_token = rt.model.sample_next(
            prev,
            prev_prev,
            temperature=0.8,
            top_k=5,
        )

        if next_token in ("<eos>", "<unk>", "<bos>"):
            next_token = "flat"

        rt.predicted_token = next_token

        # This is the only "execution" path.
        # It modifies the local paper portfolio only.
        simulate_signal(rt, next_token)


# ============================================================
# BACKGROUND LOOP
# ============================================================

def background_loop() -> None:
    with state.lock:
        for spec in ASSETS:
            state.assets[spec.key] = AssetRuntime(
                spec=spec,
                status="loading history...",
            )

    for rt in state.assets.values():
        history, source = fetch_coinapi_history(rt.spec)

        with state.lock:
            rt.price_history = history
            rt.data_source = source
            rt.status = "starting..."

            model = NGramModel()
            model.ingest_tokens(prices_to_tokens(history))
            model.finalize()
            rt.model = model

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
# DISPLAY
# ============================================================

def portfolio_text() -> str:
    with state.lock:
        prices = {
            key: rt.current_price
            for key, rt in state.assets.items()
            if rt.current_price is not None
        }

        return paper_portfolio.summary(prices)


def make_icon_image(bucket: str):
    if Image is None or ImageDraw is None:
        return None

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = COLORS.get(bucket, COLORS["unknown"])

    if bucket == "up":
        draw.polygon(
            [(32, 12), (52, 46), (12, 46)],
            fill=color,
        )
    elif bucket == "down":
        draw.polygon(
            [(12, 18), (52, 18), (32, 52)],
            fill=color,
        )
    else:
        draw.rounded_rectangle(
            [14, 26, 50, 38],
            radius=6,
            fill=color,
        )

    return img


def menu_mode_text(icon) -> str:
    return "Mode: LOCAL PAPER SIMULATION — REAL ORDERS DISABLED"


def menu_money_text(icon) -> str:
    return portfolio_text()


def _make_line(key: str):
    def _getter(icon) -> str:
        with state.lock:
            rt = state.assets.get(key)

            if not rt:
                return f"{key}: starting..."

            if rt.current_price is None:
                return f"{rt.spec.label}: loading... [{rt.status}]"

            pos = paper_portfolio.positions.get(key)

            if pos:
                pos_tag = (
                    f" [{pos.side} PAPER "
                    f"${pos.entry_value:.2f}]"
                )
            else:
                pos_tag = " [no paper position]"

            source = rt.data_source.upper()

            return (
                f"{rt.spec.label}: "
                f"${rt.current_price:,.2f} | "
                f"next: {(rt.predicted_token or '--').upper()} | "
                f"{source}{pos_tag}"
            )

    return _getter


def print_console_status() -> None:
    with state.lock:
        print("\n" + "=" * 78)
        print("LOCAL PAPER TRADING SIMULATION")
        print("=" * 78)
        print(portfolio_text())

        for rt in state.assets.values():
            price = (
                f"${rt.current_price:,.2f}"
                if rt.current_price is not None
                else "loading..."
            )

            position = paper_portfolio.positions.get(rt.spec.key)
            position_text = (
                f"{position.side} @ ${position.entry_price:,.2f}"
                if position
                else "NONE"
            )

            print(
                f"{rt.spec.label:<8} "
                f"{price:<14} "
                f"signal={(rt.predicted_token or '--'):<12} "
                f"source={rt.data_source:<9} "
                f"position={position_text}"
            )

        print("=" * 78)


# ============================================================
# CONSOLE-ONLY FALLBACK
# ============================================================

def console_loop() -> None:
    print("pystray is unavailable; using console simulation.")
    print_console_status()

    while not stop_event.is_set():
        time.sleep(REFRESH_INTERVAL_SECONDS)
        print_console_status()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("LOCAL PAPER TRADER")
    print("=" * 78)
    print("REAL TRADING: DISABLED")
    print(f"Starting virtual cash: ${INITIAL_LOCAL_CASH:,.2f}")
    print(f"Virtual order size:   ${SIMULATED_ORDER_USD:,.2f}")
    print(f"Virtual fee:          {SIMULATED_FEE_RATE * 100:.3f}%")
    print(
        "Market source:        "
        + ("CoinAPI + simulated fallback"
           if COINAPI_KEY
           else "SIMULATED LOCAL MARKET")
    )
    print("=" * 78)

    thread = threading.Thread(
        target=background_loop,
        daemon=True,
        name="paper-market-loop",
    )
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
        "local_paper_trader",
        make_icon_image("flat"),
        "Local Paper Trader",
        menu=pystray.Menu(
            pystray.MenuItem(
                menu_mode_text,
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                menu_money_text,
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            *[
                pystray.MenuItem(
                    _make_line(spec.key),
                    None,
                    enabled=False,
                )
                for spec in ASSETS
            ],
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Quit",
                quit_app,
            ),
        ),
    )

    try:
        icon.run()
    finally:
        stop_event.set()
        wake_event.set()


if __name__ == "__main__":
    main()
