"""
Kraken-Integrated Probabilistic Crypto Knapsack Trader
======================================================

Single-file implementation that:

    - Fetches market data (CoinGecko for simulation, Kraken for live).
    - Runs the probabilistic knapsack optimizer.
    - Submits orders to Kraken (dry-run or live).
    - Cancels stale orders.
    - Monitors positions.
    - Refreshes after every cycle.
    - Prints status each cycle.

IMPORTANT
---------
This code can place real orders if run with --live and the correct
environment variables. Start in dry-run mode and verify behavior.

Install:
    pip install requests numpy

Run (dry-run):
    python kraken_knapsack_trader.py --interval 900 --max-cycles 5

Run (live):
    export KRAKEN_API_KEY="your_public_key"
    export KRAKEN_API_SECRET="your_base64_private_key"

    python kraken_knapsack_trader.py \
        --live \
        --confirm-live \
        --interval 900

Configuration is controlled via command-line arguments.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import logging
import math
import os
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import requests


# ============================================================
# GLOBAL LOGGING
# ============================================================

LOG = logging.getLogger("kraken_knapsack")


# ============================================================
# KRAKEN CLIENT (EMBEDDED)
# ============================================================

class KrakenError(RuntimeError):
    pass


class KrakenSpotClient:
    BASE_URL = "https://api.kraken.com"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout: float = 20.0,
    ):
        self.api_key = ""
        self.api_secret = ""
        self.timeout = timeout
        self.session = requests.Session()
        self._nonce_lock = threading.Lock()
        self._last_nonce = 0

    def _nonce(self) -> str:
        with self._nonce_lock:
            value = max(time.time_ns() // 1_000_000, self._last_nonce + 1)
            self._last_nonce = value
            return str(value)

    def _sign(self, path: str, data: dict[str, Any]) -> str:
        encoded = requests.models.RequestEncodingMixin._encode_params(data)
        message = path.encode() + hashlib.sha256(
            (data["nonce"] + encoded).encode()
        ).digest()

        secret = base64.b64decode(self.api_secret)
        return base64.b64encode(
            hmac.new(secret, message, hashlib.sha512).digest()
        ).decode()

    def private(self, endpoint: str, data: dict[str, Any] | None = None):
        path = f"/0/private/{endpoint}"
        payload = dict(data or {})
        payload["nonce"] = self._nonce()

        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(path, payload),
        }

        response = self.session.post(
            self.BASE_URL + path,
            data=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        body = response.json()

        if body.get("error"):
            raise KrakenError("; ".join(body["error"]))

        return body["result"]

    def public(self, endpoint: str, data: dict[str, Any] | None = None):
        response = self.session.get(
            f"{self.BASE_URL}/0/public/{endpoint}",
            params=data or {},
            timeout=self.timeout,
        )
        response.raise_for_status()

        body = response.json()

        if body.get("error"):
            raise KrakenError("; ".join(body["error"]))

        return body["result"]

    def system_status(self):
        return self.public("SystemStatus")

    def asset_pairs(self):
        return self.public("AssetPairs")

    def ticker(self, pair: str):
        return self.public("Ticker", {"pair": pair})

    def ohlc(self, pair: str, interval: int = 60):
        return self.public(
            "OHLC",
            {"pair": pair, "interval": interval},
        )

    def balance(self):
        return self.private("Balance")

    def open_orders(self):
        return self.private("OpenOrders")

    def closed_orders(self):
        return self.private("ClosedOrders")

    def trades_history(self):
        return self.private("TradesHistory")

    def add_order(
        self,
        pair: str,
        side: str,
        ordertype: str,
        volume: str,
        price: str | None = None,
        validate: bool = True,
        userref: str | None = None,
        close_ordertype: str | None = None,
        close_price: str | None = None,
        close_price2: str | None = None,
    ):
        data: dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": volume,
        }

        if price is not None:
            data["price"] = price

        if validate:
            data["validate"] = "true"

        if userref is not None:
            data["userref"] = userref

        if close_ordertype is not None:
            data["close[ordertype]"] = close_ordertype

        if close_price is not None:
            data["close[price]"] = close_price

        if close_price2 is not None:
            data["close[price2]"] = close_price2

        return self.private("AddOrder", data)

    def cancel_order(self, txid: str):
        return self.private("CancelOrder", {"txid": txid})

    def cancel_all(self):
        return self.private("CancelAll")

    def cancel_all_after(self, timeout_seconds: int = 60):
        return self.private(
            "CancelAllOrdersAfter",
            {"timeout": timeout_seconds},
        )


# ============================================================
# TRADER / EXECUTION LAYER
# ============================================================

@dataclass
class PlannedOrder:
    pair: str
    side: str
    ordertype: str
    volume: str
    price: str
    stop_price: str
    tp_price: str


class Trader:
    def __init__(self, client: KrakenSpotClient, live: bool = False):
        self.client = client
        self.live = live

    def submit(self, order: PlannedOrder):
        payload = {
            "pair": order.pair,
            "side": order.side,
            "ordertype": order.ordertype,
            "volume": order.volume,
            "price": order.price,
            "stop_price": order.stop_price,
            "tp_price": order.tp_price,
        }

        if not self.live:
            LOG.warning("DRY RUN order: %s", payload)
            return {"dry_run": True, "order": payload}

        result = self.client.add_order(
            pair=order.pair,
            side=order.side,
            ordertype=order.ordertype,
            volume=order.volume,
            price=order.price,
            validate=False,
        )

        LOG.info("Kraken order submitted: %s", result)
        return result


# ============================================================
# OPTIMIZER DATA STRUCTURES
# ============================================================

@dataclass
class Coin:
    rank: int
    coin_id: str
    symbol: str
    name: str

    price: float
    market_cap: float
    volume_24h: float

    change_24h: float
    change_7d: float
    change_30d: float

    momentum: float
    liquidity_score: float
    affinity: float
    probability: float

    buy_limit: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float

    allocation: float
    quantity: float

    worst_case_loss: float

    profit_tp1: float
    profit_tp2: float
    profit_tp3: float

    expected_profit: float
    risk_reward: float

    # Kraken-specific fields
    kraken_pair: str = ""
    base_asset: str = ""
    quote_asset: str = ""
    price_decimals: int = 8
    volume_decimals: int = 8
    order_minimum: float = 0.0
    fee_rate: float = 0.004


@dataclass
class Candidate:
    indices: List[int]
    capital_used: float
    expected_profit: float
    worst_case_loss: float
    score: float


# ============================================================
# NUMERICAL HELPERS
# ============================================================
def decimal_places(value: Any, default: int = 8) -> int:
    """
    Convert Kraken's precision metadata to an integer.
    """

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def round_down_decimal(
    value: float,
    decimals: int,
) -> str:
    """
    Round down without ever increasing the order volume.
    """

    if value <= 0:
        return "0"

    decimals = max(0, int(decimals))
    quantum = Decimal("1").scaleb(-decimals)

    rounded = Decimal(str(value)).quantize(
        quantum,
        rounding=ROUND_DOWN,
    )

    return format(rounded, "f")


def normalise_pair_name(value: str) -> str:
    """
    Convert Kraken and CoinGecko symbols into a comparable form.
    """

    value = str(value).upper().strip()

    replacements = {
        "XBT": "BTC",
        "XDG": "DOGE",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return "".join(
        character
        for character in value
        if character.isalnum()
    )
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    x = clamp(x, -60.0, 60.0)

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def normalize_array(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)

    if len(x) == 0:
        return x

    finite = np.isfinite(x)

    if not finite.any():
        return np.full(len(x), 0.5, dtype=np.float64)

    valid = x[finite]

    lo = float(np.min(valid))
    hi = float(np.max(valid))

    if abs(hi - lo) < 1e-12:
        return np.full(len(x), 0.5, dtype=np.float64)

    out = (x - lo) / (hi - lo)

    out[~np.isfinite(out)] = 0.5

    return np.clip(out, 0.0, 1.0)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kraken-integrated probabilistic crypto knapsack trader."
    )

    # Optimizer arguments
    parser.add_argument(
        "--coins",
        type=int,
        default=100,
        help="Number of coins to retrieve.",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=50.0,
        help="Total capital available.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of probabilistic candidate portfolios.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Probability temperature.",
    )

    parser.add_argument(
        "--buy-pullback",
        type=float,
        default=0.02,
        help="Entry pullback from current price.",
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop distance below entry.",
    )

    parser.add_argument(
        "--tp1",
        type=float,
        default=0.05,
        help="Take-profit 1 distance above entry.",
    )

    parser.add_argument(
        "--tp2",
        type=float,
        default=0.10,
        help="Take-profit 2 distance above entry.",
    )

    parser.add_argument(
        "--tp3",
        type=float,
        default=0.20,
        help="Take-profit 3 distance above entry.",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum number of simultaneous positions.",
    )

    parser.add_argument(
        "--risk-budget",
        type=float,
        default=5.0,
        help="Maximum total modeled stop-loss risk.",
    )

    parser.add_argument(
        "--min-profit",
        type=float,
        default=1.0,
        help="Minimum modeled portfolio profit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="crypto_knapsack_results.csv",
        help="CSV output filename.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds.",
    )

    # Trading / cycle arguments
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Cycle interval in seconds.",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (requires --confirm-live).",
    )

    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Confirm understanding of live trading risks.",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximum number of cycles (0 = unlimited).",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.coins < 1:
        raise ValueError("--coins must be >= 1")

    if args.coins > 250:
        raise ValueError(
            "CoinGecko /coins/markets supports up to 250 coins per request."
        )

    if args.capital <= 0:
        raise ValueError("--capital must be > 0")

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    if not (0 <= args.buy_pullback < 1):
        raise ValueError("--buy-pullback must satisfy 0 <= x < 1")

    if not (0 < args.stop_loss < 1):
        raise ValueError("--stop-loss must satisfy 0 < x < 1")

    if not (0 < args.tp1 < args.tp2 < args.tp3):
        raise ValueError(
            "Take-profit levels must satisfy 0 < TP1 < TP2 < TP3."
        )

    if args.max_positions < 1:
        raise ValueError("--max-positions must be >= 1")

    if args.risk_budget < 0:
        raise ValueError("--risk-budget must be >= 0")

    if args.min_profit < 0:
        raise ValueError("--min-profit must be >= 0")

    if args.interval < 10:
        raise ValueError("--interval must be >= 10 seconds")

    if args.max_cycles < 0:
        raise ValueError("--max-cycles must be >= 0")


def live_enabled(args) -> bool:
    return (
        args.live
        and args.confirm_live
    )


# ============================================================
# MARKET DATA
# ============================================================

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"


def fetch_market_data(
    number_of_coins: int,
    timeout: float,
) -> list[dict]:
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": min(number_of_coins, 250),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d,30d",
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "CryptoKnapsackTrader/1.0",
    }

    response = requests.get(
        COINGECKO_URL,
        params=params,
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError("Unexpected API response.")

    return data


# ============================================================
# PROBABILITY MODEL
# ============================================================

def calculate_affinity(
    momentum: float,
    liquidity: float,
    market_cap_score: float,
) -> float:
    momentum_component = momentum - 0.5
    liquidity_component = liquidity - 0.5
    market_cap_component = market_cap_score - 0.5

    affinity = (
        0.55 * momentum_component
        + 0.30 * liquidity_component
        + 0.15 * market_cap_component
    )

    return float(affinity)


def affinity_to_probability(
    affinity: float,
    temperature: float,
) -> float:
    temperature = max(temperature, 1e-9)
    scaled = 8.0 * affinity / temperature
    return float(sigmoid(scaled))


# ============================================================
# COIN CONSTRUCTION
# ============================================================

def build_coins(
    market_data: list[dict],
    capital: float,
    buy_pullback: float,
    stop_loss_pct: float,
    tp1_pct: float,
    tp2_pct: float,
    tp3_pct: float,
    max_positions: int,
    temperature: float,
) -> list[Coin]:
    valid_rows = []

    for row in market_data:
        price = safe_float(row.get("current_price"))

        if price <= 0:
            continue

        market_cap = safe_float(row.get("market_cap"))
        volume = safe_float(row.get("total_volume"))

        change_24h = safe_float(
            row.get("price_change_percentage_24h_in_currency")
        )
        change_7d = safe_float(
            row.get("price_change_percentage_7d_in_currency")
        )
        change_30d = safe_float(
            row.get("price_change_percentage_30d_in_currency")
        )

        valid_rows.append(
            (
                row,
                price,
                market_cap,
                volume,
                change_24h,
                change_7d,
                change_30d,
            )
        )

    if not valid_rows:
        raise RuntimeError("No valid market data returned.")

    market_caps = [
        math.log1p(max(x[2], 0.0))
        for x in valid_rows
    ]

    volumes = [
        math.log1p(max(x[3], 0.0))
        for x in valid_rows
    ]

    momentum_raw = []

    for x in valid_rows:
        change_24h = x[4]
        change_7d = x[5]
        change_30d = x[6]

        momentum = (
            0.25 * change_24h
            + 0.35 * change_7d
            + 0.40 * change_30d
        )

        momentum_raw.append(momentum)

    market_cap_score = normalize_array(market_caps)
    liquidity_score = normalize_array(volumes)
    momentum_score = normalize_array(momentum_raw)

    position_allocation = capital / max_positions

    coins: list[Coin] = []

    for i, row_data in enumerate(valid_rows):
        (
            row,
            price,
            market_cap,
            volume,
            change_24h,
            change_7d,
            change_30d,
        ) = row_data

        momentum = float(momentum_score[i])
        liquidity = float(liquidity_score[i])
        cap_score = float(market_cap_score[i])

        affinity = calculate_affinity(
            momentum,
            liquidity,
            cap_score,
        )

        probability = affinity_to_probability(
            affinity,
            temperature,
        )

        buy_limit = price * (1.0 - buy_pullback)
        stop_price = buy_limit * (1.0 - stop_loss_pct)

        tp1 = buy_limit * (1.0 + tp1_pct)
        tp2 = buy_limit * (1.0 + tp2_pct)
        tp3 = buy_limit * (1.0 + tp3_pct)

        allocation = min(position_allocation, capital)
        quantity = allocation / buy_limit

        profit_tp1 = quantity * (tp1 - buy_limit)
        profit_tp2 = quantity * (tp2 - buy_limit)
        profit_tp3 = quantity * (tp3 - buy_limit)

        worst_case_loss = quantity * (buy_limit - stop_price)

        expected_profit = (
            0.25 * profit_tp1
            + 0.35 * profit_tp2
            + 0.40 * profit_tp3
        )

        risk_reward = (
            expected_profit / worst_case_loss
            if worst_case_loss > 0
            else 0.0
        )

        symbol = str(row.get("symbol", "")).upper()
        name = str(row.get("name", symbol))

        rank = int(
            safe_float(
                row.get("market_cap_rank"),
                i + 1,
            )
        )

        coin = Coin(
            rank=rank,
            coin_id=str(row.get("id", "")),
            symbol=symbol,
            name=name,

            price=price,
            market_cap=market_cap,
            volume_24h=volume,

            change_24h=change_24h,
            change_7d=change_7d,
            change_30d=change_30d,

            momentum=momentum,
            liquidity_score=liquidity,
            affinity=affinity,
            probability=probability,

            buy_limit=buy_limit,
            stop_loss=stop_price,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,

            allocation=allocation,
            quantity=quantity,

            worst_case_loss=worst_case_loss,

            profit_tp1=profit_tp1,
            profit_tp2=profit_tp2,
            profit_tp3=profit_tp3,

            expected_profit=expected_profit,
            risk_reward=risk_reward,
        )

        coins.append(coin)

    return coins


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidate(
    coins: Sequence[Coin],
    capital: float,
    max_positions: int,
    risk_budget: float,
    min_profit: float,
    rng: random.Random,
) -> Optional[Candidate]:
    if not coins:
        return None

    ordering = list(range(len(coins)))

    def random_priority(index: int) -> float:
        p = clamp(
            coins[index].probability,
            1e-9,
            1.0,
        )
        return rng.random() ** (1.0 / p)

    ordering.sort(
        key=random_priority,
        reverse=True,
    )

    selected: list[int] = []
    capital_used = 0.0
    worst_loss = 0.0
    expected_profit = 0.0

    for index in ordering:
        if len(selected) >= max_positions:
            break

        coin = coins[index]
        allocation = coin.allocation

        if capital_used + allocation > capital + 1e-12:
            continue

        if worst_loss + coin.worst_case_loss > risk_budget + 1e-12:
            continue

        p = clamp(coin.probability, 0.0, 1.0)

        if rng.random() > p:
            continue

        selected.append(index)

        capital_used += allocation
        worst_loss += coin.worst_case_loss
        expected_profit += coin.expected_profit

    if not selected:
        return None

    if expected_profit < min_profit:
        return None

    capital_efficiency = (
        expected_profit / capital_used
        if capital_used > 0
        else 0.0
    )

    risk_efficiency = (
        expected_profit / worst_loss
        if worst_loss > 0
        else 0.0
    )

    score = (
        expected_profit
        + 100.0 * capital_efficiency
        + 10.0 * risk_efficiency
    )

    return Candidate(
        indices=selected,
        capital_used=capital_used,
        expected_profit=expected_profit,
        worst_case_loss=worst_loss,
        score=score,
    )


# ============================================================
# LOCAL IMPROVEMENT
# ============================================================

def improve_candidate(
    candidate: Candidate,
    coins: Sequence[Coin],
    capital: float,
    max_positions: int,
    risk_budget: float,
    min_profit: float,
) -> Candidate:
    current = candidate
    changed = True

    while changed:
        changed = False
        selected_set = set(current.indices)

        for index, coin in enumerate(coins):
            if index in selected_set:
                continue

            if len(current.indices) >= max_positions:
                break

            new_indices = current.indices + [index]

            new_capital = sum(
                coins[i].allocation
                for i in new_indices
            )

            new_risk = sum(
                coins[i].worst_case_loss
                for i in new_indices
            )

            new_profit = sum(
                coins[i].expected_profit
                for i in new_indices
            )

            if new_capital > capital + 1e-12:
                continue

            if new_risk > risk_budget + 1e-12:
                continue

            if new_profit < min_profit:
                continue

            capital_efficiency = (
                new_profit / new_capital
                if new_capital > 0
                else 0.0
            )

            risk_efficiency = (
                new_profit / new_risk
                if new_risk > 0
                else 0.0
            )

            new_score = (
                new_profit
                + 100.0 * capital_efficiency
                + 10.0 * risk_efficiency
            )

            if new_score > current.score:
                current = Candidate(
                    indices=new_indices,
                    capital_used=new_capital,
                    expected_profit=new_profit,
                    worst_case_loss=new_risk,
                    score=new_score,
                )

                changed = True
                break

    return current


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize(
    coins: Sequence[Coin],
    capital: float,
    samples: int,
    max_positions: int,
    risk_budget: float,
    min_profit: float,
    seed: int,
) -> Optional[Candidate]:
    rng = random.Random(seed)
    best: Optional[Candidate] = None
    feasible_count = 0
    start = time.time()

    for iteration in range(samples):
        candidate = generate_candidate(
            coins=coins,
            capital=capital,
            max_positions=max_positions,
            risk_budget=risk_budget,
            min_profit=min_profit,
            rng=rng,
        )

        if candidate is None:
            continue

        feasible_count += 1

        candidate = improve_candidate(
            candidate=candidate,
            coins=coins,
            capital=capital,
            max_positions=max_positions,
            risk_budget=risk_budget,
            min_profit=min_profit,
        )

        if best is None or candidate.score > best.score:
            best = candidate

        if samples >= 10:
            step = max(1, samples // 10)

            if iteration > 0 and iteration % step == 0:
                elapsed = time.time() - start
                LOG.info(
                    "  %d/%d | feasible=%d | elapsed=%.1fs",
                    iteration,
                    samples,
                    feasible_count,
                    elapsed,
                )

    return best


# ============================================================
# FALLBACK DETERMINISTIC SEARCH
# ============================================================

def deterministic_fallback(
    coins: Sequence[Coin],
    capital: float,
    max_positions: int,
    risk_budget: float,
    min_profit: float,
) -> Optional[Candidate]:
    ranked = sorted(
        range(len(coins)),
        key=lambda i: (
            coins[i].expected_profit,
            coins[i].risk_reward,
            coins[i].probability,
        ),
        reverse=True,
    )

    selected: list[int] = []
    capital_used = 0.0
    risk_used = 0.0
    profit = 0.0

    for index in ranked:
        coin = coins[index]

        if len(selected) >= max_positions:
            break

        if capital_used + coin.allocation > capital:
            continue

        if risk_used + coin.worst_case_loss > risk_budget:
            continue

        selected.append(index)

        capital_used += coin.allocation
        risk_used += coin.worst_case_loss
        profit += coin.expected_profit

        if profit >= min_profit:
            break

    if not selected:
        return None

    if profit < min_profit:
        return None

    capital_efficiency = (
        profit / capital_used
        if capital_used > 0
        else 0.0
    )

    risk_efficiency = (
        profit / risk_used
        if risk_used > 0
        else 0.0
    )

    score = (
        profit
        + 100.0 * capital_efficiency
        + 10.0 * risk_efficiency
    )

    return Candidate(
        indices=selected,
        capital_used=capital_used,
        expected_profit=profit,
        worst_case_loss=risk_used,
        score=score,
    )


# ============================================================
# OUTPUT HELPERS
# ============================================================

def money(value: float) -> str:
    return f"${value:,.6f}"


def print_portfolio(
    candidate: Candidate,
    coins: Sequence[Coin],
) -> None:
    print()
    print("=" * 118)
    print("SELECTED PORTFOLIO")
    print("=" * 118)

    print(f"Positions       : {len(candidate.indices)}")
    print(f"Capital used    : {money(candidate.capital_used)}")
    print(f"Modeled profit  : {money(candidate.expected_profit)}")
    print(f"Worst-case risk : {money(candidate.worst_case_loss)}")
    print(f"Score           : {candidate.score:.8f}")

    print()
    print(
        f"{'Symbol':<10} "
        f"{'Qty':>14} "
        f"{'Entry':>14} "
        f"{'Stop':>14} "
        f"{'TP1':>14} "
        f"{'TP2':>14} "
        f"{'TP3':>14} "
        f"{'Risk':>14} "
        f"{'Profit':>14} "
        f"{'P':>8}"
    )

    print("-" * 145)

    for index in candidate.indices:
        coin = coins[index]

        print(
            f"{coin.symbol:<10} "
            f"{coin.quantity:>14.8f} "
            f"{money(coin.buy_limit):>14} "
            f"{money(coin.stop_loss):>14} "
            f"{money(coin.tp1):>14} "
            f"{money(coin.tp2):>14} "
            f"{money(coin.tp3):>14} "
            f"{money(coin.worst_case_loss):>14} "
            f"{money(coin.expected_profit):>14} "
            f"{coin.probability:>8.4f}"
        )


def save_csv(
    filename: str,
    coins: Sequence[Coin],
    selected_indices: Sequence[int],
) -> None:
    selected = set(selected_indices)

    fields = [
        "rank",
        "coin_id",
        "symbol",
        "name",
        "price",
        "market_cap",
        "volume_24h",
        "change_24h",
        "change_7d",
        "change_30d",
        "momentum",
        "liquidity_score",
        "affinity",
        "probability",
        "buy_limit",
        "stop_loss",
        "tp1",
        "tp2",
        "tp3",
        "allocation",
        "quantity",
        "worst_case_loss",
        "profit_tp1",
        "profit_tp2",
        "profit_tp3",
        "expected_profit",
        "risk_reward",
        "selected",
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for i, coin in enumerate(coins):
            writer.writerow(
                {
                    "rank": coin.rank,
                    "coin_id": coin.coin_id,
                    "symbol": coin.symbol,
                    "name": coin.name,
                    "price": coin.price,
                    "market_cap": coin.market_cap,
                    "volume_24h": coin.volume_24h,
                    "change_24h": coin.change_24h,
                    "change_7d": coin.change_7d,
                    "change_30d": coin.change_30d,
                    "momentum": coin.momentum,
                    "liquidity_score": coin.liquidity_score,
                    "affinity": coin.affinity,
                    "probability": coin.probability,
                    "buy_limit": coin.buy_limit,
                    "stop_loss": coin.stop_loss,
                    "tp1": coin.tp1,
                    "tp2": coin.tp2,
                    "tp3": coin.tp3,
                    "allocation": coin.allocation,
                    "quantity": coin.quantity,
                    "worst_case_loss": coin.worst_case_loss,
                    "profit_tp1": coin.profit_tp1,
                    "profit_tp2": coin.profit_tp2,
                    "profit_tp3": coin.profit_tp3,
                    "expected_profit": coin.expected_profit,
                    "risk_reward": coin.risk_reward,
                    "selected": i in selected,
                }
            )


def print_statistics(coins: Sequence[Coin]) -> None:
    if not coins:
        return

    probabilities = [c.probability for c in coins]
    expected_profits = [c.expected_profit for c in coins]
    risks = [c.worst_case_loss for c in coins]

    print()
    print("=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)

    print(f"Coins loaded          : {len(coins)}")
    print(f"Mean probability      : {statistics.mean(probabilities):.6f}")
    print(f"Median probability    : {statistics.median(probabilities):.6f}")
    print(f"Mean modeled profit   : {money(statistics.mean(expected_profits))}")
    print(f"Mean modeled risk     : {money(statistics.mean(risks))}")

    positive = sum(p > 0 for p in expected_profits)
    print(f"Positive-profit coins : {positive}/{len(coins)}")
def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    x = clamp(x, -60.0, 60.0)

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def normalize_array(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)

    if len(x) == 0:
        return x

    finite = np.isfinite(x)

    if not finite.any():
        return np.full(len(x), 0.5, dtype=np.float64)

    valid = x[finite]

    lo = float(np.min(valid))
    hi = float(np.max(valid))

    if abs(hi - lo) < 1e-12:
        return np.full(len(x), 0.5, dtype=np.float64)

    out = (x - lo) / (hi - lo)

    out[~np.isfinite(out)] = 0.5

    return np.clip(out, 0.0, 1.0)


def decimal_places(value: Any, default: int = 8) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def round_down_decimal(
    value: float,
    decimals: int,
) -> str:
    if value <= 0:
        return "0"

    decimals = max(0, int(decimals))
    quantum = Decimal("1").scaleb(-decimals)

    rounded = Decimal(str(value)).quantize(
        quantum,
        rounding=ROUND_DOWN,
    )

    return format(rounded, "f")


def normalise_pair_name(value: str) -> str:
    value = str(value).upper().strip()

    replacements = {
        "XBT": "BTC",
        "XDG": "DOGE",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return "".join(
        character
        for character in value
        if character.isalnum()
    )

def load_kraken_pair_metadata(
    client: KrakenSpotClient,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve Kraken's current pair metadata.

    Returns a lookup indexed by normalised base symbol, for example:

        BTC -> {
            "pair": "XBTUSD",
            "base": "XXBT",
            "quote": "ZUSD",
            "ordermin": 0.0001,
            "pair_decimals": 1,
            "lot_decimals": 8,
        }
    """

    raw_pairs = client.asset_pairs()
    metadata: dict[str, dict[str, Any]] = {}

    for api_name, info in raw_pairs.items():
        if not isinstance(info, dict):
            continue

        status = str(info.get("status", "online")).lower()

        if status not in {"online", "post_only"}:
            continue

        base = str(
            info.get("base", "")
        ).upper()

        quote = str(
            info.get("quote", "")
        ).upper()

        wsname = str(
            info.get("wsname", "")
        ).upper()

        altname = str(
            info.get("altname", api_name)
        ).upper()

        # Prefer USD pairs because the optimizer uses USD prices.
        pair_candidates = [
            wsname,
            altname,
            api_name.upper(),
        ]

        pair_name = next(
            (
                candidate
                for candidate in pair_candidates
                if candidate.endswith("USD")
                or candidate.endswith("/USD")
            ),
            "",
        )

        if not pair_name:
            continue

        if "/" in pair_name:
            base_symbol = pair_name.split("/", 1)[0]
        else:
            base_symbol = pair_name

            for suffix in ("USD", "ZUSD"):
                if base_symbol.endswith(suffix):
                    base_symbol = base_symbol[:-len(suffix)]
                    break

        base_symbol = normalise_pair_name(base_symbol)

        if not base_symbol:
            continue

        metadata[base_symbol] = {
            "api_name": api_name,
            "pair": altname or api_name,
            "wsname": wsname,
            "base": base,
            "quote": quote,
            "ordermin": safe_float(
                info.get("ordermin"),
                0.0,
            ),
            "pair_decimals": decimal_places(
                info.get("pair_decimals"),
                8,
            ),
            "lot_decimals": decimal_places(
                info.get("lot_decimals"),
                8,
            ),
            "status": status,
        }

    return metadata


def make_planned_order(
    coin: Coin,
    pair_metadata: dict[str, Any],
) -> PlannedOrder | None:
    """
    Build an order that satisfies Kraken's volume minimum and
    volume precision.

    Orders are skipped instead of being submitted incorrectly.
    """

    pair = str(
        pair_metadata.get("pair")
        or pair_metadata.get("api_name")
        or ""
    ).upper()

    if not pair:
        LOG.warning(
            "Skipping %s: Kraken pair name is unavailable.",
            coin.symbol,
        )
        return None

    minimum_volume = safe_float(
        pair_metadata.get("ordermin"),
        0.0,
    )

    volume_decimals = decimal_places(
        pair_metadata.get("lot_decimals"),
        8,
    )

    price_decimals = decimal_places(
        pair_metadata.get("pair_decimals"),
        8,
    )

    raw_volume = safe_float(
        coin.quantity,
        0.0,
    )

    volume = round_down_decimal(
        raw_volume,
        volume_decimals,
    )

    rounded_volume = safe_float(volume, 0.0)

    if rounded_volume <= 0:
        LOG.warning(
            "Skipping %s: rounded volume is zero. "
            "raw=%.12f precision=%d",
            coin.symbol,
            raw_volume,
            volume_decimals,
        )
        return None

    if minimum_volume > 0 and rounded_volume < minimum_volume:
        LOG.warning(
            "Skipping %s/%s: volume %.12f is below "
            "Kraken minimum %.12f.",
            coin.symbol,
            pair,
            rounded_volume,
            minimum_volume,
        )
        return None

    price = round_down_decimal(
        coin.buy_limit,
        price_decimals,
    )

    stop_price = round_down_decimal(
        coin.stop_loss,
        price_decimals,
    )

    tp_price = round_down_decimal(
        coin.tp1,
        price_decimals,
    )

    LOG.info(
        "Prepared %s %s volume=%s price=%s min_volume=%s",
        pair,
        "BUY",
        volume,
        price,
        minimum_volume,
    )

    return PlannedOrder(
        pair=pair,
        side="buy",
        ordertype="limit",
        volume=volume,
        price=price,
        stop_price=stop_price,
        tp_price=tp_price,
    )

# ============================================================
# CYCLE CONTROLLER
# ============================================================
def run_optimizer_cycle(
    args: argparse.Namespace,
    client: KrakenSpotClient,
    trader: Trader,
) -> None:
    LOG.info("Starting optimization cycle")

    # --------------------------------------------------------
    # Reconcile account state
    # --------------------------------------------------------

    balances: dict[str, Any] = {}
    open_orders: dict[str, Any] = {}

    if trader.live:
        balances = client.balance()
        open_orders = client.open_orders()

        LOG.info(
            "Account state: %d balances, %d open orders",
            len(balances),
            len(open_orders.get("open", {})),
        )
    else:
        LOG.info(
            "Dry-run mode: skipping private account reconciliation."
        )

    # --------------------------------------------------------
    # Load Kraken pair metadata
    # --------------------------------------------------------

    pair_metadata: dict[str, dict[str, Any]] = {}

    try:
        pair_metadata = load_kraken_pair_metadata(client)
        LOG.info(
            "Loaded %d eligible Kraken USD pairs.",
            len(pair_metadata),
        )
    except KrakenError:
        LOG.exception(
            "Unable to load Kraken pair metadata."
        )

    # --------------------------------------------------------
    # Fetch market data
    # --------------------------------------------------------

    LOG.info("Fetching market data...")

    market_data = fetch_market_data(
        number_of_coins=args.coins,
        timeout=args.timeout,
    )

    LOG.info(
        "Received %d market records.",
        len(market_data),
    )

    coins = build_coins(
        market_data=market_data,
        capital=args.capital,
        buy_pullback=args.buy_pullback,
        stop_loss_pct=args.stop_loss,
        tp1_pct=args.tp1,
        tp2_pct=args.tp2,
        tp3_pct=args.tp3,
        max_positions=args.max_positions,
        temperature=args.temperature,
    )

    if not coins:
        raise RuntimeError(
            "No usable coins were returned."
        )

    LOG.info(
        "Using %d valid coins.",
        len(coins),
    )

    # --------------------------------------------------------
    # Apply Kraken pair metadata before optimization
    # --------------------------------------------------------

    usable_coins: list[Coin] = []

    for coin in coins:
        symbol = normalise_pair_name(coin.symbol)
        metadata = pair_metadata.get(symbol)

        if metadata is None:
            LOG.info(
                "Skipping %s: no eligible Kraken USD pair.",
                coin.symbol,
            )
            continue

        coin.kraken_pair = str(
            metadata.get("pair")
            or metadata.get("api_name")
            or ""
        )

        coin.base_asset = str(
            metadata.get("base", "")
        )

        coin.quote_asset = str(
            metadata.get("quote", "")
        )

        coin.order_minimum = safe_float(
            metadata.get("ordermin"),
            0.0,
        )

        coin.price_decimals = decimal_places(
            metadata.get("pair_decimals"),
            8,
        )

        coin.volume_decimals = decimal_places(
            metadata.get("lot_decimals"),
            8,
        )

        # Reject positions which cannot meet the exchange minimum
        # after rounding down.
        rounded_quantity = safe_float(
            round_down_decimal(
                coin.quantity,
                coin.volume_decimals,
            ),
            0.0,
        )

        if (
            coin.order_minimum > 0
            and rounded_quantity < coin.order_minimum
        ):
            LOG.info(
                "Skipping %s: quantity %.12f is below "
                "minimum %.12f.",
                coin.symbol,
                rounded_quantity,
                coin.order_minimum,
            )
            continue

        usable_coins.append(coin)

    coins = usable_coins

    if not coins:
        LOG.warning(
            "No coins meet Kraken pair and volume requirements."
        )
        return

    LOG.info(
        "%d coins remain after Kraken validation.",
        len(coins),
    )

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    LOG.info(
        "Running probabilistic knapsack search..."
    )

    start = time.time()

    candidate = optimize(
        coins=coins,
        capital=args.capital,
        samples=args.samples,
        max_positions=args.max_positions,
        risk_budget=args.risk_budget,
        min_profit=args.min_profit,
        seed=args.seed,
    )

    elapsed = time.time() - start

    if candidate is None:
        LOG.info(
            "Probabilistic sampling did not find "
            "a feasible portfolio."
        )

        LOG.info(
            "Running deterministic fallback..."
        )

        candidate = deterministic_fallback(
            coins=coins,
            capital=args.capital,
            max_positions=args.max_positions,
            risk_budget=args.risk_budget,
            min_profit=args.min_profit,
        )

    LOG.info(
        "Optimization time: %.3f seconds",
        elapsed,
    )

    if candidate is None:
        LOG.warning(
            "No feasible portfolio found this cycle."
        )
        return

    print_portfolio(candidate, coins)

    # --------------------------------------------------------
    # Create validated orders
    # --------------------------------------------------------

    orders: list[PlannedOrder] = []

    for index in candidate.indices:
        coin = coins[index]

        symbol = normalise_pair_name(coin.symbol)
        metadata = pair_metadata.get(symbol)

        if metadata is None:
            LOG.warning(
                "Skipping %s: pair metadata disappeared.",
                coin.symbol,
            )
            continue

        order = make_planned_order(
            coin=coin,
            pair_metadata=metadata,
        )

        if order is None:
            continue

        orders.append(order)

    LOG.info(
        "Prepared %d valid orders from %d selected positions.",
        len(orders),
        len(candidate.indices),
    )

    # --------------------------------------------------------
    # Submit orders
    # --------------------------------------------------------

    for order in orders:
        try:
            result = trader.submit(order)
            LOG.info(
                "Order result for %s: %s",
                order.pair,
                result,
            )

        except KrakenError as exc:
            LOG.error(
                "Order rejected for %s: %s",
                order.pair,
                exc,
            )

    # --------------------------------------------------------
    # Renew dead-man's switch
    # --------------------------------------------------------

    if trader.live:
        try:
            client.cancel_all_after(60)
            LOG.info(
                "Dead-man's switch renewed for 60 seconds."
            )
        except KrakenError:
            LOG.exception(
                "Failed to renew dead-man's switch."
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_csv(
        filename=args.csv,
        coins=coins,
        selected_indices=candidate.indices,
    )

    LOG.info(
        "CSV saved to %s",
        args.csv,
    )

    LOG.info(
        "Cycle complete."
    )


def main() -> None:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        LOG.error("Configuration error: %s", exc)
        sys.exit(1)

    # --------------------------------------------------------
    # Logging setup
    # --------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # --------------------------------------------------------
    # Live trading guard
    # --------------------------------------------------------

    live = live_enabled(args)

    if args.live and not live:
        raise SystemExit(
            "Live trading requires --live, --confirm-live, "
        )

    LOG.warning("Trading mode: %s", "LIVE" if live else "DRY RUN")

    # --------------------------------------------------------
    # Kraken client and trader
    # --------------------------------------------------------

    client = KrakenSpotClient()
    trader = Trader(client, live=live)

    # --------------------------------------------------------
    # Cycle loop
    # --------------------------------------------------------

    cycle = 0

    while True:
        cycle += 1

        LOG.info("=== CYCLE %d START ===", cycle)

        try:
            run_optimizer_cycle(args, client, trader)
        except KrakenError:
            LOG.exception("Kraken API error in cycle %d", cycle)
        except Exception:
            LOG.exception("Unexpected cycle failure in cycle %d", cycle)

        LOG.info("=== CYCLE %d END ===", cycle)

        if args.max_cycles and cycle >= args.max_cycles:
            LOG.info("Maximum cycles reached (%d).", args.max_cycles)
            break

        LOG.info("Sleeping for %d seconds.", args.interval)
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
