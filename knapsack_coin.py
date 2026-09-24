
"""
Probabilistic Crypto Knapsack Optimizer
=======================================

NO OPTICAL VALVES
NO BIT MASKS
NO 64-BIT STATE REPRESENTATION

Uses:
    Real crypto market data
        ↓
    Momentum / liquidity scoring
        ↓
    Probabilistic candidate generation
        ↓
    Buy-limit calculation
        ↓
    Stop-loss calculation
        ↓
    Multiple take-profit levels
        ↓
    Capacity-constrained knapsack
        ↓
    Hard risk constraint
        ↓
    Positive modeled-profit constraint
        ↓
    Best portfolio

IMPORTANT
---------
This is a simulation/optimization program.

It does NOT:
    - place orders
    - connect to a trading account
    - guarantee profit
    - guarantee execution
    - predict future prices

"Positive profit" in this program means the modeled TP-weighted
payoff is positive under the assumptions supplied by the user.

A real market can move against every modeled scenario.

Install:
    pip install requests numpy

Run:
    python knapsack_coin.py

Examples:
    python knapsack_coin.py --coins 100 --capital 1000

    python knapsack_coin.py \
        --coins 100 \
        --capital 5000 \
        --samples 100000 \
        --max-positions 10 \
        --risk-budget 50

    python knapsack_coin.py \
        --coins 50 \
        --capital 2500 \
        --buy-pullback 0.02 \
        --stop-loss 0.05 \
        --tp1 0.05 \
        --tp2 0.10 \
        --tp3 0.20
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests


# ============================================================
# CONFIGURATION
# ============================================================

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

DEFAULT_TIMEOUT = 20

DEFAULT_COINS = 100
DEFAULT_CAPITAL = 1000.0
DEFAULT_SAMPLES = 100_000

DEFAULT_TEMPERATURE = 1.0

DEFAULT_BUY_PULLBACK = 0.02
DEFAULT_STOP_LOSS = 0.05

DEFAULT_TP1 = 0.05
DEFAULT_TP2 = 0.10
DEFAULT_TP3 = 0.20

DEFAULT_MAX_POSITIONS = 25

DEFAULT_RISK_BUDGET = 50.0

DEFAULT_MIN_PROFIT = 1.0

DEFAULT_SEED = 42

DEFAULT_CSV = "crypto_knapsack_results.csv"


# ============================================================
# DATA STRUCTURES
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
    """
    Numerically stable sigmoid.
    """

    x = clamp(x, -60.0, 60.0)

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def normalize_array(values: Sequence[float]) -> np.ndarray:
    """
    Min-max normalization.

    Constant arrays become 0.5 everywhere.
    """

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
        description="Probabilistic crypto knapsack optimizer without valves."
    )

    parser.add_argument(
        "--coins",
        type=int,
        default=DEFAULT_COINS,
        help="Number of coins to retrieve.",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Total capital available.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of probabilistic candidate portfolios.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Probability temperature.",
    )

    parser.add_argument(
        "--buy-pullback",
        type=float,
        default=DEFAULT_BUY_PULLBACK,
        help="Entry pullback from current price.",
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_STOP_LOSS,
        help="Stop distance below entry.",
    )

    parser.add_argument(
        "--tp1",
        type=float,
        default=DEFAULT_TP1,
        help="Take-profit 1 distance above entry.",
    )

    parser.add_argument(
        "--tp2",
        type=float,
        default=DEFAULT_TP2,
        help="Take-profit 2 distance above entry.",
    )

    parser.add_argument(
        "--tp3",
        type=float,
        default=DEFAULT_TP3,
        help="Take-profit 3 distance above entry.",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAX_POSITIONS,
        help="Maximum number of simultaneous positions.",
    )

    parser.add_argument(
        "--risk-budget",
        type=float,
        default=DEFAULT_RISK_BUDGET,
        help="Maximum total modeled stop-loss risk.",
    )

    parser.add_argument(
        "--min-profit",
        type=float,
        default=DEFAULT_MIN_PROFIT,
        help="Minimum modeled portfolio profit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="CSV output filename.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

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

    if not (
        0 < args.tp1 < args.tp2 < args.tp3
    ):
        raise ValueError(
            "Take-profit levels must satisfy 0 < TP1 < TP2 < TP3."
        )

    if args.max_positions < 1:
        raise ValueError("--max-positions must be >= 1")

    if args.risk_budget < 0:
        raise ValueError("--risk-budget must be >= 0")

    if args.min_profit < 0:
        raise ValueError("--min-profit must be >= 0")


# ============================================================
# MARKET DATA
# ============================================================

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
        "User-Agent": "CryptoKnapsackSimulator/1.0",
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
    """
    Produces a continuous selection affinity.

    Positive values increase selection probability.
    Negative values decrease it.
    """

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

        volume = safe_float(
            row.get("total_volume")
        )

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

    # Momentum can be negative.
    # Convert it to a bounded [0,1] score.
    momentum_score = normalize_array(momentum_raw)

    # Each position receives an equal maximum allocation.
    #
    # This prevents one cheap coin from receiving an enormous
    # notional position merely because its unit price is small.
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

        # ----------------------------------------------------
        # PRICE LEVELS
        # ----------------------------------------------------

        buy_limit = price * (1.0 - buy_pullback)

        stop_price = buy_limit * (1.0 - stop_loss_pct)

        tp1 = buy_limit * (1.0 + tp1_pct)
        tp2 = buy_limit * (1.0 + tp2_pct)
        tp3 = buy_limit * (1.0 + tp3_pct)

        # ----------------------------------------------------
        # POSITION SIZE
        # ----------------------------------------------------

        allocation = min(
            position_allocation,
            capital,
        )

        quantity = allocation / buy_limit

        # ----------------------------------------------------
        # PROFIT SCENARIOS
        # ----------------------------------------------------

        profit_tp1 = (
            quantity * (tp1 - buy_limit)
        )

        profit_tp2 = (
            quantity * (tp2 - buy_limit)
        )

        profit_tp3 = (
            quantity * (tp3 - buy_limit)
        )

        # ----------------------------------------------------
        # WORST-CASE LOSS
        # ----------------------------------------------------

        worst_case_loss = (
            quantity * (buy_limit - stop_price)
        )

        # ----------------------------------------------------
        # EXPECTED PROFIT
        #
        # Scenario weights:
        #
        # TP1 = 25%
        # TP2 = 35%
        # TP3 = 40%
        #
        # These are modeling assumptions, not probabilities
        # derived from the market.
        # ----------------------------------------------------

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

        symbol = str(
            row.get("symbol", "")
        ).upper()

        name = str(
            row.get("name", symbol)
        )

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
    """
    Generate one feasible portfolio.

    There is deliberately no bit-mask representation.

    A portfolio is simply:

        [3, 7, 12, 19, ...]

    containing the indices of selected coins.
    """

    if not coins:
        return None

    # --------------------------------------------------------
    # Weighted random ordering.
    #
    # Higher probability means earlier consideration.
    # --------------------------------------------------------

    ordering = list(range(len(coins)))

    def random_priority(index: int) -> float:

        p = clamp(
            coins[index].probability,
            1e-9,
            1.0,
        )

        # Larger values are more likely to be selected early.
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

        # ----------------------------------------------------
        # Capital constraint
        # ----------------------------------------------------

        if capital_used + allocation > capital + 1e-12:
            continue

        # ----------------------------------------------------
        # Risk constraint
        # ----------------------------------------------------

        if worst_loss + coin.worst_case_loss > risk_budget + 1e-12:
            continue

        # ----------------------------------------------------
        # Probabilistic acceptance
        # ----------------------------------------------------

        p = clamp(
            coin.probability,
            0.0,
            1.0,
        )

        if rng.random() > p:
            continue

        selected.append(index)

        capital_used += allocation
        worst_loss += coin.worst_case_loss
        expected_profit += coin.expected_profit

    if not selected:
        return None

    # --------------------------------------------------------
    # HARD PROFIT REQUIREMENT
    # --------------------------------------------------------

    if expected_profit < min_profit:
        return None

    # --------------------------------------------------------
    # Portfolio score
    #
    # Profit per unit of capital, adjusted for risk.
    # --------------------------------------------------------

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

        if (
            best is None
            or candidate.score > best.score
        ):
            best = candidate

        # Progress every 10%.
        if samples >= 10:

            step = max(1, samples // 10)

            if (
                iteration > 0
                and iteration % step == 0
            ):

                elapsed = time.time() - start

                print(
                    f"  {iteration:>10,}/{samples:,}"
                    f" | feasible={feasible_count:,}"
                    f" | elapsed={elapsed:.1f}s"
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
    """
    If probabilistic sampling fails, construct a portfolio
    deterministically from the highest-quality candidates.

    This avoids the previous:
        "No feasible candidate was sampled."
    problem.
    """

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
# OUTPUT
# ============================================================

def money(value: float) -> str:
    return f"${value:,.6f}"


def print_market_summary(coins: Sequence[Coin]) -> None:

    print()
    print("=" * 118)
    print("MARKET DATA")
    print("=" * 118)

    print(
        f"{'Rank':>5} "
        f"{'Symbol':<10} "
        f"{'Price':>14} "
        f"{'24h':>9} "
        f"{'7d':>9} "
        f"{'30d':>9} "
        f"{'P':>8} "
        f"{'Buy':>14} "
        f"{'Stop':>14} "
        f"{'TP1':>14} "
        f"{'TP2':>14} "
        f"{'TP3':>14}"
    )

    print("-" * 118)

    for coin in coins:

        print(
            f"{coin.rank:5d} "
            f"{coin.symbol:<10} "
            f"{money(coin.price):>14} "
            f"{coin.change_24h:>8.2f}% "
            f"{coin.change_7d:>8.2f}% "
            f"{coin.change_30d:>8.2f}% "
            f"{coin.probability:>8.4f} "
            f"{money(coin.buy_limit):>14} "
            f"{money(coin.stop_loss):>14} "
            f"{money(coin.tp1):>14} "
            f"{money(coin.tp2):>14} "
            f"{money(coin.tp3):>14}"
        )


def print_portfolio(
    candidate: Candidate,
    coins: Sequence[Coin],
) -> None:

    print()
    print("=" * 118)
    print("SELECTED PORTFOLIO")
    print("=" * 118)

    print(
        f"Positions       : {len(candidate.indices)}"
    )

    print(
        f"Capital used    : {money(candidate.capital_used)}"
    )

    print(
        f"Modeled profit  : {money(candidate.expected_profit)}"
    )

    print(
        f"Worst-case risk : {money(candidate.worst_case_loss)}"
    )

    print(
        f"Score           : {candidate.score:.8f}"
    )

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


def print_configuration(args: argparse.Namespace) -> None:

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(f"Coins             : {args.coins}")
    print(f"Capital           : {money(args.capital)}")
    print(f"Samples           : {args.samples:,}")
    print(f"Temperature       : {args.temperature}")
    print(f"Buy pullback      : {args.buy_pullback:.2%}")
    print(f"Stop loss         : {args.stop_loss:.2%}")
    print(f"TP1               : {args.tp1:.2%}")
    print(f"TP2               : {args.tp2:.2%}")
    print(f"TP3               : {args.tp3:.2%}")
    print(f"Max positions     : {args.max_positions}")
    print(f"Risk budget       : {money(args.risk_budget)}")
    print(f"Minimum profit    : {money(args.min_profit)}")
    print(f"Random seed       : {args.seed}")


# ============================================================
# CSV
# ============================================================

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


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(coins: Sequence[Coin]) -> None:

    if not coins:
        return

    probabilities = [
        c.probability
        for c in coins
    ]

    expected_profits = [
        c.expected_profit
        for c in coins
    ]

    risks = [
        c.worst_case_loss
        for c in coins
    ]

    print()
    print("=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)

    print(
        f"Coins loaded          : {len(coins)}"
    )

    print(
        f"Mean probability      : "
        f"{statistics.mean(probabilities):.6f}"
    )

    print(
        f"Median probability    : "
        f"{statistics.median(probabilities):.6f}"
    )

    print(
        f"Mean modeled profit   : "
        f"{money(statistics.mean(expected_profits))}"
    )

    print(
        f"Mean modeled risk     : "
        f"{money(statistics.mean(risks))}"
    )

    positive = sum(
        p > 0
        for p in expected_profits
    )

    print(
        f"Positive-profit coins : "
        f"{positive}/{len(coins)}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    try:
        validate_args(args)

        random.seed(args.seed)
        np.random.seed(args.seed)

        print_configuration(args)

        # ----------------------------------------------------
        # FETCH DATA
        # ----------------------------------------------------

        print()
        print("Fetching current market data...")

        market_data = fetch_market_data(
            number_of_coins=args.coins,
            timeout=args.timeout,
        )

        print(
            f"Received {len(market_data)} market records."
        )

        # ----------------------------------------------------
        # BUILD MODEL
        # ----------------------------------------------------

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

        print(
            f"Using {len(coins)} valid coins."
        )

        print_statistics(coins)

        # ----------------------------------------------------
        # MARKET TABLE
        # ----------------------------------------------------

        print_market_summary(coins)

        # ----------------------------------------------------
        # OPTIMIZATION
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("PROBABILISTIC KNAPSACK SEARCH")
        print("=" * 70)

        print(
            "Searching for a portfolio satisfying:"
        )

        print(
            f"  capital <= {money(args.capital)}"
        )

        print(
            f"  positions <= {args.max_positions}"
        )

        print(
            f"  worst-case modeled loss <= "
            f"{money(args.risk_budget)}"
        )

        print(
            f"  modeled profit >= "
            f"{money(args.min_profit)}"
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

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if candidate is None:

            print()
            print(
                "Probabilistic sampling did not find a "
                "portfolio satisfying all constraints."
            )

            print(
                "Running deterministic fallback..."
            )

            candidate = deterministic_fallback(
                coins=coins,
                capital=args.capital,
                max_positions=args.max_positions,
                risk_budget=args.risk_budget,
                min_profit=args.min_profit,
            )

        print()
        print(
            f"Optimization time: {elapsed:.3f} seconds"
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        if candidate is None:

            print()
            print("=" * 70)
            print("NO FEASIBLE PORTFOLIO")
            print("=" * 70)

            print()
            print(
                "No portfolio satisfies all constraints."
            )

            print(
                "Try increasing --risk-budget or "
                "lowering --min-profit."
            )

            print()
            print(
                "The program intentionally refuses to "
                "manufacture a profitable result."
            )

            save_csv(
                filename=args.csv,
                coins=coins,
                selected_indices=[],
            )

            print(
                f"Market data saved to: {args.csv}"
            )

            return

        print_portfolio(
            candidate=candidate,
            coins=coins,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("CONSTRAINT VALIDATION")
        print("=" * 70)

        capital_ok = (
            candidate.capital_used
            <= args.capital + 1e-9
        )

        position_ok = (
            len(candidate.indices)
            <= args.max_positions
        )

        risk_ok = (
            candidate.worst_case_loss
            <= args.risk_budget + 1e-9
        )

        profit_ok = (
            candidate.expected_profit
            >= args.min_profit - 1e-9
        )

        print(
            f"Capital constraint : "
            f"{'PASS' if capital_ok else 'FAIL'}"
        )

        print(
            f"Position constraint: "
            f"{'PASS' if position_ok else 'FAIL'}"
        )

        print(
            f"Risk constraint    : "
            f"{'PASS' if risk_ok else 'FAIL'}"
        )

        print(
            f"Profit constraint  : "
            f"{'PASS' if profit_ok else 'FAIL'}"
        )

        all_ok = (
            capital_ok
            and position_ok
            and risk_ok
            and profit_ok
        )

        print()

        if all_ok:
            print(
                "RESULT: FEASIBLE MODELED PORTFOLIO"
            )
        else:
            print(
                "RESULT: CONSTRAINT FAILURE"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_csv(
            filename=args.csv,
            coins=coins,
            selected_indices=candidate.indices,
        )

        print()
        print(
            f"CSV saved to: {args.csv}"
        )

        print()
        print("=" * 70)
        print("IMPORTANT")
        print("=" * 70)

        print(
            "The positive-profit condition is a mathematical "
            "constraint on the simulator's TP assumptions."
        )

        print(
            "It does not make real-world crypto profit "
            "inevitable or guaranteed."
        )

        print(
            "The calculated levels are model outputs, not "
            "guaranteed execution prices."
        )

    except requests.HTTPError as exc:

        print()
        print(
            "CoinGecko HTTP error:"
        )
        print(exc)

        if getattr(exc, "response", None) is not None:
            print(
                f"HTTP status: {exc.response.status_code}"
            )

        sys.exit(1)

    except requests.RequestException as exc:

        print()
        print(
            "Network error:"
        )
        print(exc)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print("Interrupted.")

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            f"ERROR: {type(exc).__name__}: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
