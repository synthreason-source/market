"""
Crypto Impulse Buy Analyzer
============================
Formula:  true_share_value = market_cap / total_supply
Reachability condition: daily_volume / market_cap flux must stay in 1–10% band
Impulse buys are detected via curvature spikes in the volume/mcap flux curve.
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
COIN_ID   = "bitcoin"      # CoinGecko coin id  (bitcoin / ethereum / solana …)
CURRENCY  = "usd"
DAYS      = 90             # lookback window
FLUX_LOW  = 0.01           # 1%  – lower healthy-flux threshold
FLUX_HIGH = 0.10           # 10% – upper healthy-flux threshold
# ───────────────────────────────────────────────────────────────────────────────


def fetch_coin_data(coin_id: str, currency: str, days: int) -> dict:
    base = "https://api.coingecko.com/api/v3"
    print(f"  Fetching market chart for {coin_id} …")
    chart = requests.get(
        f"{base}/coins/{coin_id}/market_chart",
        params={"vs_currency": currency, "days": days, "interval": "daily"},
        timeout=15,
    ).json()

    print(f"  Fetching coin details …")
    detail = requests.get(f"{base}/coins/{coin_id}", timeout=15).json()

    return chart, detail


def curvature(y: np.ndarray) -> np.ndarray:
    """κ = y'' / (1 + y'²)^(3/2)  — signed curvature of the flux curve."""
    dy  = np.gradient(y)
    d2y = np.gradient(dy)
    return d2y / (1 + dy**2) ** 1.5


def detect_impulse_buys(flux: np.ndarray, curv: np.ndarray,
                        flux_lo=FLUX_LOW, flux_hi=FLUX_HIGH,
                        curv_threshold_pct=75) -> np.ndarray:
    """
    An impulse buy is flagged when:
      1. flux is IN the healthy 1-10% band  (volume is proportional & active)
      2. curvature is a LOCAL MAXIMUM above the Nth-percentile  (sharp upward bend)
    """
    in_band   = (flux >= flux_lo) & (flux <= flux_hi)
    curv_thresh = np.percentile(np.abs(curv), curv_threshold_pct)
    local_max_idx = argrelextrema(curv, np.greater, order=2)[0]
    impulse = np.zeros(len(flux), dtype=bool)
    for i in local_max_idx:
        if in_band[i] and curv[i] > curv_thresh:
            impulse[i] = True
    return impulse


def build_prediction(flux: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple rolling-mean prediction extrapolated 14 days forward."""
    pad    = np.convolve(flux, np.ones(window)/window, mode='valid')
    trend  = np.polyfit(np.arange(len(pad)), pad, 1)
    future = np.polyval(trend, np.arange(len(pad), len(pad) + 14))
    future = np.clip(future, 0, None)
    return future


def main():
    print("\n═══ Crypto Impulse Buy Analyzer ═══\n")

    # 1. Fetch ----------------------------------------------------------------
    chart, detail = fetch_coin_data(COIN_ID, CURRENCY, DAYS)

    prices      = np.array([p[1] for p in chart["prices"]])
    volumes     = np.array([v[1] for v in chart["total_volumes"]])
    mcaps       = np.array([m[1] for m in chart["market_caps"]])
    timestamps  = [datetime.fromtimestamp(p[0]/1000) for p in chart["prices"]]

    # 2. Core formula ---------------------------------------------------------
    total_supply     = detail["market_data"]["total_supply"] or \
                       detail["market_data"]["circulating_supply"]
    current_mcap     = mcaps[-1]
    true_share_value = current_mcap / total_supply
    current_price    = prices[-1]
    overvaluation    = (current_price / true_share_value - 1) * 100

    # 3. Flux  (daily volume / market cap) ------------------------------------
    raw_flux  = volumes / mcaps                        # ratio
    flux      = gaussian_filter1d(raw_flux, sigma=1.5) # smoothed

    # 4. Curvature + impulse detection ----------------------------------------
    curv    = curvature(flux)
    impulse = detect_impulse_buys(flux, curv)

    # 5. Prediction -----------------------------------------------------------
    future_flux = build_prediction(flux)
    future_dates = pd.date_range(timestamps[-1], periods=15, freq='D')[1:]

    # ── PRINT SUMMARY ────────────────────────────────────────────────────────
    name = detail["name"]
    sym  = detail["symbol"].upper()
    print(f"\n  Coin            : {name} ({sym})")
    print(f"  Current Price   : ${current_price:>14,.2f}")
    print(f"  Market Cap      : ${current_mcap:>14,.0f}")
    print(f"  Total Supply    : {total_supply:>14,.0f}")
    print(f"  ─────────────────────────────────")
    print(f"  True Share Value: ${true_share_value:>14,.6f}  (mcap / supply)")
    print(f"  Price Premium   : {overvaluation:>+.1f}%  vs. true share value")
    print(f"  Avg Daily Flux  : {flux.mean()*100:.2f}%  (target 1–10%)")
    print(f"  Impulse Events  : {impulse.sum()} detected over {DAYS} days\n")

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="#0d0f14")
    fig.suptitle(
        f"{name} ({sym})  ·  Impulse Buy Detector  ·  {DAYS}-Day Window",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    ax_price  = fig.add_subplot(gs[0, :])
    ax_flux   = fig.add_subplot(gs[1, 0])
    ax_curv   = fig.add_subplot(gs[1, 1])
    ax_vol    = fig.add_subplot(gs[2, 0])
    ax_pred   = fig.add_subplot(gs[2, 1])

    DARK = "#0d0f14";  GRID = "#1e2130";  TEXT = "#c8cfe8"
    C_PRICE = "#4fc3f7"; C_FLUX = "#81c784"; C_WARN = "#ff7043"
    C_CURV  = "#ce93d8"; C_IMPULSE = "#ffd54f"; C_PRED = "#80deea"

    for ax in [ax_price, ax_flux, ax_curv, ax_vol, ax_pred]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5)

    xlabels = [t.strftime("%b %d") for t in timestamps]
    xt = np.linspace(0, len(timestamps)-1, 8, dtype=int)

    # — Panel 1: Price with impulse markers ——————————————————————————————————
    ax_price.plot(prices, color=C_PRICE, lw=1.5, label="Price")
    ax_price.axhline(true_share_value, color=C_WARN, lw=1,
                     linestyle="--", label=f"True Share Value ${true_share_value:.4f}")
    for i, flag in enumerate(impulse):
        if flag:
            ax_price.axvline(i, color=C_IMPULSE, alpha=0.35, lw=1)
            ax_price.scatter(i, prices[i], color=C_IMPULSE, zorder=5, s=55, marker="^")
    ax_price.set_title("Price  +  Impulse Buy Signals  (▲)", fontsize=10)
    ax_price.set_ylabel(f"Price ({CURRENCY.upper()})")
    ax_price.set_xticks(xt); ax_price.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_price.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT, loc="upper left")

    # — Panel 2: Flux with healthy band ———————————————————————————————————————
    ax_flux.fill_between(range(len(flux)), FLUX_LOW, FLUX_HIGH,
                         alpha=0.12, color=C_FLUX, label="Healthy 1–10% band")
    ax_flux.plot(raw_flux, color=C_FLUX, alpha=0.35, lw=1, label="Raw flux")
    ax_flux.plot(flux, color=C_FLUX, lw=2, label="Smoothed flux")
    ax_flux.axhline(FLUX_LOW,  color=C_FLUX, lw=0.8, linestyle=":")
    ax_flux.axhline(FLUX_HIGH, color=C_WARN, lw=0.8, linestyle=":")
    for i, flag in enumerate(impulse):
        if flag:
            ax_flux.scatter(i, flux[i], color=C_IMPULSE, zorder=5, s=45, marker="^")
    ax_flux.set_title("Daily Volume / Market Cap  (Flux)", fontsize=10)
    ax_flux.set_ylabel("Flux ratio")
    ax_flux.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
    ax_flux.set_xticks(xt); ax_flux.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_flux.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    # — Panel 3: Curvature ————————————————————————————————————————————————————
    ax_curv.fill_between(range(len(curv)), curv, 0,
                         where=(curv > 0), color=C_CURV, alpha=0.4, label="Positive κ")
    ax_curv.fill_between(range(len(curv)), curv, 0,
                         where=(curv < 0), color=C_WARN,  alpha=0.3, label="Negative κ")
    ax_curv.plot(curv, color=C_CURV, lw=1.5)
    ax_curv.axhline(0, color=TEXT, lw=0.6)
    for i, flag in enumerate(impulse):
        if flag:
            ax_curv.scatter(i, curv[i], color=C_IMPULSE, zorder=5, s=55, marker="^")
    ax_curv.set_title("Flux Curvature  κ  (impulse detector)", fontsize=10)
    ax_curv.set_ylabel("κ")
    ax_curv.set_xticks(xt); ax_curv.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_curv.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    # — Panel 4: Volume bars ——————————————————————————————————————————————————
    colors_v = [C_IMPULSE if impulse[i] else C_PRICE for i in range(len(volumes))]
    ax_vol.bar(range(len(volumes)), volumes / 1e9, color=colors_v, width=0.8, alpha=0.8)
    ax_vol.set_title("Daily Volume  (yellow = impulse day)", fontsize=10)
    ax_vol.set_ylabel("Volume (B USD)")
    ax_vol.set_xticks(xt); ax_vol.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))

    # — Panel 5: Flux prediction ——————————————————————————————————————————————
    all_flux  = np.concatenate([flux, future_flux])
    all_dates = list(range(len(flux))) + list(range(len(flux), len(flux)+len(future_flux)))
    ax_pred.plot(range(len(flux)), flux, color=C_FLUX, lw=2, label="Historical flux")
    ax_pred.plot(range(len(flux)-1, len(flux)+len(future_flux)),
                 np.concatenate([[flux[-1]], future_flux]),
                 color=C_PRED, lw=2, linestyle="--", label="14-day prediction")
    ax_pred.fill_between(range(len(flux)-1, len(flux)+len(future_flux)),
                         np.concatenate([[flux[-1]], future_flux]) * 0.85,
                         np.concatenate([[flux[-1]], future_flux]) * 1.15,
                         alpha=0.15, color=C_PRED)
    ax_pred.axhspan(FLUX_LOW, FLUX_HIGH, alpha=0.08, color=C_FLUX)
    ax_pred.axhline(FLUX_LOW,  color=C_FLUX, lw=0.7, linestyle=":")
    ax_pred.axhline(FLUX_HIGH, color=C_WARN, lw=0.7, linestyle=":")
    ax_pred.set_title("Flux Prediction  (+14 days)", fontsize=10)
    ax_pred.set_ylabel("Flux ratio")
    ax_pred.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
    ax_pred.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    # — Footer annotation ————————————————————————————————————————————————————
    fig.text(0.5, 0.005,
             f"True Share Value = Market Cap / Total Supply = ${true_share_value:.6f}  |  "
             f"Current Price = ${current_price:,.2f}  ({overvaluation:+.1f}% premium)  |  "
             f"Impulse events: {impulse.sum()}",
             ha="center", color=TEXT, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=GRID, alpha=0.7))

    out = "crypto_impulse_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"  Chart saved → {out}\n")
    plt.show()


if __name__ == "__main__":
    main()