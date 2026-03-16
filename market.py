"""
Crypto Impulse Buy Analyzer — Gradio App
BCH-gated: requires at least one payment logged within 24 hours to run analysis.
Real-time online users counter via session heartbeat registry.
"""

import gradio as gr
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from datetime import datetime
import json, os, time, threading, uuid

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BCH_ADDRESS   = "bitcoincash:qp5nsh9czn6jadwdzwldgnn86y9358l4rczv0tu37e"
PAYMENTS_FILE = "bch_payments.json"
SESSIONS_FILE = "sessions.json"
FLUX_LOW      = 0.01
FLUX_HIGH     = 0.10
SESSION_TTL   = 30   # seconds — session expires if no heartbeat

POPULAR_COINS = [
    ("Bitcoin",   "bitcoin",    "BTC"),
    ("Ethereum",  "ethereum",   "ETH"),
    ("Solana",    "solana",     "SOL"),
    ("BNB",       "binancecoin","BNB"),
    ("XRP",       "ripple",     "XRP"),
    ("USDC",      "usd-coin",   "USDC"),
    ("Cardano",   "cardano",    "ADA"),
    ("Avalanche", "avalanche-2","AVAX"),
    ("Dogecoin",  "dogecoin",   "DOGE"),
    ("Polkadot",  "polkadot",   "DOT"),
]

# ── THREAD LOCK ───────────────────────────────────────────────────────────────
_lock = threading.Lock()

# ── SESSION / ONLINE USERS ────────────────────────────────────────────────────

def _load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f)

def _purge_stale(sessions):
    cutoff = time.time() - SESSION_TTL
    return {sid: ts for sid, ts in sessions.items() if ts >= cutoff}

def heartbeat(session_id: str) -> int:
    with _lock:
        sessions = _load_sessions()
        sessions[session_id] = time.time()
        sessions = _purge_stale(sessions)
        _save_sessions(sessions)
        return len(sessions)

def get_online_count() -> int:
    with _lock:
        sessions = _purge_stale(_load_sessions())
        _save_sessions(sessions)
        return len(sessions)

def _online_html(count: int) -> str:
    dot   = "🟢" if count > 0 else "⚫"
    label = "user" if count == 1 else "users"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'background:#0d1a0f;border:1px solid #2e7d32;border-radius:8px;'
        f'padding:8px 14px;font-family:monospace;justify-content:center;">'
        f'<span style="font-size:14px">{dot}</span>'
        f'<span style="color:#69f0ae;font-size:16px;font-weight:bold;">{count}</span>'
        f'<span style="color:#a5d6a7;font-size:12px;">{label} online now</span>'
        f'</div>'
    )

# ── PAYMENT STORE ──────────────────────────────────────────────────────────────

def load_payments():
    if os.path.exists(PAYMENTS_FILE):
        try:
            with open(PAYMENTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_payments(payments):
    with open(PAYMENTS_FILE, "w") as f:
        json.dump(payments, f)

def log_payment(donor_note: str):
    with _lock:
        payments = load_payments()
        payments.append({"time": time.time(), "note": donor_note})
        save_payments(payments)

def has_recent_payment() -> bool:
    cutoff = time.time() - 86400
    return any(p["time"] >= cutoff for p in load_payments())

def recent_payment_info():
    cutoff = time.time() - 86400
    recent = [p for p in load_payments() if p["time"] >= cutoff]
    if not recent:
        return None
    latest   = max(recent, key=lambda p: p["time"])
    age_mins = int((time.time() - latest["time"]) / 60)
    return latest, age_mins, len(recent)

# ── ANALYTICS ─────────────────────────────────────────────────────────────────

def curvature(y):
    dy  = np.gradient(y)
    d2y = np.gradient(dy)
    return d2y / (1 + dy**2) ** 1.5

def detect_impulse_buys(flux, curv, flux_lo=FLUX_LOW, flux_hi=FLUX_HIGH, pct=75):
    in_band       = (flux >= flux_lo) & (flux <= flux_hi)
    curv_thresh   = np.percentile(np.abs(curv), pct)
    local_max_idx = argrelextrema(curv, np.greater, order=2)[0]
    impulse       = np.zeros(len(flux), dtype=bool)
    for i in local_max_idx:
        if in_band[i] and curv[i] > curv_thresh:
            impulse[i] = True
    return impulse

def build_prediction(flux, window=7):
    pad    = np.convolve(flux, np.ones(window)/window, mode="valid")
    trend  = np.polyfit(np.arange(len(pad)), pad, 1)
    future = np.polyval(trend, np.arange(len(pad), len(pad) + 14))
    return np.clip(future, 0, None)
def fetch_and_analyze(coin_id, days=90):
    base  = "https://api.coingecko.com/api/v3"
    try:
        chart = requests.get(
            f"{base}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=15,
        )
        chart.raise_for_status()
        chart = chart.json()

        # Explicitly check required keys
        if not isinstance(chart, dict):
            raise KeyError("Chart response is not a dict")
        if "prices" not in chart or not isinstance(chart["prices"], list):
            raise KeyError("Missing 'prices' in response")
        if "market_caps" not in chart or not isinstance(chart["market_caps"], list):
            raise KeyError("Missing 'market_caps' in response")
        if "total_volumes" not in chart or not isinstance(chart["total_volumes"], list):
            raise KeyError("Missing 'total_volumes' in response")

        prices     = np.array([p[1] for p in chart["prices"]])
        volumes    = np.array([v[1] for v in chart["total_volumes"]])
        mcaps      = np.array([m[1] for m in chart["market_caps"]])
        timestamps = [datetime.fromtimestamp(p[0]/1000) for p in chart["prices"]]

        detail = requests.get(f"{base}/coins/{coin_id}", timeout=15).json()

        total_supply     = (detail["market_data"]["total_supply"]
                            or detail["market_data"]["circulating_supply"])
        current_mcap     = mcaps[-1]
        true_share_value = current_mcap / total_supply
        current_price    = prices[-1]
        overvaluation    = (current_price / true_share_value - 1) * 100

        raw_flux    = volumes / mcaps
        flux        = gaussian_filter1d(raw_flux, sigma=1.5)
        curv        = curvature(flux)
        impulse     = detect_impulse_buys(flux, curv)
        future_flux = build_prediction(flux)
        name        = detail["name"]
        sym         = detail["symbol"].upper()

        return (prices, volumes, mcaps, timestamps, flux, raw_flux,
                curv, impulse, future_flux, true_share_value,
                current_price, current_mcap, total_supply, overvaluation, name, sym)

    except Exception as e:
        print("API error:", repr(e))
        raise RuntimeError(f"CoinGecko error: {e}")


def make_chart(coin_id, days):
    (prices, volumes, mcaps, timestamps, flux, raw_flux,
     curv, impulse, future_flux, true_share_value,
     current_price, current_mcap, total_supply, overvaluation, name, sym) = fetch_and_analyze(coin_id, days)

    DARK="#0d0f14"; GRID="#1e2130"; TEXT="#c8cfe8"
    C_PRICE="#4fc3f7"; C_FLUX="#81c784"; C_WARN="#ff7043"
    C_CURV="#ce93d8"; C_IMPULSE="#ffd54f"; C_PRED="#80deea"

    fig = plt.figure(figsize=(16, 11), facecolor=DARK)
    fig.suptitle(
        f"{name} ({sym})  ·  Impulse Buy Detector  ·  {days}-Day Window",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)
    ax_price = fig.add_subplot(gs[0, :])
    ax_flux  = fig.add_subplot(gs[1, 0])
    ax_curv  = fig.add_subplot(gs[1, 1])
    ax_vol   = fig.add_subplot(gs[2, 0])
    ax_pred  = fig.add_subplot(gs[2, 1])

    for ax in [ax_price, ax_flux, ax_curv, ax_vol, ax_pred]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5)

    xlabels = [t.strftime("%b %d") for t in timestamps]
    xt = np.linspace(0, len(timestamps)-1, 8, dtype=int)

    ax_price.plot(prices, color=C_PRICE, lw=1.5, label="Price")
    ax_price.axhline(true_share_value, color=C_WARN, lw=1, linestyle="--",
                     label=f"True Share Value ${true_share_value:.4f}")
    for i, flag in enumerate(impulse):
        if flag:
            ax_price.axvline(i, color=C_IMPULSE, alpha=0.35, lw=1)
            ax_price.scatter(i, prices[i], color=C_IMPULSE, zorder=5, s=55, marker="^")
    ax_price.set_title("Price  +  Impulse Buy Signals  (▲)", fontsize=10)
    ax_price.set_ylabel("Price (USD)"); ax_price.set_xticks(xt)
    ax_price.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_price.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT, loc="upper left")

    ax_flux.fill_between(range(len(flux)), FLUX_LOW, FLUX_HIGH, alpha=0.12, color=C_FLUX)
    ax_flux.plot(raw_flux, color=C_FLUX, alpha=0.35, lw=1, label="Raw flux")
    ax_flux.plot(flux, color=C_FLUX, lw=2, label="Smoothed flux")
    ax_flux.axhline(FLUX_LOW, color=C_FLUX, lw=0.8, linestyle=":")
    ax_flux.axhline(FLUX_HIGH, color=C_WARN, lw=0.8, linestyle=":")
    for i, flag in enumerate(impulse):
        if flag:
            ax_flux.scatter(i, flux[i], color=C_IMPULSE, zorder=5, s=45, marker="^")
    ax_flux.set_title("Daily Volume / Market Cap  (Flux)", fontsize=10)
    ax_flux.set_ylabel("Flux ratio")
    ax_flux.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
    ax_flux.set_xticks(xt); ax_flux.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_flux.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    ax_curv.fill_between(range(len(curv)), curv, 0, where=(curv>0), color=C_CURV, alpha=0.4, label="Positive κ")
    ax_curv.fill_between(range(len(curv)), curv, 0, where=(curv<0), color=C_WARN,  alpha=0.3, label="Negative κ")
    ax_curv.plot(curv, color=C_CURV, lw=1.5)
    ax_curv.axhline(0, color=TEXT, lw=0.6)
    for i, flag in enumerate(impulse):
        if flag:
            ax_curv.scatter(i, curv[i], color=C_IMPULSE, zorder=5, s=55, marker="^")
    ax_curv.set_title("Flux Curvature  κ  (impulse detector)", fontsize=10)
    ax_curv.set_ylabel("κ"); ax_curv.set_xticks(xt)
    ax_curv.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_curv.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    colors_v = [C_IMPULSE if impulse[i] else C_PRICE for i in range(len(volumes))]
    ax_vol.bar(range(len(volumes)), volumes / 1e9, color=colors_v, width=0.8, alpha=0.8)
    ax_vol.set_title("Daily Volume  (yellow = impulse day)", fontsize=10)
    ax_vol.set_ylabel("Volume (B USD)"); ax_vol.set_xticks(xt)
    ax_vol.set_xticklabels([xlabels[i] for i in xt], rotation=20)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))

    ax_pred.plot(range(len(flux)), flux, color=C_FLUX, lw=2, label="Historical flux")
    ax_pred.plot(range(len(flux)-1, len(flux)+len(future_flux)),
                 np.concatenate([[flux[-1]], future_flux]),
                 color=C_PRED, lw=2, linestyle="--", label="14-day prediction")
    ax_pred.fill_between(range(len(flux)-1, len(flux)+len(future_flux)),
                         np.concatenate([[flux[-1]], future_flux]) * 0.85,
                         np.concatenate([[flux[-1]], future_flux]) * 1.15,
                         alpha=0.15, color=C_PRED)
    ax_pred.axhspan(FLUX_LOW, FLUX_HIGH, alpha=0.08, color=C_FLUX)
    ax_pred.set_title("Flux Prediction  (+14 days)", fontsize=10)
    ax_pred.set_ylabel("Flux ratio")
    ax_pred.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
    ax_pred.legend(fontsize=7, facecolor=GRID, labelcolor=TEXT)

    fig.text(0.5, 0.005,
             f"True Share Value = ${true_share_value:.6f}  |  "
             f"Current Price = ${current_price:,.2f}  ({overvaluation:+.1f}% premium)  |  "
             f"Impulse events: {impulse.sum()}",
             ha="center", color=TEXT, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=GRID, alpha=0.7))

    # ── FIX: buffer_rgba works in matplotlib 3.8+ (tostring_rgb removed) ─────
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3]   # RGBA → RGB
    plt.close(fig)

    summary = (
        f"<div style='color:white; font-size:13px;'>"
        f"### {name} ({sym})\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Current Price | ${current_price:,.2f} |\n"
        f"| Market Cap | ${current_mcap:,.0f} |\n"
        f"| Total Supply | {total_supply:,.0f} |\n"
        f"| True Share Value | ${true_share_value:.6f} |\n"
        f"| Price Premium | {overvaluation:+.1f}% |\n"
        f"| Avg Daily Flux | {flux.mean()*100:.2f}% |\n"
        f"| Impulse Events | {impulse.sum()} in {days} days |\n"
    )
    return img, summary


# ── GRADIO HANDLERS ───────────────────────────────────────────────────────────

def on_load(request: gr.Request):
    """Assign session, register heartbeat, return initial UI state."""
    sid   = str(uuid.uuid4())
    count = heartbeat(sid)
    html  = _online_html(count)

    if has_recent_payment():
        info = recent_payment_info()
        if info:
            latest, age_mins, n = info
            remaining  = 24*60 - age_mins
            status_msg = f"✅ Access active — last payment {age_mins}m ago, expires in {remaining}m."
            return sid, gr.update(visible=False), gr.update(visible=True), status_msg, html
    return sid, gr.update(visible=True), gr.update(visible=False), "", html

def do_heartbeat(session_id: str):
    if not session_id:
        return _online_html(get_online_count())
    count = heartbeat(session_id)
    return _online_html(count)

def submit_payment(donor_note, session_id):
    if not donor_note.strip():
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Please enter your name or a note first."
    log_payment(donor_note.strip())
    heartbeat(session_id)
    return (gr.update(visible=False), gr.update(visible=True),
            f"✅ Thank you, {donor_note.strip()}! Access unlocked for 24 hours.")

def run_analysis(coin_id, days, session_id):
    if not has_recent_payment():
        return (None,
                f"<div style='color:white; font-size:13px;'>"
                "### 🔒 Access Locked\n"
                "No BCH payment in the last 24 hours. "
                "Send BCH and click **I've sent BCH** to unlock.")
    try:
        img, summary = make_chart(coin_id, int(days))
        info = recent_payment_info()
        if info:
            latest, age_mins, n = info
            summary += f"\n\n*🟢 Access active — last donation {age_mins}m ago ({n} today). Thank you!*"
        return img, summary
    except Exception as e:
        return None, f"### ❌ Error\n```\n{e}\n```\nCheck connection or try again."


# ── BUILD UI ──────────────────────────────────────────────────────────────────

COIN_CHOICES = [(f"{n} ({s})", cid) for n, cid, s in POPULAR_COINS]

CSS = """
body { background: #0d0f14 !important; }
.gradio-container { background: #0d0f14 !important; color: #c8cfe8 !important; font-family: 'Courier New', monospace; }
.panel-box { background: #12151e; border: 1px solid #2a2f45; border-radius: 10px; padding: 18px; }
.bch-box { background: #0e1a14; border: 2px solid #2e7d32; border-radius: 10px; padding: 18px; }
footer { display: none !important; }
label { color: #8892b0 !important; }
"""

with gr.Blocks(css=CSS, title="Crypto Impulse Buy Analyzer") as demo:

    session_id = gr.State("")

    # ── Header row with online counter ───────────────────────────────────────
    with gr.Row(equal_height=True):
        gr.HTML("""
        <div style="flex:1;padding:20px 0 6px;">
          <h1 style="color:#4fc3f7;font-size:26px;letter-spacing:2px;margin:0;">
            ₿ CRYPTO IMPULSE BUY ANALYZER
          </h1>
          <p style="color:#8892b0;font-size:12px;margin:4px 0 0;">
            Volume/MCap flux curvature · True share value · Impulse signal detection
          </p>
        </div>
        """)
        with gr.Column(scale=0, min_width=210):
            online_display = gr.HTML(
                _online_html(get_online_count()),
                elem_id="online_counter",
            )

    # ── Heartbeat timer (every 15 s) ──────────────────────────────────────────
    hb_timer = gr.Timer(value=15)
    hb_timer.tick(fn=do_heartbeat, inputs=[session_id], outputs=[online_display])

    with gr.Row():

        # ── LEFT sidebar ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            with gr.Group(elem_classes="bch-box") as bch_panel:
                gr.HTML("""
                <div style="text-align:center;margin-bottom:12px;">
                  <div style="font-size:30px;">🟢</div>
                  <h3 style="color:#81c784;margin:4px 0;">Keep This Tool Alive</h3>
                  <p style="color:#a5d6a7;font-size:13px;line-height:1.6;margin:0;">
                    Community-funded. Send <strong>any amount of BCH</strong><br>
                    to unlock 24-hour access for <em>everyone</em>.
                  </p>
                </div>
                """)
                gr.HTML(f"""
                <div style="background:#0a120a;border:1px solid #2e7d32;border-radius:8px;
                            padding:12px;margin:10px 0;text-align:center;">
                  <p style="color:#69f0ae;font-size:10px;margin:0 0 5px;letter-spacing:1px;">BCH ADDRESS</p>
                  <code style="color:#b9f6ca;font-size:10px;word-break:break-all;">{BCH_ADDRESS}</code>
                </div>
                """)
                donor_name = gr.Textbox(
                    label="Your name or note (required)",
                    placeholder="e.g. Alice — happy to support!",
                    lines=1,
                )
                pay_btn    = gr.Button("✅ I've sent BCH — Unlock Access", variant="primary")
                pay_status = gr.Markdown("")

            with gr.Group(elem_classes="panel-box", visible=False) as unlocked_panel:
                gr.HTML("""
                <div style="text-align:center;">
                  <div style="font-size:26px;">🔓</div>
                  <h3 style="color:#4fc3f7;margin:4px 0;">Access Unlocked</h3>
                  <p style="color:#8892b0;font-size:12px;">
                    Run analyses freely for 24 hours.<br>Thank you for keeping this alive!
                  </p>
                </div>
                """)
                gr.HTML(f"""
                <div style="background:#0a120a;border:1px solid #2e7d32;border-radius:8px;
                            padding:10px;margin:8px 0;text-align:center;">
                  <p style="color:#a5d6a7;font-size:11px;margin:0 0 4px;">Tip or extend access:</p>
                  <code style="color:#b9f6ca;font-size:10px;word-break:break-all;">{BCH_ADDRESS}</code>
                </div>
                """)

            gr.HTML("<hr style='border-color:#1e2130;margin:14px 0;'>")

            with gr.Group(elem_classes="panel-box"):
                gr.HTML("<p style='color:#4fc3f7;font-weight:bold;margin:0 0 10px;'>⚙️ Analysis Settings</p>")
                coin_dd = gr.Dropdown(choices=COIN_CHOICES, value="bitcoin", label="Coin")
                days_sl = gr.Slider(30, 365, value=90, step=1, label="Lookback Days")
                run_btn = gr.Button("▶ Run Analysis", variant="primary")

            with gr.Group(elem_classes="panel-box"):
                gr.HTML("<p style='color:#4fc3f7;font-weight:bold;margin:0 0 8px;'>⚡ Quick Select</p>")
                with gr.Row():
                    for n, cid, sym in POPULAR_COINS[:5]:
                        b = gr.Button(sym, size="sm")
                        b.click(lambda c=cid: gr.update(value=c), outputs=coin_dd)
                with gr.Row():
                    for n, cid, sym in POPULAR_COINS[5:]:
                        b = gr.Button(sym, size="sm")
                        b.click(lambda c=cid: gr.update(value=c), outputs=coin_dd)

        # ── RIGHT chart area ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            chart_img  = gr.Image(label="Analysis Chart", show_label=False, height=600)
            summary_md = gr.Markdown(
                f"<div style='color:white; font-size:13px;'>"
                "### Select a coin and click **▶ Run Analysis**\n"
                "_A BCH donation in the last 24 h is required to run._"
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    demo.load(
        on_load,
        outputs=[session_id, bch_panel, unlocked_panel, pay_status, online_display],
    )
    pay_btn.click(
        submit_payment,
        inputs=[donor_name, session_id],
        outputs=[bch_panel, unlocked_panel, pay_status],
    )
    run_btn.click(
        run_analysis,
        inputs=[coin_dd, days_sl, session_id],
        outputs=[chart_img, summary_md],
    )

    gr.HTML("""
    <div style="text-align:center;padding:12px 0 2px;color:white;font-size:11px;">
      Data via CoinGecko · True Share Value = Market Cap / Total Supply ·
      Impulse signals = curvature spikes in volume/MCap flux within 1–10% band
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
