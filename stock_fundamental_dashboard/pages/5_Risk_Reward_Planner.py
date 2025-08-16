from auth_gate import require_auth
require_auth()

# --- make project root importable so we can import io_helpers/calculations/rules ---
import os, sys
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))         # .../project root (parent of /pages)
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))   # repo root if pages is nested deeper

for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers
    import calculations
    import rules

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta


# --- EV helpers (based on your closed_trades history) ---
def _strategy_stats(strategy_name: str):
    df = io_helpers.load_closed_trades()
    if df is None or df.empty:
        return {"wr": 0.5, "rwin": 1.5, "rloss": 1.0, "n": 0}
    sdf = df[df["Strategy"] == strategy_name] if strategy_name else df
    if sdf.empty:
        return {"wr": 0.5, "rwin": 1.5, "rloss": 1.0, "n": 0}
    r = sdf.get("RMultiple")
    if r is None or r.dropna().empty:
        return {"wr": 0.5, "rwin": 1.5, "rloss": 1.0, "n": len(sdf)}
    r = r.dropna()
    wins = r[r > 0]; losses = r[r <= 0]
    wr = len(wins) / len(r) if len(r) else 0.5
    rwin = wins.mean() if len(wins) else 1.5
    rloss = abs(losses.mean()) if len(losses) else 1.0
    return {"wr": wr, "rwin": float(rwin), "rloss": float(rloss), "n": len(sdf)}

def _ev(wr, rwin, rloss):
    return wr*rwin - (1-wr)*rloss


# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers, calculations, rules

# ---------- Page setup ----------
st.set_page_config(page_title="Risk / Reward Planner", layout="wide")

# === Unified CSS (same as Dashboard; fonts 16px) ===
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb;               /* app background */
  --surface:#ffffff;          /* cards/tables background */
  --text:#0f172a;             /* main text */
  --muted:#475569;            /* secondary text */
  --border:#e5e7eb;           /* card & table borders */
  --shadow:0 8px 24px rgba(15, 23, 42, .06);

  /* accent colors for section stripes */
  --primary:#4f46e5;          /* indigo */
  --info:#0ea5e9;             /* sky   */
  --success:#10b981;          /* green */
  --warning:#f59e0b;          /* amber */
  --danger:#ef4444;           /* red   */
}
html, body, [class*="css"]{
  font-size:16px !important; color:var(--text);
}
.stApp{
  background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg);
}
h1, h2, h3, h4{
  color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px;
}

/* Section header card (visual separators) */
.sec{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  box-shadow:var(--shadow);
  padding:.65rem .9rem;
  margin:1rem 0 .6rem 0;
  display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{
  content:""; display:inline-block;
  width:8px; height:26px; border-radius:6px; background:var(--primary);
}
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }

/* Tables / editors */
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] table, div[data-testid="stDataEditor"] table{
  border-collapse:separate !important; border-spacing:0;
}
div[data-testid="stDataFrame"] table tbody tr:hover td,
div[data-testid="stDataEditor"] table tbody tr:hover td{
  background:#f8fafc !important;
}
div[data-testid="stDataFrame"] td, div[data-testid="stDataEditor"] td{
  border-bottom:1px solid var(--border) !important;
}

/* Inputs & buttons */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar theme (dark, same as Dashboard) */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("📐 Risk / Reward Planner")

# ========== 0) Pick an existing queued row (QUEUED ONLY) ==========
qdf_raw = io_helpers.load_trade_queue()
if qdf_raw is None or qdf_raw.empty:
    st.warning("Trade Queue is empty. Go to **Systematic Decision** and push ideas to queue first.")
    st.stop()

queue_df = qdf_raw.reset_index().rename(columns={"index": "RowId"})

options = ["— Choose queued idea —"] + [
    f"{int(r.RowId)} — {r.Name} ({r.Strategy})"
    for _, r in queue_df.iterrows()
]
sel = st.selectbox("Queued idea to edit", options, index=0)

# Gate: must pick a queued idea (no new plans)
if sel == "— Choose queued idea —":
    st.info("Pick a **queued idea** to continue. New plans are disabled (systematic mode).")
    st.stop()

# Prefill from the selected queue row
editing_rowid = int(sel.split(" — ")[0])
r = queue_df.loc[queue_df.RowId == editing_rowid].iloc[0]
prefill = {
    "stock":    r.Name,
    "strategy": r.Strategy,
    "entry":    r.Entry,
    "stop":     r.Stop,
    "take":     r.Take,
    "r":        r.RR,
}



# ======= Query params fallback (old links still work) =======
try:
    qp = dict(st.query_params)
except Exception:
    qp = {k: v[0] for k, v in st.experimental_get_query_params().items()}

def qget(k, cast=None, default=None):
    """queued-row values override URL params"""
    if k in prefill and prefill[k] is not None and not (isinstance(prefill[k], float) and np.isnan(prefill[k])):
        return prefill[k]
    v = qp.get(k, default)
    if v is None: return default
    if cast is None: return v
    try: return cast(v)
    except Exception: return default

# =============== Helpers ===============
def fmt(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    try:
        return (f"{x:,.{d}f}%" if pct else f"{x:,.{d}f}")
    except Exception:
        return "N/A"

def rr_band(rr):
    if rr is None or not np.isfinite(rr): return "N/A", "#64748b"
    if rr < 1.5:  return "Low (<1.5)", "#ef4444"
    if rr < 2.0:  return "OK (1.5–2.0)", "#f59e0b"
    return "Good (≥2.0)", "#16a34a"

def latest_current_price(stock_rows: pd.DataFrame) -> float | None:
    cur_val = None
    if "CurrentPrice" in stock_rows.columns:
        s = stock_rows["CurrentPrice"].dropna()
        if not s.empty:
            cur_val = s.iloc[-1]
    if cur_val is None:
        annual = stock_rows[stock_rows.get("IsQuarter", False) != True]
        if "SharePrice" in annual.columns and not annual.empty:
            s2 = annual.sort_values("Year")["SharePrice"].dropna()
            if not s2.empty:
                cur_val = s2.iloc[-1]
    try:
        return float(cur_val) if cur_val is not None else None
    except Exception:
        return None

def get_latest_annual_row(stock_rows: pd.DataFrame) -> pd.Series | None:
    annual = stock_rows[stock_rows.get("IsQuarter", False) != True].copy()
    if annual.empty or "Year" not in annual.columns:
        return None
    annual = annual.dropna(subset=["Year"]).sort_values("Year")
    return annual.iloc[-1] if not annual.empty else None

# =============== Data load ===============
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add/Edit** first.")
    st.stop()

stocks = sorted([s for s in df["Name"].dropna().unique()])

# Prefill selections
stock_q    = qget("stock",    str, None)
strategy_q = qget("strategy", str, None)
entry_q    = qget("entry",    float, None)
stop_q     = qget("stop",     float, None)
take_q     = qget("take",     float, None)
r_q        = qget("r",        float, None)

# ─────────────────────────────────────────
# 1) Pick a Stock & Strategy
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">1) Pick a Stock & Strategy</div>'
    '<div class="d">Select an idea from your Trade Queue (new plans disabled)</div></div>',
    unsafe_allow_html=True
)

colA, colB = st.columns([2, 1])

with colA:
    stock_index = stocks.index(stock_q) if (stock_q in stocks) else 0
    stock_name = st.selectbox("Stock", options=stocks, index=stock_index, key="rr_stock")

with colB:
    strategies = list(rules.RULESETS.keys())
    strat_index = strategies.index(strategy_q) if (strategy_q in strategies) else 0
    strategy = st.selectbox("Strategy", options=strategies, index=strat_index, key="rr_strategy")

# Lookups for current price & decision context
stock_rows = df[df["Name"] == stock_name].sort_values(["Year"])
price_now  = latest_current_price(stock_rows)
latest_row = get_latest_annual_row(stock_rows)
metrics    = calculations.calc_ratios(latest_row) if latest_row is not None else {}
ev         = rules.evaluate(metrics, strategy) if metrics else {"score": 0, "pass": False, "reasons": []}

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Current Price", (f"{price_now:,.4f}" if price_now is not None else "N/A"))
with c2: st.metric("Decision Score", f"{ev.get('score',0)}%")
with c3: st.metric("Decision Status", "✅ PASS" if ev.get("pass") else "❌ REJECT")
with c4:
    if (not ev.get("pass")) and ev.get("reasons"):
        st.write("Unmet: " + "; ".join(ev["reasons"]))
    else:
        st.write("Looks OK by chosen strategy.")

st.divider()

# Try to fetch latest ATR if OHLC helper exists
atr_fn = getattr(io_helpers, "latest_atr", None)
atr_period_default = 14
atr_val, atr_date = (atr_fn(stock_name, period=atr_period_default) if callable(atr_fn) else (None, None))
atr_available = atr_val is not None

# ─────────────────────────────────────────
# 2) Define Entry / Stop / Take & Risk
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">2) Define Your Entry / Stop / Take & Risk</div>'
    '<div class="d">ATR stop available when OHLC is present</div></div>',
    unsafe_allow_html=True
)

cA, cB, cC, cD = st.columns(4)

default_entry = (
    float(entry_q) if entry_q is not None else
    float(prefill.get("entry")) if prefill.get("entry") is not None and not pd.isna(prefill.get("entry")) else
    float(price_now) if price_now is not None else 0.0
)

with cA:
    entry = st.number_input("Entry Price", min_value=0.0, value=default_entry, step=0.001, format="%.4f", key="rr_entry")

with cB:
    stop_modes = ["% below entry", "Manual price"] + (["ATR-based"] if atr_available else [])
    stop_mode = st.radio("Stop-loss Mode", stop_modes, horizontal=True, key="rr_stopmode")

    if stop_mode == "% below entry" and (stop_q is None and pd.isna(prefill.get("stop"))):
        stop_pct = st.number_input("Stop % below Entry", min_value=0.1, max_value=50.0, value=8.0, step=0.1, format="%.1f", key="rr_stoppct")
        stop = entry * (1 - stop_pct / 100.0)
    elif stop_mode == "Manual price" or (stop_q is not None) or (prefill.get("stop") is not None and not pd.isna(prefill.get("stop"))):
        stop_default = (
            float(stop_q) if stop_q is not None else
            float(prefill.get("stop")) if prefill.get("stop") is not None and not pd.isna(prefill.get("stop")) else
            max(0.0, entry * 0.92)
        )
        stop = st.number_input("Stop-loss Price", min_value=0.0, value=stop_default, step=0.001, format="%.4f", key="rr_stop")
    else:
        # ATR-based
        st.markdown(f"**ATR({atr_period_default})** latest ≈ **{(f'{atr_val:,.4f}' if atr_val is not None else 'N/A')}**"
                    + (f" (as of {pd.to_datetime(atr_date).date()})" if atr_date is not None else ""))
        atr_mult = st.number_input("ATR Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.5, format="%.1f", key="rr_atr_mult")
        stop = max(0.0, entry - (atr_val or 0) * atr_mult) if atr_available else max(0.0, entry * 0.92)

with cC:
    take_mode = st.radio("Take-profit Mode", ["R multiple", "Manual price"], horizontal=True, key="rr_tpmode")
    if take_mode == "R multiple" and (take_q is None and pd.isna(prefill.get("take"))):
        r_default = float(r_q) if r_q is not None else float(prefill.get("r")) if prefill.get("r") is not None and not pd.isna(prefill.get("r")) else 2.0
        r_multiple = st.number_input("Target (R)", min_value=0.5, max_value=10.0, value=r_default, step=0.5, format="%.1f", key="rr_rmult")
        risk_per_share = max(entry - stop, 0.0)
        take = entry + r_multiple * risk_per_share if risk_per_share > 0 else entry
    else:
        take_default = (
            float(take_q) if take_q is not None else
            float(prefill.get("take")) if prefill.get("take") is not None and not pd.isna(prefill.get("take")) else
            (entry * 1.15 if entry else 0.0)
        )
        take = st.number_input("Take-profit Price", min_value=0.0, value=take_default, step=0.001, format="%.4f", key="rr_take")

with cD:
    acct = st.number_input(
        "Account Size (MYR)",
        min_value=0.0, value=10000.0, step=100.0, format="%.2f",
        key="rr_acct"
    )
    # Cap risk at 30% (you can choose lower, but not higher)
    risk_pct = st.number_input(
        "Risk % per Trade",
        min_value=0.1, max_value=30.0, value=1.0, step=0.1, format="%.1f",
        key="rr_riskpct"
    )
    lot_size = st.number_input(
        "Lot Size (shares)",
        min_value=1, value=100, step=1,
        help="Shares per 1 lot (e.g., 100). Do NOT put how many lots you want to buy here.",
        key="rr_lot"
    )
    planned_lots = st.number_input(
        "Planned lots to buy",
        min_value=0, value=0, step=1,
        help="How many lots you intend to buy. 'Final allowed' will show remaining lots after this.",
        key="rr_planned_lots"
    )
    st.caption("Tip: If you want 13 lots, keep Lot Size=100 and set **Planned lots to buy = 13**.")

# ---- Planned sizing (define early; used by Safety locks etc.) ----
lots_planned_now = int(st.session_state.get("rr_planned_lots", 0) or 0)
planned_shares   = int(lots_planned_now * lot_size)


# ==== Limits & Caps (risk/cost) ====
if entry and entry > 0 and stop is not None:
    rps = max(entry - stop, 0.0)                       # risk/share
    cash_risk = acct * (risk_pct / 100.0)              # MYR

    # Max by risk (rounded to lot size)
    max_sh_risk_raw = math.floor(cash_risk / rps) if rps > 0 else 0
    max_sh_risk = max((max_sh_risk_raw // lot_size) * lot_size, 0)
    lots_risk = (max_sh_risk // lot_size) if lot_size else 0

    # Max by cost/buying power (rounded to lot size)
    max_sh_cost = max((math.floor(acct / entry) // lot_size) * lot_size, 0)
    lots_cost = (max_sh_cost // lot_size) if lot_size else 0

    # Planned lots input
    lots_planned = int(st.session_state.get("rr_planned_lots", 0) or 0)

    # 👉 Final allowed driven by COST ONLY (as requested)
    final_allowed_lots = max(lots_cost - lots_planned, 0)
    final_allowed_shares = final_allowed_lots * lot_size

    # Display as a compact 4-metric row
    m1, m2, m3, m4 = st.columns([1.3, 1.7, 1.7, 1.7])
    with m1: st.metric("Cash risk budget (MYR)", f"{cash_risk:,.2f}")
    with m2: st.metric("Max by risk", f"{max_sh_risk:,} sh (≈ {lots_risk} lots)")
    with m3: st.metric("Max by cost", f"{max_sh_cost:,} sh (≈ {lots_cost} lots)")
    with m4: st.metric("Final allowed", f"{final_allowed_shares:,} sh (≈ {final_allowed_lots} lots)")

    # Warnings
    if lots_planned > 0 and rps > 0 and lots_planned > lots_risk:
        st.warning(f"Planned lots **{lots_planned}** exceed your **risk cap** of {lots_risk} lots. Reduce size or tighten stop.")
    if lots_planned > lots_cost:
        st.error(f"Planned lots **{lots_planned}** exceed your **buying power** cap of {lots_cost} lots.")
else:
    st.info("Enter Entry & Stop to see risk, caps and allowed size.")



# =============== Calculations ===============
def valid_prices(entry, stop, take):
    if entry is None or stop is None or take is None:
        return False, "Please provide Entry, Stop, and Take-Profit."
    if stop >= entry:
        return False, "Stop-loss must be **below** Entry for a long setup."
    if take <= entry:
        return False, "Take-profit must be **above** Entry for a long setup."
    return True, ""

ok, msg = valid_prices(entry, stop, take)
if not ok:
    st.warning(msg)
    st.stop()

risk_per_share   = entry - stop
reward_per_share = take - entry
rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else None

cash_risk = acct * (risk_pct / 100.0)

# Risk-based sizing (STOP is already used: risk_per_share = entry - stop)
raw_shares = math.floor(cash_risk / risk_per_share) if (risk_per_share and risk_per_share > 0) else 0
shares_by_risk = max((raw_shares // lot_size) * lot_size, 0)

# Cost-based cap: cannot spend more than Account Size
max_sh_by_cost = 0
if entry and entry > 0:
    max_sh_by_cost = max((math.floor(acct / entry) // lot_size) * lot_size, 0)

# Final shares must satisfy BOTH caps
shares = min(shares_by_risk, max_sh_by_cost) if max_sh_by_cost > 0 else shares_by_risk

# Optional: tell user if the cost cap binds
if max_sh_by_cost > 0 and shares_by_risk > max_sh_by_cost:
    st.info("Shares limited by account buying power (cost cap).")
elif shares_by_risk > 0 and shares == shares_by_risk and (max_sh_by_cost == 0 or shares_by_risk <= max_sh_by_cost):
    st.info("Shares limited by your risk cap.")


position_cost  = shares * entry
potential_loss = shares * (entry - stop)
potential_gain = shares * (take - entry)


# Band & viability
band_label, band_color = rr_band(rr)
viability = ("✅ Ready" if (rr is not None and np.isfinite(rr) and rr >= 1.5 and shares > 0)
             else "⚠️ Low R or zero size" if (rr is not None and rr < 1.5)
             else "⏳ Incomplete")
st.markdown(
    f"""<div style="display:inline-block;padding:.25rem .6rem;border-radius:999px;
        background:{band_color};color:white;font-weight:700;margin:.25rem 0;">
        R:R Band — {band_label}</div>""",
    unsafe_allow_html=True
)

# Band & viability
band_label, band_color = rr_band(rr)
viability = ("✅ Ready" if (rr is not None and np.isfinite(rr) and rr >= 1.5 and shares > 0)
             else "⚠️ Low R or zero size" if (rr is not None and rr < 1.5)
             else "⏳ Incomplete")
st.markdown(
    f"""<div style="display:inline-block;padding:.25rem .6rem;border-radius:999px;
        background:{band_color};color:white;font-weight:700;margin:.25rem 0;">
        R:R Band — {band_label}</div>""",
    unsafe_allow_html=True
)

# ---- Strategy EV (based on your history) ----
stats = _strategy_stats(strategy)
ev_r = _ev(stats["wr"], stats["rwin"], stats["rloss"])
st.metric(
    "Strategy EV (per 1R)",
    f"{(ev_r if ev_r is not None else 0):.2f}R",
    help=f"WinRate≈{stats['wr']*100:.1f}% | AvgWin≈{stats['rwin']:.2f}R | AvgLoss≈{stats['rloss']:.2f}R | N={stats['n']}"
)

# ---- Safety locks (fixed EV threshold = -0.25R, lock = 7 days) ----
min_rr = st.slider(
    "Minimum allowed R:R", 1.2, 3.0, 1.8, 0.1,
    help="Block Save/Update if plan R:R is below this"
)

# Planned size must be > 0 and R:R must meet the slider
rr_ok_planned = bool(rr is not None and np.isfinite(rr) and rr >= min_rr and planned_shares > 0)

# Fixed EV rule
ev_thresh = -0.25     # -0.25R threshold (no UI control)
ev_block_days = 7     # 7-day lock

# Current EV status
ev_bad = (ev_r is not None and np.isfinite(ev_r) and ev_r < ev_thresh)

# Per-strategy block key
block_key = f"ev_block_until_{strategy or 'ALL'}"
now = datetime.now()
block_until = st.session_state.get(block_key, None)

# Start/refresh a 7-day block window when EV < -0.25R
if ev_bad and (not block_until or now >= block_until):
    st.session_state[block_key] = now + timedelta(days=ev_block_days)
    block_until = st.session_state[block_key]

# Active block?
blocked_active = bool(block_until and now < block_until)

# Note: translate EV and threshold to MYR per trade (using planned 1R)
oneR_money = planned_shares * max(entry - stop, 0.0)
if oneR_money > 0 and ev_r is not None and np.isfinite(ev_r):
    st.caption(
        f"Note: EV ≈ {ev_r:.2f}R → {ev_r*oneR_money:,.2f} MYR per trade. "
        f"Threshold {ev_thresh:.2f}R ≈ {ev_thresh*oneR_money:,.2f} MYR per trade."
    )

# Warnings / lock message
if planned_shares <= 0:
    st.warning("Planned lots is 0. Set **Planned lots to buy** to enable Save/Update.")
elif not rr_ok_planned:
    st.warning("R:R below minimum. Increase take-profit or tighten stop.")
if ev_bad:
    st.warning(f"Strategy EV {ev_r:.2f}R is below threshold ({ev_thresh:.2f}R).")
if blocked_active:
    st.error(f"EV block active for '{strategy}'. Unlocks on {block_until.strftime('%Y-%m-%d %H:%M')}.")

# We'll use rr_ok_planned & blocked_active below




# ─────────────────────────────────────────
# 2a) Optional Multi-Target Take-Profits
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec warning"><div class="t">2a) Optional Multi-Target Take-Profits</div>'
    '<div class="d">Define TP1/TP2/TP3 by R multiples</div></div>',
    unsafe_allow_html=True
)
m1, m2, m3 = st.columns(3)
with m1:
    tp1_r = st.number_input("TP1 (R)", min_value=0.25, max_value=10.0, value=1.0, step=0.25, format="%.2f", key="tp1r")
with m2:
    tp2_r = st.number_input("TP2 (R)", min_value=0.25, max_value=10.0, value=2.0, step=0.25, format="%.2f", key="tp2r")
with m3:
    tp3_r = st.number_input("TP3 (R)", min_value=0.25, max_value=10.0, value=3.0, step=0.25, format="%.2f", key="tp3r")

TP1 = entry + tp1_r * risk_per_share
TP2 = entry + tp2_r * risk_per_share
TP3 = entry + tp3_r * risk_per_share

tp_table = pd.DataFrame({
    "Target": ["TP1", "TP2", "TP3"],
    "Multiple (R)": [tp1_r, tp2_r, tp3_r],
    "Price": [TP1, TP2, TP3],
    "Gain/Share": [TP1 - entry, TP2 - entry, TP3 - entry],
})
st.dataframe(tp_table.round(4), use_container_width=True, height=180)

# ─────────────────────────────────────────
# 3) Results (metrics table)
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec info"><div class="t">3) Result</div>'
    '<div class="d">Key metrics & projected P/L</div></div>',
    unsafe_allow_html=True
)
# KPIs must follow the user's *planned* lots
lots_planned   = int(st.session_state.get("rr_planned_lots", 0) or 0)
planned_shares = int(lots_planned * lot_size)
planned_cost   = planned_shares * entry
planned_loss   = planned_shares * max(entry - stop, 0.0)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric("Stop-loss", fmt(stop, 4))
with k2: st.metric("Take-profit", fmt(take, 4))
with k3: st.metric("R : R", f"{rr:.2f}" if rr is not None and np.isfinite(rr) else "N/A")
with k4: st.metric("Shares", f"{planned_shares:,d}")
with k5: st.metric("Position Cost", f"{planned_cost:,.2f}")
with k6: st.metric("Risk (MYR)", f"{planned_loss:,.2f}")


# --- Summary uses the user's *planned* lots (2-decimal display) ---
# planned_shares / planned_cost / planned_loss already computed above

planned_gain = planned_shares * max(take - entry, 0.0)  # gain at your chosen take

# ATR-based loss (only if ATR available)
atr_stop_price = None
atr_loss = None
if atr_available and (atr_val is not None):
    atr_mult_used = float(st.session_state.get("rr_atr_mult", 2.0) or 2.0)
    atr_stop_price = max(0.0, entry - atr_val * atr_mult_used)
    atr_loss = planned_shares * max(entry - atr_stop_price, 0.0)

detail = pd.DataFrame(
    {
        "Entry": [entry],
        "Stop": [stop],
        "Take": [take],
        "Risk/Share": [risk_per_share],
        "Reward/Share": [reward_per_share],
        "R:R": [rr],

        # 🔢 Planned sizing
        "Planned Lots": [lots_planned],                 # keep as int
        "Planned Shares": [planned_shares],             # keep as int
        "Planned Cost (MYR)": [planned_cost],
        "Planned Loss @ Stop (MYR)": [planned_loss],
        "Planned Gain @ Take (MYR)": [planned_gain],

        # 🧯 Extra risk readouts
        "Loss @ ATR Stop (MYR)": [atr_loss],
        "Max Risk Budget (MYR)": [cash_risk],

        # Targets
        "TP1": [TP1],
        "TP2": [TP2],
        "TP3": [TP3],

        "Viability": [viability],
    }
).T
detail.columns = [stock_name]

# Full-size, no scroll. Show 2 decimals for numbers (ints remain ints).
detail_fmt = detail.copy()
for i, v in detail_fmt[stock_name].items():
    if isinstance(v, (int, np.integer)):
        continue
    try:
        detail_fmt.at[i, stock_name] = round(float(v), 2) if v is not None and np.isfinite(v) else v
    except Exception:
        pass

st.table(detail_fmt)


# ─────────────────────────────────────────
# 4) Save / Update
# ─────────────────────────────────────────
st.markdown(
    '<div class="sec success"><div class="t">4) Save Plan</div>'
    '<div class="d">Add NEW queue row or UPDATE the selected queued idea</div></div>',
    unsafe_allow_html=True
)

summary_lines = [
    f"Plan for {stock_name} ({strategy})",
    f"Entry={entry:.4f}, Stop={stop:.4f}, Take={take:.4f}",
    f"R:R={rr:.2f}" if rr is not None and np.isfinite(rr) else "R:R=N/A",
    f"Shares={planned_shares:,d}, Cost={planned_cost:,.2f}, Risk={planned_loss:,.2f}, Gain={planned_gain:,.2f}",
    f"TP1={TP1:.4f}, TP2={TP2:.4f}, TP3={TP3:.4f}",
    f"Band={band_label}, Viability={viability}",
]
reason_text = " | ".join(summary_lines)

colL, colR = st.columns([1, 1])
with colL:
    st.text_area("Plan summary (stored as `Reasons`)", value=reason_text, height=90, key="rr_reason")

# UPDATED: include rr_ok and ev_ok in the gating
disabled = (
    not (planned_shares > 0 and np.isfinite(risk_per_share) and risk_per_share > 0 and rr_ok_planned)
    or blocked_active
)



with colR:
    if st.button("💾 Update queued plan", type="primary", use_container_width=True, disabled=disabled):
        ok = io_helpers.update_trade_candidate(
            editing_rowid,
            Entry=entry, Stop=stop, Take=take, Shares=planned_shares, RR=rr,
            TP1=TP1, TP2=TP2, TP3=TP3,
            Reasons=st.session_state["rr_reason"],
        )
        if ok:
            st.success("Updated the selected Trade Queue row.")
        else:
            st.error("Could not find the selected queue row to update.")




