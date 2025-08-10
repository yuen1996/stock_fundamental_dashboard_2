# --- make project root importable so we can import io_helpers/calculations/rules ---
import os, sys
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))         # .../stock_fundamental_dashboard
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))   # repo root (one level above)

for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# now these imports will work whether files are in the package root or repo root
try:
    import io_helpers, calculations, rules
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # fallback if you move them under utils/

# add config import
try:
    import config
except ModuleNotFoundError:
    from utils import config  # type: ignore


# 4_Systematic_Decision.py

# --- path patch: allow imports from both package root and repo root ---
import os, sys
PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))      # parent of /pages
REPO_ROOT    = os.path.dirname(PACKAGE_ROOT)                   # parent of package
for p in (PACKAGE_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
# ---------------------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np

# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import calculations, io_helpers, rules
except Exception:
    import calculations
    import io_helpers
    import rules

# ──────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Systematic Decision", layout="wide")

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

/* Section header card */
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

# Title & intro
st.header("🚦 Systematic Decision Engine")
st.caption(
    "Funnel-first evaluation: Cash-flow → TTM consistency → 5Y growth → Valuation @ entry → Dividend → Momentum. "
    "Use the filters to surface the strongest names."
)


# ──────────────────────────────────────────────────────────
# Load latest ANNUAL row per stock
# ──────────────────────────────────────────────────────────
df = io_helpers.load_data()

# Ensure the new column exists so filters don't crash
if "IndustryBucket" not in df.columns:
    df["IndustryBucket"] = "General"   # default bucket

if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please add stocks in **Add / Edit** first.")
    st.stop()

annual_only = df[df["IsQuarter"] != True].copy()
if annual_only.empty:
    st.info("No annual rows available.")
    st.stop()

latest = (
    annual_only
    .sort_values(["Name", "Year"])
    .groupby("Name", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)


# ──────────────────────────────────────────────────────────
# Funnel controls (FD / EPF + filters)
# ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">⚙️ Funnel Controls</div>'
    '<div class="d">Set FD/EPF baselines and filter with the funnel</div></div>',
    unsafe_allow_html=True
)

c_fd, c_epf = st.columns(2)
with c_fd:
    fd_rate = st.number_input(
        "FD Rate (baseline for dividend)",
        value=float(getattr(config, "FD_RATE", 0.035)),
        step=0.001, format="%.3f"
    )
with c_epf:
    epf_rate = st.number_input(
        "EPF (KWSP) Rate (good dividend)",
        value=float(getattr(config, "EPF_RATE", 0.058)),
        step=0.001, format="%.3f"
    )

# small helper for CAGR from 5Y annual
def _cagr(first: float|None, last: float|None, years: int) -> float|None:
    try:
        if first is None or last is None: return None
        if first <= 0 or last <= 0: return None
        return (last/first)**(1/years) - 1.0
    except Exception:
        return None


# ── Filters that follow your Industry & IndustryBucket settings ──────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Filters</div>'
    '<div class="d">Industry / Bucket plus funnel filters (Cash, Dividend, Annual, CAGR)</div></div>',
    unsafe_allow_html=True
)

# Clean lists
ind_list = sorted(
    [s for s in df.get("Industry", pd.Series(dtype="string"))
         .astype("string").str.strip().dropna().unique() if s and s.lower() != "nan"]
) or ["—"]
bucket_list = list(getattr(config, "INDUSTRY_BUCKETS", ["General"]))

# Row 1: industry/bucket
r1c1, r1c2 = st.columns([2, 2])
with r1c1:
    pick_ind = st.multiselect("Industry (free text)", ind_list, default=ind_list)
with r1c2:
    pick_bucket = st.multiselect("Industry Bucket", bucket_list, default=bucket_list)

# Row 2: funnel filters
r2c1, r2c2, r2c3, r2c4 = st.columns([1.4, 1.6, 2.2, 1.2])
with r2c1:
    cash_strong = st.checkbox("Cash strong only", value=False,
                              help="Require Cash-flow block ≥ threshold and pass cash gate")
with r2c2:
    cash_thr = st.slider("Cash block ≥", 0, 100, 60)
with r2c3:
    div_filter = st.selectbox("Dividend filter",
        ["Any", f"≥ FD ({fd_rate:.1%})", f"≥ EPF ({epf_rate:.1%})"],
        help="Filter stocks by yield class at entry price")
with r2c4:
    annual_improving = st.checkbox("Annual improving", value=False,
        help="Latest FY ROE or Net Margin improved vs prior FY")

# Row 3: CAGR filters
r3c1, r3c2 = st.columns([2,2])
with r3c1:
    rev_cagr_min = st.slider("Min Revenue CAGR (5y)", 0, 30, 0, format="%d%%")
with r3c2:
    eps_cagr_min = st.slider("Min EPS CAGR (5y)", 0, 30, 0, format="%d%%")


# ---- Apply Industry + Bucket filters here (AFTER the UI above defines the vars) ----
l = latest.copy()
l["Industry"] = l.get("Industry", pd.Series(dtype="string")).astype("string").str.strip()
if "IndustryBucket" not in l.columns:
    l["IndustryBucket"] = "General"

if pick_ind:
    l = l[l["Industry"].isin(pick_ind)]
if pick_bucket:
    l = l[l["IndustryBucket"].isin(pick_bucket)]

latest_filt = l.reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# Evaluate (filtered set, with Min Score / PASS-only gates)
# ──────────────────────────────────────────────────────────
rows = []
for _, row in latest_filt.iterrows():
    cur = row.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = row.get("SharePrice", np.nan)

    name = row["Name"]
    stock_rows = df[df["Name"] == name]
    bucket = row.get("IndustryBucket", "") or "General"

    # --- funnel scorer (industry-aware) ---
    res = calculations.compute_industry_scores(
        stock_name=name,
        stock_df=stock_rows,
        bucket=bucket,
        entry_price=cur,
        fd_rate=float(fd_rate),
    )

    gates = res.get("gates", {})
    blocks = res.get("blocks", {})

    # --- derive helpers for filters & display ---
    b_cash = blocks.get("cashflow_first", {}).get("score", np.nan)
    b_ttm  = blocks.get("ttm_vs_lfy", {}).get("score", np.nan)
    b_gq   = blocks.get("growth_quality", {}).get("score", np.nan)
    b_div  = blocks.get("dividend", {}).get("score", np.nan)
    b_mom  = blocks.get("momentum", {}).get("score", np.nan)
    val_lbl= blocks.get("valuation_entry", {}).get("label", "")

    # Dividend yield class at entry price (simple): last FY Dividend / entry
    try:
        ann = stock_rows[stock_rows["IsQuarter"] != True].sort_values("Year")
        div_ps = float(pd.to_numeric(ann.get("Dividend"), errors="coerce").dropna().iloc[-1]) if not ann.empty else np.nan
    except Exception:
        div_ps = np.nan
    yld = (div_ps / cur) if (pd.notna(div_ps) and pd.notna(cur) and cur > 0) else np.nan

    if pd.notna(yld):
        if yld >= epf_rate:
            div_class = "Good (≥EPF)"
        elif yld >= fd_rate:
            div_class = "OK (≥FD)"
        else:
            div_class = "Bad (<FD)"
    else:
        div_class = "—"

    # Annual improving: latest FY ROE or NPM up vs prior FY
    def _last2_delta(series: pd.Series) -> float|None:
        s = pd.to_numeric(series, errors="coerce").dropna().tail(2)
        if len(s) < 2: return None
        return float(s.iloc[-1] - s.iloc[-2])

    roe_delta = _last2_delta(ann.get("ROE (%)", pd.Series(dtype=float))) if 'ann' in locals() else None
    npm_last = None
    if 'ann' in locals() and not ann.empty:
        np_last2 = pd.to_numeric(ann.get("NetProfit"), errors="coerce").dropna().tail(2)
        rv_last2 = pd.to_numeric(ann.get("Revenue"), errors="coerce").dropna().tail(2)
        if len(np_last2) == 2 and len(rv_last2) == 2 and all(rv_last2 > 0):
            npm_last = (np_last2.iloc[-1]/rv_last2.iloc[-1]*100) - (np_last2.iloc[-2]/rv_last2.iloc[-2]*100)
    annual_ok = ((roe_delta is not None and roe_delta > 0) or (npm_last is not None and npm_last > 0))

    # 5Y CAGR filters: compute from annual Revenue & EPS (EPS = NP/shares if missing)
    rev_cagr = eps_cagr = None
    try:
        a5 = ann.tail(5)
        if len(a5) >= 2:
            rev_first = pd.to_numeric(a5["Revenue"], errors="coerce").dropna().iloc[0] if "Revenue" in a5 else None
            rev_last  = pd.to_numeric(a5["Revenue"], errors="coerce").dropna().iloc[-1] if "Revenue" in a5 else None
            rev_cagr  = _cagr(rev_first, rev_last, max(1, len(a5)-1))

            np_ser = pd.to_numeric(a5.get("NetProfit"), errors="coerce")
            sh_ser = pd.to_numeric(a5.get("NumShares"), errors="coerce")
            eps_first = (np_ser.iloc[0]/sh_ser.iloc[0]) if (len(np_ser.dropna())>0 and len(sh_ser.dropna())>0 and sh_ser.iloc[0]>0) else None
            eps_last  = (np_ser.iloc[-1]/sh_ser.iloc[-1]) if (len(np_ser.dropna())>0 and len(sh_ser.dropna())>0 and sh_ser.iloc[-1]>0) else None
            eps_cagr  = _cagr(eps_first, eps_last, max(1, len(a5)-1))
    except Exception:
        pass

    # --- apply funnel filters ---
    if cash_strong:
        if not gates.get("cashflow_ok", False): 
            continue
        if pd.isna(b_cash) or b_cash < cash_thr:
            continue

    if div_filter.endswith("FD") and not pd.isna(yld) and yld < fd_rate:
        continue
    if "EPF" in div_filter and not pd.isna(yld) and yld < epf_rate:
        continue

    if annual_improving and not annual_ok:
        continue

    if rev_cagr_min > 0 and (rev_cagr is None or rev_cagr*100 < rev_cagr_min):
        continue
    if eps_cagr_min > 0 and (eps_cagr is None or eps_cagr*100 < eps_cagr_min):
        continue

    # --- row for display ---
    rows.append({
        "Name":         name,
        "Industry":     row.get("Industry", ""),
        "Bucket":       res["bucket"],
        "Year":         int(row["Year"]),
        "Price":        cur,
        "Cash✓":        "✅" if (gates.get("cashflow_ok") and not pd.isna(b_cash) and b_cash >= cash_thr) else "—",
        "CashScore":    round(b_cash, 1) if pd.notna(b_cash) else np.nan,
        "TTM✓":         "✅" if (not pd.isna(b_ttm) and b_ttm >= 50) else "—",
        "TTMScore":     round(b_ttm, 1) if pd.notna(b_ttm) else np.nan,
        "Growth✓":      "✅" if (not pd.isna(b_gq) and b_gq >= 60) else "—",
        "GrowthScore":  round(b_gq, 1) if pd.notna(b_gq) else np.nan,
        "Valuation":    val_lbl,
        "Dividend":     div_class,
        "DivScore":     round(b_div, 1) if pd.notna(b_div) else np.nan,
        "Momentum":     round(b_mom, 1) if pd.notna(b_mom) else np.nan,
        "Score":        res["composite"],
        "Decision":     res["decision"],
        "Unmet":        "; ".join(res.get("why", []) + res.get("gates",{}).get("notes", [])),
    })




dec_df = (
    pd.DataFrame(rows)
    .sort_values(["Decision", "Score", "Name"], ascending=[True, False, True])
    .reset_index(drop=True)
)

st.markdown(
    '<div class="sec info"><div class="t">🧮 Evaluation Result</div>'
    '<div class="d">Latest annual row per stock</div></div>',
    unsafe_allow_html=True
)
st.dataframe(dec_df, use_container_width=True, height=380)

# ──────────────────────────────────────────────────────────
# Per-stock details: gates, block scores, NPM trend, alerts
# ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">🔎 Detail</div>'
    '<div class="d">Per-stock blocks, Net Profit Margin trend & alerts</div></div>',
    unsafe_allow_html=True
)

if dec_df.empty:
    st.info("No rows to inspect.")
else:
    pick_name = st.selectbox("Pick a stock", dec_df["Name"].tolist())

    row_latest = latest_filt[latest_filt["Name"] == pick_name].iloc[0]
    cur = row_latest.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = row_latest.get("SharePrice", np.nan)
    stock_rows = df[df["Name"] == pick_name]
    bucket = row_latest.get("IndustryBucket", "") or "General"

    res = calculations.compute_industry_scores(
        stock_name=pick_name,
        stock_df=stock_rows,
        bucket=bucket,
        entry_price=cur,
        fd_rate=float(fd_rate),
    )

    # ---- Gates badges ----
    gates = res.get("gates", {})
    c1, c2 = st.columns(2)
    c1.metric("Data OK",       "✅" if gates.get("data_ok") else "❌")
    c2.metric("Cash-flow OK",  "✅" if gates.get("cashflow_ok") else "❌")
    if gates.get("notes"):
        st.caption("Notes: " + " · ".join(gates.get("notes", [])))

    # ---- Block scores table ----
    blk = res.get("blocks", {})
    rows_blk = []
    for key, label in [
        ("cashflow_first", "Cash-flow (5Y)"),
        ("ttm_vs_lfy",     "TTM vs LFY"),
        ("growth_quality", "Growth & Quality (5Y)"),
        ("valuation_entry","Valuation @ Entry"),
        ("dividend",       "Dividend"),
        ("momentum",       "Momentum"),
    ]:
        b = blk.get(key, {}) or {}
        rows_blk.append({
            "Block":   label,
            "Score":   round(b.get("score", float("nan")), 1),
            "Conf %":  round(100.0 * (b.get("conf", 1.0) or 0.0), 0),
            "Label":   b.get("label", ""),
        })
    st.dataframe(pd.DataFrame(rows_blk), use_container_width=True, height=230)

    # ---- Net Profit Margin trend (5Y) ----
    ann = stock_rows[stock_rows["IsQuarter"] != True].copy().sort_values("Year").tail(5)
    npm_tbl = pd.DataFrame({"Year": ann["Year"]})
    with np.errstate(divide="ignore", invalid="ignore"):
        npm = pd.to_numeric(ann.get("NetProfit"), errors="coerce") / pd.to_numeric(ann.get("Revenue"), errors="coerce")
    npm_tbl["NPM %"] = (npm * 100).round(1)
    npm_tbl["Drop≥3pp"] = npm_tbl["NPM %"].diff().le(-3.0).fillna(False)
    st.write("**Net Profit Margin — last 5 years** (drops ≥ 3pp highlighted)")
    st.dataframe(npm_tbl, use_container_width=True, height=180)

    # ---- Alerts (if any) ----
    alerts = res.get("alerts", [])
    if alerts:
        st.write("**Alerts**")
        for a in alerts:
            sev = str(a.get("severity", "info")).upper()
            msg = str(a.get("message", ""))
            st.write(f"- {sev}: {msg}")



# ──────────────────────────────────────────────────────────
# Actions for PASS candidates —  Score ≥ 75  (Push only)
# ──────────────────────────────────────────────────────────
pass_df = dec_df[(dec_df["Decision"] == "PASS") & (dec_df["Score"] >= 50)].copy()

st.markdown(
    '<div class="sec success"><div class="t">✅ PASS Stocks (Score ≥ 50)</div>'
    '<div class="d">Tick rows to push into Trade Queue</div></div>',
    unsafe_allow_html=True
)

if pass_df.empty:
    st.info("No PASS candidates meet the ≥ 50 score threshold.")
else:
    # Build editable table — only one checkbox column (Push)
    pass_df = pass_df[["Name", "Industry", "Year", "CurrentPrice", "Score", "Unmet"]].copy()
    pass_df.insert(0, "SelectPush", False)
    pass_df["Strategy"] = strategy      # so we know which ruleset was used

    edited_pass = st.data_editor(
        pass_df,
        use_container_width=True,
        height=320,
        hide_index=True,
        column_config={
            "SelectPush":  st.column_config.CheckboxColumn("Push"),
            "CurrentPrice": st.column_config.NumberColumn("Current Price", format="%.4f", disabled=True),
            "Score":        st.column_config.NumberColumn("Score", format="%.0f", disabled=True),
            "Unmet":        st.column_config.TextColumn("Reasons (auto from evaluation)", disabled=True),
            "Strategy":     st.column_config.TextColumn("Strategy", disabled=True),
        },
        key="pass_actions_editor",
    )

    # Push-to-Queue button
if st.button("📥 Push selected to Queue"):
    pushed = 0
    for _, r in edited_pass.iterrows():
        if bool(r.get("SelectPush", False)):
            io_helpers.push_trade_candidate(
                name=r["Name"],
                strategy=strategy,
                score=float(r["Score"]),
                current_price=float(r["CurrentPrice"]) if pd.notna(r["CurrentPrice"]) else None,
                reasons=str(r.get("Unmet") or ""),
            )
            pushed += 1

    if pushed > 0:
        st.success(f"Pushed {pushed} stock(s) to Trade Queue.")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
    else:
        st.info("No rows were ticked. Tick **Push** in the first column, then try again.")



# ──────────────────────────────────────────────────────────
# Current Trade Queue + Manage (row-exact)
# ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">📋 Current Trade Queue</div>'
    '<div class="d">Plans waiting for execution</div></div>',
    unsafe_allow_html=True
)
tq = io_helpers.load_trade_queue().copy()

st.markdown(
    '<div class="sec warning"><div class="t">🔧 Manage Queue</div>'
    '<div class="d">Mark Live / Delete — acts on exact RowId</div></div>',
    unsafe_allow_html=True
)

# BEGIN PATCH  ─────────────────────────────────────────────
if tq.empty:
    st.info("Queue is empty.")
else:
    # ---------- explicit RowId & numeric coercion ----------
    tq = tq.reset_index().rename(columns={"index": "RowId"})
    for c in ["Entry", "Stop", "Take", "Shares", "RR"]:
        tq[c] = pd.to_numeric(tq.get(c), errors="coerce")

    # ---------- plan completeness ----------
    def _plan_ok_row(r):
        e, s, t, sh, rr = r.get("Entry"), r.get("Stop"), r.get("Take"), r.get("Shares"), r.get("RR")
        return (
            np.isfinite(e) and e > 0 and
            np.isfinite(s) and s > 0 and e > s and              # stop below entry (long)
            np.isfinite(t) and t > 0 and                        # take present
            np.isfinite(rr) and                                 # rr present
            pd.notna(sh) and (int(sh) if not pd.isna(sh) else 0) > 0  # shares > 0
        )
    tq["PlanOK"] = tq.apply(_plan_ok_row, axis=1)

    # default delete-reason helper
    def _default_reason(rr):
        try:
            return "R:R below threshold" if float(rr) < 1.5 else "Duplicate idea"
        except Exception:
            return "Duplicate idea"

    DELETE_REASONS = [
        "Duplicate idea", "Fails rules on recheck", "R:R below threshold",
        "Market conditions changed", "Wrong symbol / data error",
        "Moved to Watchlist", "Other (specify)",
    ]

    # display table (read-only planning fields + PlanOK)
    table = tq[[
        "RowId", "Name", "Strategy", "Entry", "Stop", "Take", "Shares", "RR",
        "Timestamp", "Reasons", "PlanOK"
    ]].copy()
    table.insert(0, "Select", False)

    edited_q = st.data_editor(
        table,
        use_container_width=True,
        height=360,
        hide_index=True,
        column_config={
            "Select":    st.column_config.CheckboxColumn("Sel"),
            "RowId":     st.column_config.NumberColumn("RowId", disabled=True),
            "Entry":     st.column_config.NumberColumn("Entry", format="%.4f", disabled=True),
            "Stop":      st.column_config.NumberColumn("Stop",  format="%.4f", disabled=True),
            "Take":      st.column_config.NumberColumn("Take",  format="%.4f", disabled=True),
            "Shares":    st.column_config.NumberColumn("Shares", format="%d", disabled=True),
            "RR":        st.column_config.NumberColumn("RR",    format="%.2f", disabled=True),
            "Timestamp": st.column_config.TextColumn("Added", disabled=True),
            "Reasons":   st.column_config.TextColumn("Notes/Reasons", disabled=True),
            "PlanOK":    st.column_config.CheckboxColumn("Plan Ready?", disabled=True),
        },
        key="queue_manage_editor",
    )

    c1, c2, _ = st.columns([1.6, 1.6, 3])

    # ---------- Mark Live (block if plan incomplete) ----------
    with c1:
        if st.button("✅ Mark Live selected"):
            moved, blocked = 0, 0
            blocked_ids = []
            for _, r in edited_q.iterrows():
                if not r.Select:
                    continue
                if not bool(r.get("PlanOK", False)):
                    blocked += 1
                    blocked_ids.append(int(r.RowId))
                    continue
                if io_helpers.mark_live_row(int(r.RowId)):
                    moved += 1

            if moved > 0:
                st.success(f"Marked live: {moved} row(s).")
                (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
            elif blocked > 0:
                st.warning(
                    f"{blocked} row(s) blocked — plan incomplete (need Entry, Stop, Take, Shares, RR): "
                    + ", ".join(map(str, blocked_ids))
                    + ". Open **Risk / Reward Planner** to finish the plan."
                )
            else:
                st.info("Nothing selected.")

    # ---------- Delete (row-exact) ----------
    with c2:
        if st.button("🗑️ Delete selected"):
            deleted, invalid = 0, 0
            for _, r in edited_q.iterrows():
                if not r.Select:
                    continue
                reason = "Duplicate idea"  # default; keep minimal
                if io_helpers.delete_trade_row(int(r.RowId), reason):
                    deleted += 1
            msg = f"Deleted {deleted} row(s)."
            st.success(msg)
            (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
# END PATCH  ─────────────────────────────────────────────

