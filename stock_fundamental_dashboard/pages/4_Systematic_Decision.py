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
    "PASS only when all mandatory checks succeed and score ≥ "
    f"{rules.MIN_SCORE}% (per selected ruleset)."
)

# ──────────────────────────────────────────────────────────
# Load latest ANNUAL row per stock
# ──────────────────────────────────────────────────────────
df = io_helpers.load_data()
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
# Strategy choice
# ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec"><div class="t">📋 Ruleset</div>'
    '<div class="d">Choose the playbook for evaluation</div></div>',
    unsafe_allow_html=True
)
opts = list(rules.RULESETS) + ["Industry (New)"]
strategy = st.selectbox("Strategy / Playbook", opts, index=0)

# Optional global FD override (used by Industry scorer)
fd_rate = st.number_input("FD Rate (for dividend check)", value=float(getattr(config, "FD_RATE", 0.035)), step=0.001, format="%.3f")


# ──────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────
rows = []
for _, row in latest.iterrows():
    cur = row.get("CurrentPrice", np.nan)
    if pd.isna(cur):
        cur = row.get("SharePrice", np.nan)

    name = row["Name"]
    stock_rows = df[df["Name"] == name]

    if strategy == "Industry (New)":
        bucket = row.get("IndustryBucket", "") or "General"
        res = calculations.compute_industry_scores(
            stock_name=name,
            stock_df=stock_rows,
            bucket=bucket,
            entry_price=cur,
            fd_rate=float(fd_rate),
        )
        rows.append({
            "Name":         name,
            "Industry":     row.get("Industry", ""),
            "Bucket":       res["bucket"],
            "Year":         int(row["Year"]),
            "CurrentPrice": cur,
            "Score":        res["composite"],
            "Decision":     res["decision"],
            "Unmet":        "; ".join(res["why"] + res["gates"]["notes"]),
            "Valuation":    res["blocks"]["valuation_entry"]["label"],
        })
    else:
        # Legacy strategies
        if strategy == "VQGM":
            metrics = calculations.compute_ttm(stock_rows, current_price=cur) or {}
        else:
            metrics = calculations.calc_ratios(row)

        ev = rules.evaluate(metrics, strategy)
        rows.append({
            "Name":         name,
            "Industry":     row.get("Industry", ""),
            "Year":         int(row["Year"]),
            "CurrentPrice": cur,
            "Score":        ev["score"],
            "Decision":     "PASS" if ev["pass"] else "REJECT",
            "Unmet":        "; ".join(ev["reasons"]),
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

