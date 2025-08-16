from auth_gate import require_auth
require_auth()

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
    
# add config import (same robust style)
try:
    import config
except ModuleNotFoundError:
    from utils import config  # type: ignore    


# --- path patch so this page can import from project root ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of /pages)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import re

import json  # ← needed for download buttons

# pretty JSON downloader used in all detail panels
def _download_json_button(label: str, obj, filename: str, key: str | None = None) -> None:
    import json
    import numpy as np
    import pandas as pd
    try:
        payload = json.dumps(obj, default=str, indent=2)
    except Exception:
        # last-resort fallback: try converting DataFrames/Series to records
        try:
            if isinstance(obj, pd.DataFrame):
                obj = obj.replace({pd.NA: None, np.nan: None}).to_dict("records")
            elif isinstance(obj, pd.Series):
                obj = obj.replace({pd.NA: None, np.nan: None}).to_dict()
            payload = json.dumps(obj, default=str, indent=2)
        except Exception:
            payload = json.dumps({"error": "failed to serialize"}, indent=2)
    st.download_button(label, data=payload, file_name=filename, mime="application/json",
                       key=key or f"dl_{filename}")


def _records(df: pd.DataFrame) -> list[dict]:
    try:
        return (
            df.replace({pd.NA: None, np.nan: None})
              .to_dict(orient="records")
        )
    except Exception:
        try:
            return df.to_dict(orient="records")
        except Exception:
            return []
        
def _records(df: pd.DataFrame) -> list[dict]:
    import pandas as pd, numpy as np
    if df is None or not isinstance(df, pd.DataFrame):
        return []
    df2 = df.copy()

    # If columns are MultiIndex (e.g., ("Income Statement","Revenue")), flatten to strings
    if isinstance(df2.columns, pd.MultiIndex):
        def _lab(c):
            # reuse your nice_label convention
            try:
                return " • ".join([str(x) for x in c])
            except Exception:
                return str(c)
        df2.columns = [ _lab(c) for c in df2.columns.to_list() ]

    # If the index is MultiIndex and you forgot to reset_index upstream, flatten that too
    if isinstance(df2.index, pd.MultiIndex):
        df2.index = [ " • ".join([str(x) for x in tup]) for tup in df2.index ]

    # Turn pandas NA/NaN into JSON-null
    df2 = df2.replace({pd.NA: None, np.nan: None})
    return df2.to_dict(orient="records")


# --- robust imports: prefer package (utils), fall back to top-level ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers
    import calculations
    import rules
    

def _df_to_records(df):
    import pandas as pd
    if df is None:
        return []
    if not isinstance(df, pd.DataFrame):
        return []
    # keep strings readable; let numbers remain numbers
    return df.replace({pd.NA: None}).to_dict("records")



# ------- Drag-and-drop imports (prefer stylable fallback) -------
DRAG_LIB = None
try:
    from st_draggable_list import DraggableList  # pip install st-draggable-list
    DRAG_LIB = "draggable-list"
except Exception:
    try:
        from streamlit_sortables import sort_items  # pip install streamlit-sortables
        DRAG_LIB = "sortables"
    except Exception:
        DRAG_LIB = None

# ---------- Force Wide Layout on Page ----------
st.set_page_config(layout="wide")

# ---------- Unified Page CSS (same as Dashboard) ----------
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
div[data-testid="stDataFrame"] table{ border-collapse:separate !important; border-spacing:0; }
div[data-testid="stDataFrame"] table tbody tr:hover td{ background:#f8fafc !important; }
div[data-testid="stDataFrame"] td{ border-bottom:1px solid var(--border) !important; }

/* → enable horizontal scrolling on ANY dataframe */
div[data-testid="stDataFrame"] {
  overflow-x: auto;
}

/* → collapse the grid into single borders, like Raw Data */
div[data-testid="stDataFrame"] table {
  border-collapse: collapse !important;
  width: max-content;      /* allow it to be wider than the container */
  min-width: 100%;         /* but at least fill the width */
}

/* → draw 1px solid borders on every cell */
div[data-testid="stDataFrame"] table th,
div[data-testid="stDataFrame"] table td {
  border: 1px solid var(--border) !important;
}

/* → keep the header row pinned when you scroll */
div[data-testid="stDataFrame"] table thead th {
  position: sticky !important;
  top: 0 !important;
  background: #f9fafb !important;
  z-index: 10 !important;
}


/* Inputs */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }
.stSlider > div [data-baseweb="slider"]{ margin-top:.25rem; }

/* Buttons */
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

# Extra CSS for the drag chips (kept minimal & matching Dashboard scale)
DRAG_CSS = """
<style>
.sort-help { color:#475569; font-size:.90rem; margin:.15rem 0 .25rem 0; }
.sort-box {
  border:1px dashed #cbd5e1; background:#f8fafc; border-radius:12px;
  padding:.15rem .30rem; max-height:90px; overflow:auto;
}
.sort-box [draggable="true"]{
  display:inline-flex !important; width:auto !important; max-width:100% !important;
  border-radius:9999px !important; background:#eef2ff !important; border:1px solid #c7d2fe !important;
  color:#0f172a !important; font-size:10px !important; padding:1px 6px !important; line-height:1.15 !important; margin:2px !important;
}
.sort-box > div{ display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
.sdl-item, .sdl-item *{
  display:inline-flex !important; width:auto !important; max-width:100% !important;
  border-radius:9999px !important; background:#eef2ff !important; color:#0f172a !important; border:1px solid #c7d2fe !important;
  font-size:10px !important; padding:1px 6px !important; margin:2px !important;
}
.sdl-wrapper, .sdl-container, .sdl-list{ display:flex !important; flex-wrap:wrap !important; gap:4px !important; }
</style>
"""
st.markdown(DRAG_CSS, unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stDataFrame"] table thead th {
  position: sticky; top: 0; z-index: 2; background: #f9fafb;
}
div[data-testid="stDataFrame"] table th:first-child,
div[data-testid="stDataFrame"] table td:first-child {
  position: sticky; left: 0; z-index: 3; background: #ffffff;
}
</style>
""", unsafe_allow_html=True)



st.header("🔍 View Stock")

df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data.")
    st.stop()

# Ensure compatibility
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

# ---------- Filter bar ----------
st.markdown(
    '<div class="sec"><div class="t">🔎 Filter</div>'
    '<div class="d">Find stocks by name, industry, and industry bucket</div></div>',
    unsafe_allow_html=True
)

# Robust: handle missing Industry / IndustryBucket
industries = sorted([s for s in df.get("Industry", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()])
buckets    = sorted([s for s in df.get("IndustryBucket", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()])
industry_opts = ["All"] + industries
bucket_opts   = ["All"] + buckets

c1, c2, c3 = st.columns([2, 1.2, 1.2])
with c1:
    q_text = st.text_input("Find stock (name contains…)", value="", key="vs_query").strip()
with c2:
    pick_ind = st.selectbox("Industry", options=industry_opts, index=0, key="vs_industry")
with c3:
    pick_bucket = st.selectbox("Industry Bucket", options=bucket_opts, index=0, key="vs_bucket")

# Apply filters
df_f = df.copy()

# normalize to strings for safe contains()
name_col = df_f["Name"].astype(str)
if q_text:
    q = q_text.lower()
    df_f = df_f[name_col.str.lower().str.contains(q, na=False)]

if pick_ind != "All" and "Industry" in df_f.columns:
    df_f = df_f[df_f["Industry"].astype(str) == pick_ind]

if pick_bucket != "All" and "IndustryBucket" in df_f.columns:
    df_f = df_f[df_f["IndustryBucket"].astype(str) == pick_bucket]

# Final list used by the expanders below
stocks = sorted([s for s in df_f["Name"].dropna().astype(str).unique().tolist()])

st.caption(
    f"Showing **{len(stocks)}** stock(s)"
    f"{' in '+pick_ind if pick_ind!='All' else ''}"
    f"{' · bucket '+pick_bucket if pick_bucket!='All' else ''}"
    f"{' matching “'+q_text+'”' if q_text else ''}."
)


# ---------- Field definitions ----------
ANNUAL_SECTIONS = [
    ("Income Statement", [
        ("Net Profit",                        "NetProfit"),
        ("Gross Profit",                      "GrossProfit"),
        ("Revenue",                           "Revenue"),
        ("Cost Of Sales",                     "CostOfSales"),
        ("Finance Costs",                     "FinanceCosts"),
        ("Administrative Expenses",           "AdminExpenses"),
        ("Selling & Distribution Expenses",   "SellDistExpenses"),
    ]),
    ("Balance Sheet", [
        ("Number of Shares",                  "NumShares"),
        ("Current Asset",                     "CurrentAsset"),
        ("Other Receivables",                 "OtherReceivables"),
        ("Trade Receivables",                 "TradeReceivables"),
        ("Biological Assets",                 "BiologicalAssets"),
        ("Inventories",                       "Inventories"),
        ("Prepaid Expenses",                  "PrepaidExpenses"),
        ("Intangible Asset",                  "IntangibleAsset"),
        ("Current Liability",                 "CurrentLiability"),
        ("Total Asset",                       "TotalAsset"),
        ("Total Liability",                   "TotalLiability"),
        ("Shareholder Equity",                "ShareholderEquity"),
        ("Reserves",                          "Reserves"),
        # 🔁 rename only (keep same key, new label)
        ("Cash and bank balance",             "Cash"),
        # ➕ new granular liabilities/loans
        ("Current lease liabilities",         "LeaseLiabCurrent"),
        ("Non-current lease liabilities",     "LeaseLiabNonCurrent"),
        ("Borrowings",                        "Borrowings"),
        ("Other loans",                       "OtherLoans"),
    ]),
    ("Cash Flow Statement", [
        # 🔁 precise labels (same keys where possible)
        ("Net cash flow generated from/(used in) operating activities", "CFO"),
        ("Purchase of property, plant and equipment",                   "CapEx"),
        ("Income Tax Expense",                                          "IncomeTax"),
        # ➕ depreciation split
        ("Depreciation of property, plant and equipment",               "DepPPE"),
        ("Depreciation of investment property",                         "DepInvProp"),
        ("Depreciation of right-of-use assets",                         "DepROU"),
    ]),
]


QUARTERLY_SECTIONS = [
    ("Quarterly Income Statement", [
        ("Quarterly Net Profit", "Q_NetProfit"),
        ("Quarterly Gross Profit", "Q_GrossProfit"),
        ("Quarterly Revenue", "Q_Revenue"),
        ("Quarterly Cost Of Sales", "Q_CostOfSales"),
        ("Quarterly Finance Costs", "Q_FinanceCosts"),
        ("Quarterly Administrative Expenses", "Q_AdminExpenses"),
        ("Quarterly Selling & Distribution Expenses", "Q_SellDistExpenses"),
    ]),

    ("Quarterly Balance Sheet", [
        ("Number of Shares", "Q_NumShares"),
        ("Cash and bank balance", "Q_Cash"),
        ("Current Asset", "Q_CurrentAsset"),
        ("Other Receivables", "Q_OtherReceivables"),
        ("Trade Receivables", "Q_TradeReceivables"),
        ("Biological Assets", "Q_BiologicalAssets"),
        ("Inventories", "Q_Inventories"),
        ("Prepaid Expenses", "Q_PrepaidExpenses"),
        ("Intangible Asset", "Q_IntangibleAsset"),
        ("Current Liability", "Q_CurrentLiability"),
        ("Current lease liabilities", "Q_LeaseLiabCurrent"),
        ("Non-current lease liabilities", "Q_LeaseLiabNonCurrent"),
        ("Borrowings", "Q_Borrowings"),
        ("Other loans", "Q_OtherLoans"),
        ("Total Asset", "Q_TotalAsset"),
        ("Total Liability", "Q_TotalLiability"),
            ("Shareholder Equity", "Q_ShareholderEquity"),
        ("Reserves", "Q_Reserves"),
        ("Total Debt / Borrowings", "Q_TotalDebt"),
    ]),

    ("Quarterly Cash Flow Statement", [
        ("Net cash flow generated from/(used in) operating activities", "Q_CFO"),
        ("Purchase of property, plant and equipment",                   "Q_CapEx"),
        ("Income Tax Expense",                                          "Q_IncomeTax"),
        ("Depreciation of property, plant and equipment",               "Q_DepPPE"),
        ("Depreciation of investment property",                         "Q_DepInvProp"),
        ("Depreciation of right-of-use assets",                         "Q_DepROU"),
    ]),

    ("Quarterly Other Data", [
        ("Each end per every quarter price", "Q_EndQuarterPrice"),
    ]),
]


# ---------- Helpers ----------
def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    try:
        s = str(x).replace(",", "").strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _format_4(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.4f}"
    except Exception:
        return ""
    

def _reorder_empty_last(df_numeric: pd.DataFrame) -> pd.DataFrame:
    if df_numeric.empty:
        return df_numeric
    mask = df_numeric.isna().all(axis=1)
    return pd.concat([df_numeric[~mask], df_numeric[mask]], axis=0)

def _auto_height(df: pd.DataFrame, row_h: int = 28, base: int = 64, max_h: int = 1600) -> int:
    """Compute a good table height so it can show all rows without vertical scrolling."""
    n = max(1, len(df.index) + 1)  # + header
    return int(min(max_h, base + row_h * n))


def quarter_key_to_num(q):
    if pd.isna(q):
        return np.nan
    s = str(q).upper().strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 4 else np.nan
    except Exception:
        return np.nan

def nice_label(col):
    if isinstance(col, tuple) and len(col) == 2:
        return f"{col[0]} • {col[1]}"
    return str(col)

# (Kept for fallback visual only)
PILL_STYLE = {
    "list": {"backgroundColor": "transparent", "display": "flex", "flexWrap": "wrap", "gap": "8px", "padding": "4px"},
    "item": {"backgroundColor": "#eef2ff", "borderRadius": "9999px", "padding": "4px 10px",
             "border": "1px solid #c7d2fe", "color": "#1e293b", "fontSize": "13px"},
    "itemLabel": {"color": "#1e293b"},
}

def _drag_labels(labels, key_suffix):
    if DRAG_LIB == "sortables":
        from streamlit_sortables import sort_items
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            new_labels = sort_items(items=labels, key=f"sort_{key_suffix}")
            st.markdown('</div>', unsafe_allow_html=True)
        return new_labels if isinstance(new_labels, list) else labels

    if DRAG_LIB == "draggable-list":
        from st_draggable_list import DraggableList
        data = [{"id": str(i), "order": i, "name": lab} for i, lab in enumerate(labels)]
        with st.container():
            st.markdown('<div class="sort-box">', unsafe_allow_html=True)
            result = DraggableList(data, key=f"drag_{key_suffix}")
            st.markdown('</div>', unsafe_allow_html=True)
        if isinstance(result, list):
            if result and isinstance(result[0], dict) and "name" in result[0]:
                return [d["name"] for d in result]
            return [str(x) for x in result]
        return labels

    st.info("Install a drag component to enable reordering: `pip install streamlit-sortables` "
            "or `pip install streamlit-draggable-list`.")
    return labels

def drag_reorder(columns, key_suffix, help_text="Drag to set column order. Left-most = first."):
    labels = [nice_label(c) for c in columns]
    state_key = f"reorder_saved_{key_suffix}"
    saved = st.session_state.get(state_key)
    if isinstance(saved, list):
        base = [l for l in saved if l in labels] + [l for l in labels if l not in saved]
    else:
        base = labels

    show = st.checkbox("🔧 Reorder (drag) — compact", value=False, key=f"reorder_toggle_{key_suffix}")
    if show:
        st.markdown(f'<div class="sort-help">{help_text}</div>', unsafe_allow_html=True)
        current_order = _drag_labels(base, key_suffix)
        if isinstance(current_order, list) and current_order:
            st.session_state[state_key] = current_order
    else:
        current_order = st.session_state.get(state_key, base)

    label_to_col = {nice_label(c): c for c in columns}
    ordered_cols = [label_to_col[l] for l in current_order if l in label_to_col]
    for c in columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    return ordered_cols

THRESHOLDS = {
    "ROE (%)":                 ("ge", 15,   10,   "Higher is better"),
    "Debt-Asset Ratio (%)":    ("le", 40,   60,   "Lower is better"),
    "P/E":                     ("le", 15,   20,   "Lower is better"),
    "P/B":                     ("le", 2.0,  3.0,  "Lower is better"),
    "Gross Profit Margin (%)": ("ge", 30,   20,   "Higher is better"),
    "Net Profit Margin (%)":   ("ge", 15,   8,    "Higher is better"),
    "Current Ratio":           ("ge", 2.0,  1.5,  "Higher is better"),
    "Quick Ratio":             ("ge", 1.5,  1.0,  "Higher is better"),
    "Dividend Yield (%)":      ("ge", 5.0,  3.0,  "Higher is better"),
    "Dividend Payout Ratio (%)":("le", 60,  80,   "Lower is safer"),
    "Three Fees Ratio (%)":    ("le", 30,   40,   "Lower is better"),
    "Total Cost %":            ("le", 70,   80,   "Lower is better"),
    "EPS":                     ("ge", 0.0,  0.0,  "Positive is better"),
    "NTA per share":           ("ge", 0.0,  0.0,  "Positive is better"),
    "BVPS":                    ("ge", 0.0,  0.0,  "Positive is better"),
}

COLORS = {"good": "#16a34a22", "ok": "#3b82f622", "bad": "#ef444422"}

def _rate(value, rule):
    if rule is None or value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None, ""
    direction, great, ok, note = rule
    try:
        v = float(value)
    except Exception:
        return None, ""
    if direction == "ge":
        state = "good" if v >= great else ("ok" if v >= ok else "bad")
        expl = f"value {v:,.4f} → {'≥' if state!='bad' else '<'} thresholds; {note}"
    else:
        state = "good" if v <= great else ("ok" if v <= ok else "bad")
        expl = f"value {v:,.4f} → {'≤' if state!='bad' else '>'} thresholds; {note}"
    return state, expl

def style_ratio_table(df_in: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Color + tooltips for BOTH orientations:
      • metrics in columns (Year/Period rows)
      • metrics in index  (Years/Periods in columns)
    Build one colors matrix and one tips matrix, then set_tooltips() ONCE
    so every ratio shows a tooltip (not just the last processed column/row).
    """
    df = df_in.copy()
    df_num = df.apply(pd.to_numeric, errors="coerce")

    # matrices (same shape as df)
    colors = pd.DataFrame("", index=df.index, columns=df.columns)
    tips   = pd.DataFrame("", index=df.index, columns=df.columns)

    for r in df.index:
        row_rule = THRESHOLDS.get(r)  # if metric label is the row name
        for c in df.columns:
            col_rule = THRESHOLDS.get(c)  # if metric label is the column name
            rule = col_rule or row_rule
            if not rule:
                continue

            v = df_num.loc[r, c]
            state, _ = _rate(v, rule)
            if state:
                colors.loc[r, c] = f"background-color: {COLORS[state]};"

            # tooltip text
            direction, great, ok, note = rule
            bound = f"≥ {great}" if direction == "ge" else f"≤ {great}"
            mid   = f"≥ {ok}"    if direction == "ge" else f"≤ {ok}"
            label = c if col_rule else r
            try:
                vtxt = "–" if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{float(v):,.4f}"
            except Exception:
                vtxt = str(v)
            tips.loc[r, c] = f"{label}: {vtxt} → {state.upper() if state else '—'} (good {bound}, OK {mid}). {note}"

    sty = (
        df.style
          .format(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
          .apply(lambda _: colors, axis=None)   # apply the whole color matrix
          .set_tooltips(tips)                   # set ALL tooltips once (no overwrite)
    )
    return sty

def style_raw_spike_table(
    df_in: pd.DataFrame,
    *,
    is_columns_period: bool,       # True = periods are columns, False = periods are rows
    alert_pct: float = 1.0,        # 1.0 = 100% change
    materiality_ratio: float = 0.05,
    highlight_decreases: bool = True,      # NEW: flag large drops too
    scan_all_pairs: bool = True            # NEW: scan across the full history
) -> "pd.io.formats.style.Styler":
    """
    Highlight sudden changes between adjacent periods across the ENTIRE series.
    - Colors both cells in any pair whose change is >= alert_pct (in absolute terms)
      and whose absolute delta is at least `materiality_ratio * median(|series|)`.
    - Works for either orientation (periods as columns or as rows).
    """
    df = df_in.apply(pd.to_numeric, errors="coerce").copy()
    if df.empty:
        return df.style.format(lambda x: "" if pd.isna(x) else f"{x:,.4f}")

    colors = pd.DataFrame("", index=df.index, columns=df.columns)
    tips   = pd.DataFrame("", index=df.index, columns=df.columns)

    def _fmt(v):
        try: return f"{float(v):,.4f}"
        except Exception: return "–"

    if is_columns_period:
        # periods go left→right across columns
        cols = list(df.columns)
        for r in df.index:
            row_vals = df.loc[r, :].values.astype(float)
            med = np.nanmedian(np.abs(row_vals))
            floor = max(1e-12, materiality_ratio * (0 if np.isnan(med) else med))

            # scan all adjacent pairs (c[k-1] → c[k])
            for k in range(1, len(cols)):
                c_prev, c_curr = cols[k-1], cols[k]
                prev_v = df.loc[r, c_prev]
                curr_v = df.loc[r, c_curr]
                if pd.isna(prev_v) or pd.isna(curr_v):
                    continue

                base  = max(abs(float(prev_v)), floor)
                delta = float(curr_v) - float(prev_v)
                pct   = delta / base if base != 0 else np.nan

                if not np.isfinite(pct) or abs(delta) < floor:
                    continue

                # big move?
                if (abs(pct) >= alert_pct) and (highlight_decreases or pct >= 0):
                    colors.loc[r, c_prev] = "background-color: #f59e0b33; border: 1px solid #f59e0b;"
                    colors.loc[r, c_curr] = "background-color: #f59e0b33; border: 1px solid #f59e0b;"
                    label = "Jump" if pct >= 0 else "Drop"
                    tips.loc[r, c_curr] = f"⚠️ {label}: {pct*100:+.0f}% ({_fmt(prev_v)} → {_fmt(curr_v)})"
                    tips.loc[r, c_prev] = f"⚠️ Previous value before {label.lower()} ({_fmt(prev_v)})"
    else:
        # periods go top→bottom across rows
        rows = list(df.index)
        for c in df.columns:
            col_vals = df.loc[:, c].values.astype(float)
            med = np.nanmedian(np.abs(col_vals))
            floor = max(1e-12, materiality_ratio * (0 if np.isnan(med) else med))

            for k in range(1, len(rows)):
                r_prev, r_curr = rows[k-1], rows[k]
                prev_v = df.loc[r_prev, c]
                curr_v = df.loc[r_curr, c]
                if pd.isna(prev_v) or pd.isna(curr_v):
                    continue

                base  = max(abs(float(prev_v)), floor)
                delta = float(curr_v) - float(prev_v)
                pct   = delta / base if base != 0 else np.nan

                if not np.isfinite(pct) or abs(delta) < floor:
                    continue

                if (abs(pct) >= alert_pct) and (highlight_decreases or pct >= 0):
                    colors.loc[r_prev, c] = "background-color: #f59e0b33; border: 1px solid #f59e0b;"
                    colors.loc[r_curr,  c] = "background-color: #f59e0b33; border: 1px solid #f59e0b;"
                    label = "Jump" if pct >= 0 else "Drop"
                    tips.loc[r_curr, c] = f"⚠️ {label}: {pct*100:+.0f}% ({_fmt(prev_v)} → {_fmt(curr_v)})"
                    tips.loc[r_prev, c] = f"⚠️ Previous value before {label.lower()} ({_fmt(prev_v)})"

    return (
        df.style
          .format(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
          .apply(lambda _: colors, axis=None)
          .set_tooltips(tips)
    )




def _show_styled(styler, height=420):
    """
    Render a pandas Styler in-place so that:
      • it fills 100% of the Streamlit column width,
      • header row & first column stay pinned,
      • Pandas Styler tooltips (.pd-t) render ABOVE everything and stay near the cell,
      • works for both orientations (metrics in columns OR in index).
    """
    try:
        html_table = styler.to_html()  # includes Pandas' own <style> for .pd-t tooltips
    except Exception:
        st.dataframe(getattr(styler, "data", styler),
                     use_container_width=True,
                     height=height)
        return

    wrapper = f"""
<style>
  :root {{
    --border: #e5e7eb;
    --hover: #f8fafc;
    --shadow: 0 8px 24px rgba(15,23,42,.06);
  }}
  .wrap {{
    border:1px solid var(--border);
    border-radius:12px;
    box-shadow:var(--shadow);
    background:#fff;
    overflow:hidden;       /* clip only OUTSIDE the scroll area */
    width:100%;
  }}
  .scroll {{
    overflow-x:auto;
    overflow-y:auto;
    max-height:{int(height)}px;
    position: relative;    /* make a stacking context for tooltips */
  }}
   .wrap table {{
    border-collapse: separate;   /* was: collapse */
    border-spacing: 0;
    width: max-content;
    min-width: 100%;
  }}
   .wrap thead th {{
    position: sticky;            /* no top here */
    z-index: 20;
    background: #f9fafb;
    border-bottom: 1px solid var(--border);
  }}
  /* stack multiple header rows so they don't overlap */
  .wrap thead tr:nth-child(1) th {{ top: 0; }}
  .wrap thead tr:nth-child(2) th {{ top: 36px; }}
  .wrap thead tr:nth-child(3) th {{ top: 72px; }}

  .wrap th, .wrap td {{
    border:1px solid var(--border);
    padding:8px 10px;
    white-space:nowrap;
  }}
  .wrap tbody tr:hover td,
  .wrap tbody tr:hover th {{ background:var(--hover); }}
  .wrap tbody tr:nth-child(2n) td,
  .wrap tbody tr:nth-child(2n) th {{ background:#fcfcfd; }}
  .wrap tbody th {{
    position:sticky; left:0; z-index:18;          /* pinned first column */
    background:#fff; border-right:1px solid var(--border);
  }}
  .wrap thead th:first-child {{
    position:sticky; left:0; z-index:22;          /* pinned corner */
    background:#f9fafb; border-right:1px solid var(--border);
  }}
  .pd-styler {{ width: max-content; }}

  /* —— CRITICAL: make Pandas .pd-t tooltips sit above and anchor to the cell —— */
  .wrap td, .wrap th {{ position: relative; overflow: visible; }}
  .wrap .pd-t {{
    z-index: 9999 !important;               /* above sticky headers/columns */
    background: rgba(17,24,39,.92) !important;
    color: #fff !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
    pointer-events: none !important;        /* no flicker on hover */
    transform: translate(8px, -8px) !important;  /* nicer offset than Pandas' default */
    white-space: normal !important;         /* allow wrapping for long text */
    max-width: 320px !important;            /* prevent giant boxes */
  }}
</style>
<div class="wrap"><div class="scroll">{html_table}</div></div>
"""
    st.markdown(wrapper, unsafe_allow_html=True)


# ---------- Builders ----------
def build_annual_raw_numeric(annual_df: pd.DataFrame) -> pd.DataFrame:
    if annual_df.empty:
        return pd.DataFrame()
    years = sorted([int(y) for y in annual_df["Year"].dropna().unique()])
    rows = []
    for sec, items in ANNUAL_SECTIONS:
        for label, key in items:
            rows.append((sec, label, key))
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Section", "Field"])
    out = pd.DataFrame(index=idx, columns=[str(y) for y in years], dtype=float)
    ann_by_year = {int(r["Year"]): r for _, r in annual_df.iterrows()}
    for (sec, label, key), (i_sec, i_field) in zip(rows, out.index):
        for y in years:
            val = np.nan
            row = ann_by_year.get(y)
            if row is not None and key in row:
                val = _to_float(row[key])
            out.loc[(i_sec, i_field), str(y)] = val
    return out

def build_quarter_raw_numeric(quarter_df: pd.DataFrame) -> pd.DataFrame:
    if quarter_df.empty:
        return pd.DataFrame()
    q = quarter_df.copy()
    q["Qnum"] = q["Quarter"].map(quarter_key_to_num)
    q = q.dropna(subset=["Year", "Qnum"])
    q["Year"] = q["Year"].astype(int)
    q = q.sort_values(["Year", "Qnum"])
    periods = [f"{int(r['Year'])} Q{int(r['Qnum'])}" for _, r in q.iterrows()]
    seen, cols, row_by_period = set(), [], {}
    for period, (_, r) in zip(periods, q.iterrows()):
        if period in seen:
            continue
        seen.add(period)
        cols.append(period)
        row_by_period[period] = r
    rows = []
    for sec, items in QUARTERLY_SECTIONS:
        for label, key in items:
            rows.append((sec, label, key))
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Section", "Field"])
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for (sec, label, key), (i_sec, i_field) in zip(rows, out.index):
        for c in cols:
            row = row_by_period.get(c)
            val = np.nan
            if row is not None and key in row:
                val = _to_float(row[key])
            out.loc[(i_sec, i_field), c] = val
    return out

# ---------- Chart helpers ----------
def field_options(sections):
    opts = []
    for sec, items in sections:
        for lbl, _ in items:
            opts.append((f"{sec} • {lbl}", (sec, lbl)))
    return opts

def plot_single_series(x_values, y_values, title, yname, height=320):
    import re as _re
    def _safe(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9_]+", "_", str(s))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=yname))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), title=title, xaxis_title="", yaxis_title="")
    # 🔑 unique key based on title+yname
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"singleplot_{_safe(title)}_{_safe(yname)}",
    )


def multi_panel_charts(count, options, x_labels, series_getter, key_prefix, chart_height=320):
    import re as _re
    count = max(1, min(4, int(count)))
    option_labels = [o[0] for o in options]
    if not option_labels:
        st.info("No series available to chart.")
        return

    row1 = st.columns(2)
    row2 = st.columns(2) if count > 2 else (None, None)

    def _safe(s: str) -> str:
        return _re.sub(r"[^A-Za-z0-9_]+", "_", str(s))

    def render_cell(col_container, i):
        with col_container:
            default_idx = i if i < len(option_labels) else 0
            sel = st.selectbox(
                f"Chart {i+1} – pick a series",
                options=option_labels,
                index=default_idx,
                key=f"{key_prefix}_sel_{i}",
            )
            payload = dict(options)[sel]
            y = series_getter(payload)
            if y is None or (pd.isna(pd.Series(y)).all()):
                st.info("No data for this selection.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_labels, y=y, mode="lines+markers", name=sel))
                fig.update_layout(height=chart_height, margin=dict(l=10, r=10, t=40, b=10), title=sel, xaxis_title="", yaxis_title="")
                # 🔑 give this plot a unique key
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"{key_prefix}_plot_{i}_{_safe(sel)}",
                )

    render_cell(row1[0], 0)
    if count >= 2: render_cell(row1[1], 1)
    if count >= 3 and row2[0] is not None: render_cell(row2[0], 2)
    if count >= 4 and row2[1] is not None: render_cell(row2[1], 3)


# ---------- UI ----------
for stock_name in stocks:
    with st.expander(stock_name, expanded=False):
        stock = df[df["Name"] == stock_name].sort_values(["Year"])
        annual = stock[stock["IsQuarter"] != True].copy()
        quarterly = stock[stock["IsQuarter"] == True].copy()
        
        # Resolve Industry / IndustryBucket for this stock (safe defaults)
        industry = ""
        industry_bucket = ""
        if "Industry" in stock.columns and stock["Industry"].notna().any():
            try:
                industry = str(stock["Industry"].dropna().astype(str).iloc[-1])
            except Exception:
                industry = ""
        if "IndustryBucket" in stock.columns and stock["IndustryBucket"].notna().any():
            try:
                industry_bucket = str(stock["IndustryBucket"].dropna().astype(str).iloc[-1])
            except Exception:
                industry_bucket = ""


        # ---- Current Price metric (STRICT: CurrentPrice only)
        cur_val = 0.0
        if "CurrentPrice" in stock.columns:
            s = stock["CurrentPrice"].dropna()
            if not s.empty:
                cur_val = float(s.iloc[-1])
                st.metric("Current Price", f"{cur_val:,.4f}")


        # ---- Section header before the tabs
        st.markdown(
            '<div class="sec"><div class="t">📚 Financial Reports</div>'
            '<div class="d">Annual & Quarterly data, ratios & charts</div></div>',
            unsafe_allow_html=True
        )

        tabs = st.tabs(["Annual Report", "Quarterly Report"])

        # =========================
        # ANNUAL
        # =========================
        with tabs[0]:
            st.subheader(f"{stock_name} - Annual Financial Data")

            # ---- Annual — Raw Data
            st.markdown('<div class="sec info"><div class="t">🧾 Annual — Raw Data</div></div>', unsafe_allow_html=True)
            st.markdown("#### Raw Data")
            ann_numeric = build_annual_raw_numeric(annual)

            # ---- Append a TTM column (last 4 quarters) to the annual RAW table
            ttm_col_name = None  # ensure defined even if the try-block fails
            try:
                q = quarterly.copy()
                q["Year"] = pd.to_numeric(q["Year"], errors="coerce")
                # try to parse quarter number (Q1..Q4 or 1..4)
                qnum = q["Quarter"].astype(str).str.extract(r"(\d)", expand=False)
                q["Qnum"] = pd.to_numeric(qnum, errors="coerce")
                q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])
                q4 = q.tail(4).copy()

                if not q4.empty:
                    last_year = int(q4["Year"].iloc[-1])
                    ttm_col_name = f"TTM (to {last_year})"
                    if ttm_col_name not in ann_numeric.columns:
                        ann_numeric[ttm_col_name] = np.nan

                    # --- compatibility for older column names ---
                    if "Q_IncomeTax" not in q4.columns and "Q_Tax" in q4.columns:
                        q4["Q_IncomeTax"] = q4["Q_Tax"]
                    # fallback: if splits are missing but old Q_DepAmort exists, use it as DepPPE
                    if ("Q_DepPPE" not in q4.columns and
                        "Q_DepInvProp" not in q4.columns and
                        "Q_DepROU" not in q4.columns and
                        "Q_DepAmort" in q4.columns):
                        q4["Q_DepPPE"] = q4["Q_DepAmort"]

                    def _sum_q(col):
                        return float(pd.to_numeric(q4.get(col), errors="coerce").dropna().sum()) \
                            if col in q4.columns else np.nan

                    def _last_q(col):
                        s = pd.to_numeric(q4.get(col) if col in q4.columns else quarterly.get(col),
                                        errors="coerce").dropna()
                        return float(s.iloc[-1]) if not s.empty else np.nan

                    # map annual keys -> quarterly keys (sum over 4 qtrs or take latest)
                    SUM_MAP = {
                        # 4Q sums (P&L)
                        "NetProfit":        "Q_NetProfit",
                        "GrossProfit":      "Q_GrossProfit",
                        "Revenue":          "Q_Revenue",
                        "CostOfSales":      "Q_CostOfSales",
                        "FinanceCosts":     "Q_FinanceCosts",
                        "AdminExpenses":    "Q_AdminExpenses",
                        "SellDistExpenses": "Q_SellDistExpenses",

                        # 4Q sums (Cash Flow)
                        "CFO":              "Q_CFO",
                        "CapEx":            "Q_CapEx",
                        "IncomeTax":        "Q_IncomeTax",

                        # Dep splits (sum each)
                        "DepPPE":           "Q_DepPPE",
                        "DepInvProp":       "Q_DepInvProp",
                        "DepROU":           "Q_DepROU",
                    }
                    LAST_MAP = {
                        # latest snapshot balances
                        "NumShares":            "Q_NumShares",
                        "CurrentAsset":         "Q_CurrentAsset",
                        "OtherReceivables":     "Q_OtherReceivables",
                        "TradeReceivables":     "Q_TradeReceivables",
                        "BiologicalAssets":     "Q_BiologicalAssets",
                        "Inventories":          "Q_Inventories",
                        "PrepaidExpenses":      "Q_PrepaidExpenses",
                        "IntangibleAsset":      "Q_IntangibleAsset",
                        "CurrentLiability":     "Q_CurrentLiability",
                        "TotalAsset":           "Q_TotalAsset",
                        "TotalLiability":       "Q_TotalLiability",
                        "ShareholderEquity":    "Q_ShareholderEquity",
                        "Reserves":             "Q_Reserves",
                        "Cash":                 "Q_Cash",
                        "SharePrice":           "Q_EndQuarterPrice",

                        # optional debt snapshot (if captured quarterly)
                        "Borrowings":           "Q_Borrowings",
                        "OtherLoans":           "Q_OtherLoans",
                        "LeaseLiabCurrent":     "Q_LeaseLiabCurrent",
                        "LeaseLiabNonCurrent":  "Q_LeaseLiabNonCurrent",
                        "TotalDebt":            "Q_TotalDebt",
                    }

                    # fill the new column using your existing row index (Section, Field)
                    for sec, items in ANNUAL_SECTIONS:
                        for label, key in items:
                            val = np.nan
                            if key in SUM_MAP:
                                val = _sum_q(SUM_MAP[key])
                            elif key in LAST_MAP:
                                val = _last_q(LAST_MAP[key])
                            elif key == "CostOfSales":
                                # fallback via Revenue - GrossProfit
                                rv = _sum_q("Q_Revenue")
                                gp = _sum_q("Q_GrossProfit")
                                if pd.notna(rv) and pd.notna(gp):
                                    val = rv - gp
                            elif key == "TotalDebt":
                                # derive TotalDebt from parts if not keyed directly
                                parts = ["Q_Borrowings", "Q_OtherLoans", "Q_LeaseLiabCurrent", "Q_LeaseLiabNonCurrent"]
                                vals = [_last_q(p) for p in parts]
                                if any(pd.notna(v) for v in vals):
                                    val = float(np.nansum([v for v in vals if pd.notna(v)]))

                            # write into MultiIndex row (Section, Field)
                            if (sec, label) in ann_numeric.index:
                                ann_numeric.loc[(sec, label), ttm_col_name] = val

            except Exception:
                # swallow any issues building the TTM column so the page keeps rendering
                pass

            # --- Force the TTM price cell to use the live/current price
            try:
                if ttm_col_name and (ttm_col_name in ann_numeric.columns) and cur_val is not None and np.isfinite(cur_val):
                    # your price field is "Other Data" • "End of year share price" (key -> SharePrice)
                    if ("Other Data", "End of year share price") in ann_numeric.index:
                        ann_numeric.loc[("Other Data", "End of year share price"), ttm_col_name] = float(cur_val)
                    else:
                        # generic fallback: find the row whose key == "SharePrice"
                        for sec, items in ANNUAL_SECTIONS:
                            for label, key in items:
                                if key == "SharePrice" and (sec, label) in ann_numeric.index:
                                    ann_numeric.loc[(sec, label), ttm_col_name] = float(cur_val)
                                    raise StopIteration
            except StopIteration:
                pass
            except Exception:
                pass

            
            ann_raw_layout = st.radio(
                "Raw data layout (annual)",
                ["Fields → columns (Year rows)", "Years → columns (Field rows)"],
                horizontal=True,
                key=f"annual_raw_layout_{stock_name}"
            )

            if ann_numeric.empty:
                st.info("No annual raw data available.")
            else:
                if ann_raw_layout.startswith("Years"):
                    disp_num = _reorder_empty_last(ann_numeric)
                    new_cols = drag_reorder(
                        disp_num.columns.tolist(),
                        key_suffix=f"ann_raw_yearcols_{stock_name}",
                        help_text="Drag to reorder **Year** columns."
                    )
                    disp_num = disp_num[new_cols]
                    st.session_state[f"ann_raw_x_{stock_name}"] = [str(x) for x in new_cols]
                else:
                    disp_num = _reorder_empty_last(ann_numeric.T)
                    field_cols = disp_num.columns.tolist()
                    new_cols = drag_reorder(
                        field_cols,
                        key_suffix=f"ann_raw_fieldcols_{stock_name}",
                        help_text="Drag to reorder **Field** columns."
                    )
                    disp_num = disp_num[new_cols]
                    st.session_state[f"ann_raw_x_{stock_name}"] = [str(x) for x in disp_num.index.tolist()]

            # ---- Annual Raw table (styled only)
            if ann_raw_layout.startswith("Years"):
                styled = style_raw_spike_table(disp_num, is_columns_period=True)   # periods = columns
            else:
                styled = style_raw_spike_table(disp_num, is_columns_period=False)  # periods = rows
            _show_styled(styled, height=_auto_height(disp_num, row_h=34, base=96, max_h=1400))
            st.caption("⚠️ Alert: Highlighted cells show ≥100% YoY jump vs the previous year, and a material change (≥5% of median level).")

            # 🔄 REPLACEMENT: "Show calculation of TTM (to YYYY)" + keep Annual Raw download
            with st.expander("Show calculation details — TTM", expanded=False):
                # --- visible matrix order for JSON (same as before)
                try:
                    vis = disp_num.reset_index() if isinstance(disp_num, pd.DataFrame) else pd.DataFrame()
                except Exception:
                    vis = pd.DataFrame()

                matrix_df = ann_numeric.copy()
                try:
                    matrix_df = matrix_df[new_cols]          # keep the user's visible column order
                except Exception:
                    pass

                # --- last quarters table (inputs shown like your sample)
                q = stock[stock.get("IsQuarter", False) == True].copy()
                q["Year"] = pd.to_numeric(q["Year"], errors="coerce")
                q["Qnum"] = pd.to_numeric(q["Quarter"].astype(str).str.extract(r"(\d)", expand=False), errors="coerce")
                q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])

                # pick columns (robust to alt names)
                cols_cand = {
                    "Revenue":         ["Quarterly Revenue","Q_Revenue","Revenue","Sales","Q_Sales","Q_TotalRevenue"],
                    "GrossProfit":     ["Q_GrossProfit","GrossProfit","Quarterly Gross Profit"],
                    "OperatingProfit": ["Q_OperatingProfit","Q_EBIT","OperatingProfit","EBIT","Quarterly Operating Profit"],
                    "NetProfit":       ["Quarterly Net Profit","Q_NetProfit","Q_Profit","Q_NetIncome","NetProfit","NetIncome"],
                    "EPS":             ["Quarterly EPS","Q_EPS","EPS","Basic EPS","Diluted EPS","EPS (Basic)","EPS (Diluted)"],
                    "SharesQ":         ["Q_NumShares","Number of Shares","SharesOutstanding","ShareOutstanding","ShareCount","BasicShares","NumShares","Number of shares"],
                    "PriceQ":          ["Q_EndQuarterPrice","Q_SharePrice","Each end per every quarter price","Price","CurrentPrice"],
                    "EquityQ":         ["Q_ShareholderEquity","ShareholderEquity","Shareholder Equity","TotalEquity","Total Equity","Equity"],
                    "DebtQ":           ["Q_TotalDebt","Q_Borrowings","Q_OtherLoans","TotalDebt","Debt","Borrowings"],
                    "CashQ":           ["Q_Cash","Cash","CashAndEquivalents","Cash & Cash Equivalents","Cash & Equivalents"],
                    "EBITDAQ":         ["Q_EBITDA","EBITDA","Quarterly EBITDA"],
                    "CFOQ":            ["Q_CFO"],
                    "CapExQ":          ["Q_CapEx"],
                }
                def pick(df, cands): return next((c for c in cands if c in df.columns), None)

                rev_c  = pick(q, cols_cand["Revenue"])
                gp_c   = pick(q, cols_cand["GrossProfit"])
                op_c   = pick(q, cols_cand["OperatingProfit"])
                np_c   = pick(q, cols_cand["NetProfit"])
                eps_c  = pick(q, cols_cand["EPS"])
                shq_c  = pick(q, cols_cand["SharesQ"])
                prq_c  = pick(q, cols_cand["PriceQ"])
                eqq_c  = pick(q, cols_cand["EquityQ"])
                debtq_c= pick(q, cols_cand["DebtQ"])
                cashq_c= pick(q, cols_cand["CashQ"])
                ebd_c  = pick(q, cols_cand["EBITDAQ"])
                cfo_c  = pick(q, cols_cand["CFOQ"])
                capex_c= pick(q, cols_cand["CapExQ"])

                show_cols = ["Year","Quarter"] + [c for c in [rev_c,gp_c,op_c,np_c,eps_c,shq_c,prq_c,eqq_c] if c]
                if show_cols:
                    st.caption("Recent quarters (most recent at bottom). TTM sums take the latest 4 rows.")
                    st.dataframe(q[show_cols].tail(8), use_container_width=True, height=260)

                # --- helpers
                import numpy as _np, pandas as _pd
                def _to_num(s): return _pd.to_numeric(s, errors="coerce")
                def _last_num(s):
                    try:
                        s = _to_num(s).dropna()
                        return float(s.iloc[-1]) if not s.empty else _np.nan
                    except Exception:
                        return _np.nan

                # ===== TTM sums from last 4 quarters =====
                q4 = q.tail(4)
                last_year = int(q4["Year"].iloc[-1]) if not q4.empty else None

                rev4   = _to_num(q.get(rev_c)).tail(4).sum() if rev_c else _np.nan
                gp4    = _to_num(q.get(gp_c)).tail(4).sum()  if gp_c  else _np.nan
                op4    = _to_num(q.get(op_c)).tail(4).sum()  if op_c  else _np.nan
                np4    = _to_num(q.get(np_c)).tail(4).sum()  if np_c  else _np.nan
                eps_sum= _to_num(q.get(eps_c)).tail(4).sum() if eps_c else _np.nan
                sh4    = _to_num(q.get(shq_c)).tail(4) if shq_c else _pd.Series(dtype="float64")
                sh_avg = float(sh4.dropna().mean()) if not sh4.dropna().empty else _np.nan
                eps_ttm= eps_sum if (_np.isfinite(eps_sum) and float(eps_sum) != 0.0) else \
                        (float(np4)/float(sh_avg) if (_np.isfinite(np4) and _np.isfinite(sh_avg) and sh_avg>0) else _np.nan)

                cfo4   = _to_num(q.get(cfo_c)).tail(4).sum()   if cfo_c   else _np.nan
                capex4 = _to_num(q.get(capex_c)).tail(4).sum() if capex_c else _np.nan
                ebitda4= _to_num(q.get(ebd_c)).tail(4).sum()   if ebd_c   else _np.nan

                # ===== Pull LATEST ANNUAL values for balance-sheet =====
                def latest_annual(cands):
                    for col in cands:
                        if col in annual.columns:
                            s = _to_num(annual[col]).dropna()
                            if not s.empty:
                                return float(s.iloc[-1])
                    return _np.nan

                shares_latest = latest_annual(["NumShares","Number of Shares","SharesOutstanding","ShareOutstanding","ShareCount","BasicShares","Number of shares"])
                equity_latest = latest_annual(["ShareholderEquity","Shareholder Equity","TotalEquity","Total Equity","Equity"])
                cur_asset     = latest_annual(["CurrentAsset"])
                cur_liab      = latest_annual(["CurrentLiability"])
                debt_latest   = latest_annual(["TotalDebt","Debt","Borrowings"])
                cash_latest   = latest_annual(["Cash","CashAndEquivalents","Cash & Equivalents"])

                # ===== Use the CURRENT screen price for valuation =====
                price_now  = float(cur_val) if (cur_val is not None) else _last_num(q.get(prq_c))
                market_cap = (price_now * shares_latest) if (_np.isfinite(price_now) and _np.isfinite(shares_latest)) else _np.nan
                net_debt   = ((debt_latest if _np.isfinite(debt_latest) else 0.0) -
                            (cash_latest if _np.isfinite(cash_latest) else 0.0))

                # ===== Ratios =====
                def _pct(n, d): 
                    try: return (float(n)/float(d))*100.0 if (_np.isfinite(n) and _np.isfinite(d) and d!=0) else _np.nan
                    except Exception: return _np.nan

                gross_m   = _pct(gp4, rev4)
                oper_m    = _pct(op4, rev4)
                net_m     = _pct(np4, rev4)
                pe_ttm    = (price_now / eps_ttm)          if (_np.isfinite(price_now) and _np.isfinite(eps_ttm) and eps_ttm!=0) else _np.nan
                ps_ttm    = (market_cap / rev4)            if (_np.isfinite(market_cap) and _np.isfinite(rev4) and rev4!=0)      else _np.nan
                pb_ttm    = (market_cap / equity_latest)   if (_np.isfinite(market_cap) and _np.isfinite(equity_latest) and equity_latest!=0) else _np.nan
                ev_ebitda = ((market_cap + net_debt)/ebitda4) if (_np.isfinite(market_cap) and _np.isfinite(ebitda4) and ebitda4!=0) else _np.nan

                # ===== Display like your sample =====
                def _fmt0(x):  return "–" if not _np.isfinite(x) else f"{x:,.0f}"
                def _fmt2(x):  return "–" if not _np.isfinite(x) else f"{x:,.2f}"
                def _fmt4(x):  return "–" if not _np.isfinite(x) else f"{x:,.4f}"

                if last_year: st.markdown(f"**TTM window:** latest four quarters → *to {last_year}*")

                st.markdown(
                    f"- **TTM Revenue** = sum(last 4 {rev_c or 'Q_Revenue'}) = **{_fmt0(rev4)}**  \n"
                    f"- **TTM Gross Margin** = GrossProfit_TTM ÷ Revenue_TTM = {_fmt0(gp4)} ÷ {_fmt0(rev4)} = **{_fmt2(gross_m)}%**  \n"
                    f"- **TTM Operating Margin** = OperatingProfit_TTM ÷ Revenue_TTM = {_fmt0(op4)} ÷ {_fmt0(rev4)} = **{_fmt2(oper_m)}%**  \n"
                    f"- **TTM Net Profit** = sum(last 4 {np_c or 'Q_NetProfit'}) = **{_fmt0(np4)}**  \n"
                    f"- **TTM Net Margin** = NetProfit_TTM ÷ Revenue_TTM = {_fmt0(np4)} ÷ {_fmt0(rev4)} = **{_fmt2(net_m)}%**  \n"
                    f"- **TTM EPS** = sum(last 4 {eps_c or 'EPS'})"
                    f"{'' if (_np.isfinite(eps_sum) and float(eps_sum)!=0.0) else ' *(fallback: Net Profit ÷ avg Shares)*'}"
                    f" = **{_fmt4(eps_ttm)}**  \n"
                    f"- **P/E (TTM)** = Price ÷ EPS_TTM = {_fmt4(price_now)} ÷ {_fmt4(eps_ttm)} = **{_fmt2(pe_ttm)}**  \n"
                    f"- **P/S (TTM)** = MarketCap ÷ Revenue_TTM = {_fmt0(market_cap)} ÷ {_fmt0(rev4)} = **{_fmt2(ps_ttm)}**  \n"
                    f"- **P/B (TTM)** = MarketCap ÷ Equity(latest ANNUAL) = {_fmt0(market_cap)} ÷ {_fmt0(equity_latest)} = **{_fmt2(pb_ttm)}**  \n"
                    f"- **EV/EBITDA (TTM)** = (MarketCap + NetDebt) ÷ EBITDA_TTM = ({_fmt0(market_cap)} + {_fmt0(net_debt)}) ÷ {_fmt0(ebitda4)} = **{_fmt2(ev_ebitda)}**"
                )

                # ===== Compact detail for JSON
                ttm_simple = {
                    "label": f"TTM (to {last_year})" if last_year else "TTM",
                    "quarters_used": ([(int(y), str(qt)) for y, qt in zip(q4["Year"], q4["Quarter"])] if not q4.empty else []),
                    "components": {
                        "TTM Revenue": (float(rev4) if _np.isfinite(rev4) else None),
                        "TTM Net Profit": (float(np4) if _np.isfinite(np4) else None),
                        "TTM Gross Profit": (float(gp4) if _np.isfinite(gp4) else None),
                        "TTM Cost Of Sales": (float(rev4-gp4) if (_np.isfinite(rev4) and _np.isfinite(gp4)) else None),
                        "CFO (TTM)": (float(cfo4) if _np.isfinite(cfo4) else None),
                        "CapEx (TTM)": (float(capex4) if _np.isfinite(capex4) else None),
                        "EBITDA (TTM)": (float(ebitda4) if _np.isfinite(ebitda4) else None),
                        "Shares (latest)": (float(shares_latest) if _np.isfinite(shares_latest) else None),     # ← ANNUAL
                        "Current Asset (latest)": (float(cur_asset) if _np.isfinite(cur_asset) else None),      # ← ANNUAL
                        "Current Liability (latest)": (float(cur_liab) if _np.isfinite(cur_liab) else None),    # ← ANNUAL
                        "Equity (latest)": (float(equity_latest) if _np.isfinite(equity_latest) else None),     # ← ANNUAL
                        "Debt (latest)": (float(debt_latest) if _np.isfinite(debt_latest) else None),           # ← ANNUAL
                        "Cash (latest)": (float(cash_latest) if _np.isfinite(cash_latest) else None),           # ← ANNUAL
                        "Price (latest)": (float(price_now) if _np.isfinite(price_now) else None),              # ← CURRENT
                    },
                }

                # --- human-readable annual JSON: [{year: 2021, Revenue: …}, …]
                years_in_matrix = [c for c in matrix_df.columns if isinstance(c, (str,int,float)) and not str(c).startswith("TTM")]
                annual_simple = []
                for y in years_in_matrix:
                    try:
                        y_int = int(float(str(y)))
                    except Exception:
                        continue
                    rec = {"year": y_int}
                    for sec, items in ANNUAL_SECTIONS:
                        for label, _key in items:
                            try:
                                v = matrix_df.loc[(sec, label), str(y)]
                                if pd.notna(v):
                                    rec[label] = float(v)
                            except Exception:
                                pass
                    annual_simple.append(rec)

                ann_payload = {
                    "stock": stock_name,
                    "industry": industry,
                    "industry_bucket": industry_bucket,
                    "annual": annual_simple,
                    "ttm": ttm_simple,
                }

                _download_json_button(
                    "📥 Download Annual Raw (JSON)",
                    ann_payload,
                    f"{stock_name}_annual_raw.json",
                    key=f"annual_raw_{stock_name}"
                )


                    
            # ---- TTM (last 4 quarters) — shown ABOVE CAGR
            ttm = calculations.compute_ttm(stock, current_price=cur_val)

            # === TTM deep debug (temporary) ===
            # Turn this on in the UI to see what columns are found, last 4 values, and the computed TTM pieces.
            if st.checkbox("TTM deep debug", key=f"ttm_deep_dbg_{stock_name}"):
                import pandas as pd, numpy as np
                revenue_candidates = ["Quarterly Revenue","Q_Revenue","Revenue","Sales","Q_Sales","Q_TotalRevenue"]
                np_candidates      = ["Quarterly Net Profit","Q_NetProfit","Q_Profit","Q_NetIncome","NetProfit","NetIncome"]
                eps_candidates     = ["Quarterly EPS","Q_EPS","EPS","Basic EPS","Diluted EPS","EPS (Basic)","EPS (Diluted)"]
                shares_candidates  = ["Number of Shares","SharesOutstanding","ShareOutstanding","ShareCount",
                                      "BasicShares","NumShares","Number of shares","Q_NumShares"]
                price_candidates   = ["CurrentPrice","Q_EndQuarterPrice","Q_SharePrice","Each end per every quarter price","Price"]

                def first_present(cands):
                    return next((c for c in cands if c in stock.columns), None)

                rev_col  = first_present(revenue_candidates)
                np_col   = first_present(np_candidates)
                eps_col  = first_present(eps_candidates)
                sh_col   = first_present(shares_candidates)
                p_col    = first_present(price_candidates)

                st.write("Resolved columns by presence:", {
                    "Revenue": rev_col, "NetProfit": np_col, "EPS": eps_col, "Shares": sh_col, "Any price col": p_col
                })

                q = stock[stock.get("IsQuarter", False) == True].copy()
                cols_to_show = ["Year", "Quarter"] + [c for c in [rev_col, np_col, eps_col, sh_col, p_col] if c]
                st.write("Quarterly (tail 6):")
                st.dataframe(q[cols_to_show].tail(6), use_container_width=True)

                def to_num(s):
                    return calculations._to_num(s) if s is not None else pd.Series(dtype="float64")

                rev_ttm = to_num(q.get(rev_col)).tail(4).sum() if rev_col else None
                np_ttm  = to_num(q.get(np_col)).tail(4).sum()  if np_col  else None
                sh4     = to_num(q.get(sh_col)).tail(4).dropna() if sh_col else pd.Series(dtype="float64")
                sh_avg  = sh4.mean() if not sh4.empty else None
                eps_sum = to_num(q.get(eps_col)).tail(4).sum() if eps_col else None

                eps_fallback = (eps_sum is None) or (isinstance(eps_sum, (int,float,np.floating)) and float(eps_sum) == 0.0)
                eps_ttm_dbg = eps_sum
                if eps_fallback and (np_ttm is not None) and (sh_avg and sh_avg > 0):
                    eps_ttm_dbg = float(np_ttm) / float(sh_avg)

                st.write({
                    "DEBUG rev_ttm(sum last 4)": rev_ttm,
                    "DEBUG np_ttm(sum last 4)":  np_ttm,
                    "DEBUG shares last4":        sh4.tolist() if not sh4.empty else [],
                    "DEBUG shares_avg":          sh_avg,
                    "DEBUG eps_ttm (sum or fallback)": eps_ttm_dbg,
                    "Chosen CurrentPrice":       cur_val,
                    "compute_ttm() output":      ttm
                })

            if ttm:
                # ==== Build a corrected TTM snapshot from the raw quarterlies (no other functions touched) ====
                import pandas as pd, numpy as np

                # Helpers
                def _q_to_int(q):
                    try:
                        return int(str(q).upper().replace("Q","").strip())
                    except Exception:
                        return None

                def _coalesce(df, cols):
                    """Row-wise first-non-null across header variants into one Series."""
                    s = None
                    for c in cols:
                        if c in df.columns:
                            s = df[c] if s is None else s.combine_first(df[c])
                    return pd.to_numeric(s, errors="coerce") if s is not None else pd.Series(dtype="float64")

                def _last4_nonnull(series):
                    s = pd.to_numeric(series, errors="coerce").dropna()
                    return s.tail(4)

                # Pull & sort quarterly rows
                q = stock[stock.get("IsQuarter", False) == True].copy()
                if "Quarter" in q.columns:
                    q["_Qn"] = q["Quarter"].apply(_q_to_int)
                else:
                    q["_Qn"] = np.nan
                sort_keys = [k for k in ["Year", "_Qn"] if k in q.columns]
                if sort_keys:
                    q = q.sort_values(sort_keys, kind="mergesort")
                q = q.drop(columns=["_Qn"], errors="ignore")

                # Recognize BOTH compact and verbose headings (with •)
                rev_cols  = ["Quarterly Income Statement • Quarterly Revenue","Quarterly Revenue","Q_Revenue","Revenue","Sales","Q_Sales","Q_TotalRevenue"]
                gp_cols   = ["Quarterly Income Statement • Quarterly Gross Profit","Q_GrossProfit","GrossProfit","Quarterly Gross Profit"]
                op_cols   = ["Quarterly Income Statement • Quarterly Operating Profit","Q_OperatingProfit","Q_EBIT","OperatingProfit","EBIT","Quarterly Operating Profit"]
                np_cols   = ["Quarterly Income Statement • Quarterly Net Profit","Quarterly Net Profit","Q_NetProfit","Q_Profit","Q_NetIncome","NetProfit","NetIncome"]
                eps_cols  = ["Quarterly Income Statement • Quarterly EPS","Quarterly EPS","Q_EPS","EPS","Basic EPS","Diluted EPS","EPS (Basic)","EPS (Diluted)"]
                sh_cols   = ["Quarterly Balance Sheet • Number of Shares","Q_NumShares","Number of Shares","SharesOutstanding","ShareOutstanding","ShareCount","BasicShares","NumShares","Number of shares"]
                px_cols   = ["CurrentPrice","Q_EndQuarterPrice","Q_SharePrice","Each end per every quarter price","Price"]
                ebitda_cols = ["Quarterly Cash Flow Statement • Quarterly EBITDA (optional)","Quarterly EBITDA","Q_EBITDA","EBITDA (Quarterly)"]
                eq_cols   = ["Quarterly Balance Sheet • Shareholder Equity","Q_ShareholderEquity","ShareholderEquity","Shareholder Equity","TotalEquity","Total Equity","Equity"]
                debt_cols = ["TotalDebt","Debt","Borrowings"]
                cash_cols = ["Cash","CashAndEquivalents","Cash & Equivalents"]

                # Coalesce per metric
                Rev_q  = _coalesce(q, rev_cols)
                GP_q   = _coalesce(q, gp_cols)
                OP_q   = _coalesce(q, op_cols)
                NP_q   = _coalesce(q, np_cols)
                EPS_q  = _coalesce(q, eps_cols)
                SH_q   = _coalesce(q, sh_cols)
                PX_q   = _coalesce(q, px_cols)
                EBITDA_q = _coalesce(q, ebitda_cols)

                # Take the latest 4 NON-NULL values per metric
                Rev4, GP4, OP4, NP4, EPS4 = map(_last4_nonnull, [Rev_q, GP_q, OP_q, NP_q, EPS_q])
                SH4 = _last4_nonnull(SH_q)
                EBITDA4 = _last4_nonnull(EBITDA_q)

                # Core TTM flows
                rev_ttm = float(Rev4.sum()) if not Rev4.empty else None
                gp_ttm  = float(GP4.sum())  if not GP4.empty  else None
                op_ttm  = float(OP4.sum())  if not OP4.empty  else None
                np_ttm  = float(NP4.sum())  if not NP4.empty  else None
                ebitda_ttm = float(EBITDA4.sum()) if not EBITDA4.empty else (float(ttm.get("TTM EBITDA")) if ttm.get("TTM EBITDA") is not None else None)

                # EPS TTM: prefer quarterly EPS sum; fallback to NP / avg shares
                eps_ttm = float(EPS4.sum()) if not EPS4.empty else None
                sh_avg  = float(SH4.mean()) if not SH4.empty and SH4.mean() > 0 else None
                if (eps_ttm is None or eps_ttm == 0.0) and (np_ttm is not None) and (sh_avg is not None):
                    eps_ttm = np_ttm / sh_avg

                # Margins
                gm_ttm = (gp_ttm / rev_ttm * 100.0) if (gp_ttm is not None and rev_ttm not in (None, 0)) else None
                nm_ttm = (np_ttm / rev_ttm * 100.0) if (np_ttm is not None and rev_ttm not in (None, 0)) else None

                # Price & valuation
                # prefer page price (cur_val) else latest quarterly price
                price_now = None
                try:
                    price_now = float(cur_val) if cur_val not in (None, np.nan) else None
                except Exception:
                    price_now = None
                if price_now is None:
                    px_last = PX_q.dropna().tail(1)
                    price_now = float(px_last.iloc[0]) if not px_last.empty else None

                # latest shares (quarterly, else annual)
                shares_latest = float(SH_q.dropna().tail(1).iloc[0]) if not SH_q.dropna().empty else None
                if shares_latest is None and "NumShares" in stock.columns:
                    try:
                        shares_latest = float(pd.to_numeric(stock["NumShares"], errors="coerce").dropna().tail(1).iloc[0])
                    except Exception:
                        shares_latest = None

                market_cap = (price_now * shares_latest) if (price_now is not None and shares_latest not in (None, 0)) else None
                pe_ttm = (price_now / eps_ttm) if (price_now is not None and eps_ttm and eps_ttm > 0) else None
                ps_ttm = (market_cap / rev_ttm) if (market_cap is not None and rev_ttm and rev_ttm > 0) else None

                # P/B via latest BVPS if present on the page; else MarketCap / Equity(latest)
                bvps_cols = ["BVPS","Book Value Per Share","BookValuePerShare","NTA per share"]
                BVPS_all = _coalesce(stock, bvps_cols)
                bvps_last = float(BVPS_all.dropna().tail(1).iloc[0]) if not BVPS_all.dropna().empty else None

                if bvps_last and bvps_last > 0 and price_now is not None:
                    pb_ttm = price_now / bvps_last
                    equity_latest = None
                else:
                    EQ_q = _coalesce(q, eq_cols)
                    equity_latest = float(EQ_q.dropna().tail(1).iloc[0]) if not EQ_q.dropna().empty else None
                    pb_ttm = (market_cap / equity_latest) if (market_cap is not None and equity_latest not in (None, 0)) else None

                # EV/EBITDA
                Debt_q = _coalesce(q, debt_cols)
                Cash_q = _coalesce(q, cash_cols)
                debt_latest = float(Debt_q.dropna().tail(1).iloc[0]) if not Debt_q.dropna().empty else None
                cash_latest = float(Cash_q.dropna().tail(1).iloc[0]) if not Cash_q.dropna().empty else None
                net_debt = (debt_latest or 0.0) - (cash_latest or 0.0)
                ev_ebitda = ((market_cap + net_debt) / ebitda_ttm) if (market_cap is not None and ebitda_ttm not in (None, 0)) else None

                # Merge with existing ttm dict so keys you already reference still exist
                ttm_fixed = {
                    **ttm,
                    "TTM Revenue": rev_ttm,
                    "TTM Gross Profit": gp_ttm,
                    "TTM Operating Profit": op_ttm,
                    "TTM Net Profit": np_ttm,
                    "TTM EBITDA": ebitda_ttm,
                    "TTM EPS": eps_ttm,
                    "TTM Gross Margin (%)": gm_ttm,
                    "TTM Net Margin (%)": nm_ttm,
                    "P/E (TTM)": pe_ttm,
                    "P/S (TTM)": ps_ttm,
                    "P/B (TTM)": pb_ttm,
                    "EV/EBITDA (TTM)": ev_ebitda,
                    "MarketCap": market_cap,
                }

                # ==== UI (unchanged style – only feeding from ttm_fixed) ====
                st.markdown(
                    '<div class="sec success"><div class="t">🕒 TTM (Trailing 12 Months)</div>'
                    '<div class="d">Sums of the latest 4 quarters; ratios derived from those totals</div></div>',
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4 = st.columns(4)

                def _fmt(v, d=2):
                    try:
                        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                            return "–"
                        return f"{float(v):,.{d}f}"
                    except Exception:
                        return "–"

                with c1:
                    st.metric("TTM Revenue", _fmt(ttm_fixed.get("TTM Revenue"), 0))
                    st.metric("TTM Net Profit", _fmt(ttm_fixed.get("TTM Net Profit"), 0))
                with c2:
                    gm = ttm_fixed.get("TTM Gross Margin (%)")
                    nm = ttm_fixed.get("TTM Net Margin (%)")
                    st.metric("TTM Gross Margin", f"{_fmt(gm, 2)}%" if gm is not None else "–")
                    st.metric("TTM Net Margin", f"{_fmt(nm, 2)}%" if nm is not None else "–")
                with c3:
                    st.metric("TTM EPS", _fmt(ttm_fixed.get("TTM EPS"), 4))
                    st.metric("P/E (TTM)", _fmt(ttm_fixed.get("P/E (TTM)"), 2))
                with c4:
                    st.metric("P/S (TTM)", _fmt(ttm_fixed.get("P/S (TTM)"), 2))
                    st.metric("P/B (TTM)", _fmt(ttm_fixed.get("P/B (TTM)"), 2))
                    st.metric("EV/EBITDA (TTM)", _fmt(ttm_fixed.get("EV/EBITDA (TTM)"), 2))

            # TTM — details expander (calculation audit)  ← REPLACED
            with st.expander("Show calculation details — TTM", expanded=False):
                # Build the "recent quarters" table (same as before)
                q = stock[stock.get("IsQuarter", False) == True].copy()
                q["Year"]  = pd.to_numeric(q["Year"], errors="coerce")
                q["Qnum"]  = pd.to_numeric(q["Quarter"].astype(str).str.extract(r"(\d)", expand=False), errors="coerce")
                q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])

                # Use the columns you already resolved earlier in the TTM section
                show_cols = ["Year", "Quarter"]
                tmp_cols = []
                for ser, label in [(Rev_q, "Revenue"), (GP_q, "Gross Profit"), (OP_q, "Operating Profit"),
                                (NP_q, "Net Profit"), (EPS_q, "EPS"), (SH_q, "Shares"), (PX_q, "Price")]:
                    if ser is not None and not ser.empty:
                        q[label] = ser
                        tmp_cols.append(label)
                show_cols += tmp_cols

                st.caption("Recent quarters (most recent at bottom). TTM sums take the latest 4 rows.")
                st.dataframe(q[show_cols].tail(8), use_container_width=True, height=260)

                # Pretty formatting helpers (no JSON printing)
                import numpy as _np
                def _f0(x): return "–" if x is None or (isinstance(x, float) and not _np.isfinite(x)) else f"{float(x):,.0f}"
                def _f2(x): return "–" if x is None or (isinstance(x, float) and not _np.isfinite(x)) else f"{float(x):,.2f}"
                def _f4(x): return "–" if x is None or (isinstance(x, float) and not _np.isfinite(x)) else f"{float(x):,.4f}"

                # Nice heading like photo 2
                try:
                    last_year = int(q.tail(1)["Year"].iloc[0])
                    st.markdown(f"**TTM window:** latest four quarters → to {last_year}")
                except Exception:
                    pass

                # Human-readable formulas (use variables computed earlier in the TTM section)
                st.markdown(
                    f"- **TTM Revenue** = sum(last 4 quarters) = **{_f0(rev_ttm)}**  \n"
                    f"- **TTM Gross Margin** = GrossProfit_TTM ÷ Revenue_TTM = {_f0(gp_ttm)} ÷ {_f0(rev_ttm)} = **{_f2(gm_ttm)}%**  \n"
                    f"- **TTM Operating Margin** = OperatingProfit_TTM ÷ Revenue_TTM = {_f0(op_ttm)} ÷ {_f0(rev_ttm)} = **{_f2((op_ttm/rev_ttm*100) if (op_ttm not in (None,0) and rev_ttm not in (None,0)) else None)}%**  \n"
                    f"- **TTM Net Profit** = sum(last 4 quarters) = **{_f0(np_ttm)}**  \n"
                    f"- **TTM Net Margin** = NetProfit_TTM ÷ Revenue_TTM = {_f0(np_ttm)} ÷ {_f0(rev_ttm)} = **{_f2(nm_ttm)}%**  \n"
                    f"- **TTM EPS** = **{_f4(eps_ttm)}**  \n"
                    f"- **P/E (TTM)** = Price ÷ EPS_TTM = {_f4(price_now)} ÷ {_f4(eps_ttm)} = **{_f2(pe_ttm)}**  \n"
                    f"- **P/S (TTM)** = MarketCap ÷ Revenue_TTM = {_f0(market_cap)} ÷ {_f0(rev_ttm)} = **{_f2(ps_ttm)}**  \n"
                    f"- **P/B (TTM)** = MarketCap ÷ Equity(latest) = {_f0(market_cap)} ÷ {_f0(equity_latest)} = **{_f2(pb_ttm)}**  \n"
                    f"- **EV/EBITDA (TTM)** = (MarketCap + NetDebt) ÷ EBITDA_TTM = ({_f0(market_cap)} + {_f0((debt_latest or 0) - (cash_latest or 0))}) ÷ {_f0(ebitda_ttm)} = **{_f2(ev_ebitda)}**"
                )

                # ✅ Keep the download (unchanged)
                ttm_payload = {
                    "stock": stock_name,
                    "type": "ttm_snapshot",
                    "inputs": {
                        "Price": price_now,
                        "Shares_avg4Q": (float(SH4.mean()) if 'SH4' in locals() and hasattr(SH4, 'mean') and not SH4.dropna().empty else None),
                        "MarketCap": market_cap,
                        "Equity_latest": (equity_latest if 'equity_latest' in locals() else None),
                        "Debt_latest": debt_latest,
                        "Cash_latest": cash_latest
                    },
                    "ttm": {
                        "Revenue": rev_ttm, "GrossProfit": gp_ttm, "OperatingProfit": op_ttm,
                        "NetProfit": np_ttm, "EPS": eps_ttm, "EBITDA": ebitda_ttm
                    },
                    "ratios": {
                        "GrossMargin_%": gm_ttm, "NetMargin_%": nm_ttm,
                        "PE_TTM": pe_ttm, "PS_TTM": ps_ttm, "PB_TTM": pb_ttm, "EV_EBITDA_TTM": ev_ebitda
                    }
                }
                _download_json_button("📥 Download TTM Snapshot (JSON)", ttm_payload, f"{stock_name}_ttm.json", key=f"ttm_snap_{stock_name}")



            # helper: unique widget keys for this stock block
            def k(suffix: str, s=stock_name):
                # default arg binds the current stock_name so keys are stable
                return f"{suffix}_{s}"

        

            # ---- 💧 Cash Flow Wealth (always shown)
            st.markdown(
                '<div class="sec warning"><div class="t">💧 Cash Flow Wealth</div>'
                '<div class="d">Cash generation, reinvestment, and balance-sheet strength</div></div>',
                unsafe_allow_html=True
            )

            def _fmt2(v, d=2):
                try:
                    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        return "–"
                    return f"{float(v):,.{d}f}"
                except Exception:
                    return "–"

            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                st.metric("TTM CFO", _fmt2(ttm.get("TTM CFO"), 0))
                st.metric("Cash Conversion (CFO/NP)", f"{_fmt2(ttm.get('Cash Conversion (CFO/NP, %)'), 2)}%")
            with r1c2:
                st.metric("TTM CapEx", _fmt2(ttm.get("TTM CapEx"), 0))
                st.metric("Debt / FCF (yrs)", _fmt2(ttm.get("Debt / FCF (yrs)"), 2))
            with r1c3:
                st.metric("TTM FCF", _fmt2(ttm.get("TTM FCF"), 0))
                st.metric("FCF / Share (TTM)", _fmt2(ttm.get("FCF / Share (TTM)"), 4))
            with r1c4:
                st.metric("FCF Margin (TTM)", f"{_fmt2((ttm.get('TTM FCF') or 0) / (ttm.get('TTM Revenue') or 1) * 100.0, 2)}%")
                st.metric("FCF Yield (TTM)", f"{_fmt2(ttm.get('FCF Yield (TTM) (%)'), 2)}%")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1:
                st.metric("Net Cash (Debt)", _fmt2(ttm.get("Net Cash (Debt)"), 0))
            with r2c2:
                st.metric("Net Cash / MC", f"{_fmt2(ttm.get('Net Cash / MC (%)'), 2)}%")
            with r2c3:
                st.metric("Interest Coverage", _fmt2(ttm.get("Interest Coverage (EBITDA/Fin)"), 2))
            with r2c4:
                st.metric("Market Cap (est.)", _fmt2((ttm.get("P/S (TTM)") or 0) * (ttm.get("TTM Revenue") or 0), 0))

            
            # Cash-Flow Wealth — details expander (formulas with numbers)
            with st.expander("Show calculation details — Cash Flow Wealth", expanded=False):
                def n(x):
                    try:
                        x = float(x)
                        return x if np.isfinite(x) else np.nan
                    except Exception:
                        return np.nan

                cfo       = n((ttm or {}).get("TTM CFO"))
                capex     = n((ttm or {}).get("TTM CapEx"))
                fcf       = n((ttm or {}).get("TTM FCF"))
                if not np.isfinite(fcf): fcf = n(cfo) - n(capex)

                rev       = n((ttm or {}).get("TTM Revenue"))
                ebitda    = n((ttm or {}).get("TTM EBITDA"))
                fincost   = n((ttm or {}).get("Finance Costs (TTM)"))  # if present
                shares    = n((ttm or {}).get("Shares"))
                mc        = n((ttm or {}).get("MarketCap"))
                fcf_ps    = n((ttm or {}).get("FCF / Share (TTM)"))
                fcf_yield = n((ttm or {}).get("FCF Yield (TTM) (%)"))
                cash_conv = n((ttm or {}).get("Cash Conversion (CFO/NP, %)"))
                net_cash  = n((ttm or {}).get("Net Cash (Debt)"))
                netcash_mc_pct = n((ttm or {}).get("Net Cash / MC (%)"))
                debt_fcf  = n((ttm or {}).get("Debt / FCF (yrs)"))

                # If MarketCap missing, rough fallback from P/S × Revenue
                ps = n((ttm or {}).get("P/S (TTM)"))
                if not np.isfinite(mc) and np.isfinite(ps) and np.isfinite(rev):
                    mc = ps * rev

                fcf_margin = (fcf / rev * 100.0) if (np.isfinite(fcf) and np.isfinite(rev) and rev != 0) else np.nan
                if not np.isfinite(fcf_yield) and np.isfinite(fcf) and np.isfinite(mc) and mc != 0:
                    fcf_yield = fcf / mc * 100.0
                icov = (ebitda / fincost) if (np.isfinite(ebitda) and np.isfinite(fincost) and fincost != 0) else np.nan

                st.markdown(
                    f"- **FCF** = CFO − CapEx = { (cfo if np.isfinite(cfo) else np.nan):,.0f} − { (capex if np.isfinite(capex) else np.nan):,.0f} = **{ (fcf if np.isfinite(fcf) else np.nan):,.0f}**  \n"
                    f"- **FCF Margin (TTM)** = FCF ÷ Revenue_TTM = { (fcf if np.isfinite(fcf) else np.nan):,.0f} ÷ { (rev if np.isfinite(rev) else np.nan):,.0f} = **{ (fcf_margin if np.isfinite(fcf_margin) else np.nan):,.2f}%**  \n"
                    f"- **FCF / Share (TTM)** = FCF ÷ Shares = { (fcf if np.isfinite(fcf) else np.nan):,.0f} ÷ { (shares if np.isfinite(shares) else np.nan):,.0f} = **{ (fcf_ps if np.isfinite(fcf_ps) else np.nan):,.4f}**  \n"
                    f"- **FCF Yield (TTM)** = FCF ÷ MarketCap = { (fcf if np.isfinite(fcf) else np.nan):,.0f} ÷ { (mc if np.isfinite(mc) else np.nan):,.0f} = **{ (fcf_yield if np.isfinite(fcf_yield) else np.nan):,.2f}%**  \n"
                    f"- **Cash Conversion (CFO/NP)** = **{ (cash_conv if np.isfinite(cash_conv) else np.nan):,.2f}%**  \n"
                    f"- **Net Cash (Debt)** = **{ (net_cash if np.isfinite(net_cash) else np.nan):,.0f}**  \n"
                    f"- **Net Cash / MarketCap** = **{ ((netcash_mc_pct) if np.isfinite(netcash_mc_pct) else np.nan):,.2f}%**  \n"
                    f"- **Interest Coverage** ≈ EBITDA ÷ Finance Costs = { (ebitda if np.isfinite(ebitda) else np.nan):,.0f} ÷ { (fincost if np.isfinite(fincost) else np.nan):,.0f} = **{ (icov if np.isfinite(icov) else np.nan):,.2f}×**  \n"
                    f"- **Debt / FCF (yrs)** = **{ (debt_fcf if np.isfinite(debt_fcf) else np.nan):,.2f}**"
                )

                cfw_payload = {
                    "stock": stock_name, "type": "cashflow_wealth_detail",
                    "inputs": {
                        "CFO_TTM": cfo, "CapEx_TTM": capex, "Revenue_TTM": rev,
                        "MarketCap": mc, "Shares": shares, "EBITDA_TTM": ebitda,
                        "FinanceCost_TTM": fincost
                    },
                    "derived": {
                        "FCF": fcf, "FCF_Margin_%": fcf_margin, "FCF_per_Share": fcf_ps,
                        "FCF_Yield_%": fcf_yield, "Cash_Conversion_%": cash_conv,
                        "Net_Cash": net_cash, "NetCash_over_MC_%": netcash_mc_pct,
                        "Interest_Coverage": icov, "Debt_over_FCF_years": debt_fcf
                    }
                }
                _download_json_button("📥 Download Cash-flow Wealth (JSON)",
                                    cfw_payload, f"{stock_name}_cashflow_wealth.json",
                                    key=f"dl_cfw_{stock_name}")

                                            
                            


            # === Snowflake & Momentum (inside the same Annual tab) ===

            # === Momentum inputs from uploaded price file (trading-view CSV) ===
            ohlc_latest = None
            try:
                # load_ohlc reads /data/ohlc/<Name>.csv (set by the Momentum page)
                try:
                    import io_helpers as _ioh
                except Exception:
                    from utils import io_helpers as _ioh

                oh = _ioh.load_ohlc(stock_name)
                if oh is not None and not oh.empty:
                    ohlc_df = oh.copy()  # <-- IMPORTANT: do NOT overwrite the master 'df'
                    if "Date" in ohlc_df.columns:
                        ohlc_df["Date"] = pd.to_datetime(ohlc_df["Date"], errors="coerce")
                    ohlc_df["Close"] = pd.to_numeric(ohlc_df.get("Close"), errors="coerce")
                    ohlc_df = (
                        ohlc_df.dropna(subset=["Date", "Close"])
                            .sort_values("Date")
                            .reset_index(drop=True)
                    )

                    if not ohlc_df.empty:
                        close = ohlc_df["Close"]
                        price = float(close.iloc[-1])

                        # 200-DMA needs >=200 trading rows
                        ma200 = float(close.rolling(200, min_periods=200).mean().iloc[-1]) if len(close) >= 200 else None

                        # 12-month return: prefer ~252 trading days; fallback to 365-day window
                        ret_12m = None
                        if len(close) >= 252 and pd.notna(close.iloc[-252]) and float(close.iloc[-252]) != 0.0:
                            base = float(close.iloc[-252])
                            ret_12m = float(price / base - 1.0)
                        else:
                            cutoff = ohlc_df["Date"].iloc[-1] - pd.Timedelta(days=365)
                            win = ohlc_df[ohlc_df["Date"] >= cutoff]
                            if len(win) >= 2:
                                base = float(win["Close"].iloc[0])
                                if base != 0.0:
                                    ret_12m = float(price / base - 1.0)

                        ohlc_latest = {"price": price, "ma200": ma200, "ret_12m": ret_12m}
            except Exception:
                ohlc_latest = None



            # Friendly messages so you know why Momentum may not score fully
            if ohlc_latest is None:
                st.info("Momentum: no price file found for this stock (upload daily CSV in the Momentum page).")
            else:
                msgs = []
                if ohlc_latest.get("ma200") is None:
                    msgs.append("need ≥200 daily rows for 200-DMA")
                if ohlc_latest.get("ret_12m") is None:
                    msgs.append("need ≥252 daily rows for 12-month return")
                if msgs:
                    st.warning("Momentum not fully scored — " + " & ".join(msgs))



            # ---- ❄️ Snowflake — Overall Performance --------------------
            try:
                scores = calculations.compute_factor_scores(
                    stock_name, stock, ttm, ohlc_latest=ohlc_latest, industry=None
                )
                base   = ["Value", "Quality", "Growth", "Cash", "Momentum"]
                labels = ["Future Value", "Earnings Quality", "Growth Consistency",
                        "Cash Strength", "Momentum"]
                vals   = [int(scores.get(k, 0) or 0) for k in base]
            except Exception:
                scores = {}  # keep defined for detail table below
                labels = ["Future Value", "Earnings Quality", "Growth Consistency",
                        "Cash Strength", "Momentum"]
                vals   = [0, 0, 0, 0, 0]

            # Warn if EPS YoY didn’t hit 8/8 (strict rule on last 12 quarters)
            det = scores.get("_detail", [])
            if isinstance(det, list) and any(d.get("Input","").startswith("EPS YoY %") and d.get("Score") is None for d in det):
                miss = [d for d in det if d.get("Input","").startswith("EPS YoY %")]
                pf = miss[0].get("Components", {}).get("pairs_found", 0) if miss else 0
                st.warning(f"EPS YoY not scored — only {pf}/8 pairs available in the last 12 quarters.")

            # ---- Radar --------------------------------------------------
            lab2 = labels + [labels[0]]
            val2 = vals   + [vals[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=val2, theta=lab2, fill="toself", name=stock_name,
                hovertemplate="<b>%{theta}</b>: %{r}/100<extra></extra>"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=420
            )
            st.markdown("#### Snowflake — Overall Performance")
            st.plotly_chart(fig, use_container_width=True)

            # ---- Composite & pillar mini-cards --------------------------
            composite = int(round(sum(vals) / max(1, len(vals))))
            st.metric("Composite Score", f"{composite}/100")
            _cols = st.columns(5)
            for i, (lab, v) in enumerate(zip(labels, vals)):
                with _cols[i]:
                    st.metric(lab, f"{v}/100")

            # ---- Data Health (avoid nested expanders) -------------------
            dh = (ttm or {}).get("DataHealth", {}) if ttm else {}
            ok = not dh.get("missing")
            st.markdown(f"**Data Health:** {'✅ OK' if ok else '⚠️ Incomplete'}")


            # ---- Snowflake details (expander + JSON) --------------------
            with st.expander("Show calculation details — Snowflake", expanded=False):
                detail = scores.get("_detail", [])
                if detail:
                    detail_df = pd.DataFrame(detail)  # <-- use a local name; do NOT overwrite 'df'
                    detail_df["Raw"] = detail_df["Raw"].apply(
                        lambda x: "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):,.4f}"
                    )
                    detail_df["Score"] = detail_df["Score"].apply(
                        lambda x: "–" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{int(round(float(x))):d}"
                    )

                    st.dataframe(
                        detail_df[["Pillar", "Input", "Raw", "Score", "Source"]],
                        use_container_width=True, height=300
                    )

                    # component drill-down (HTML <details>)
                    for _, row in detail_df.iterrows():
                        comp = row.get("Components")
                        if comp:
                            st.markdown(
                                f"<details><summary><b>{row['Pillar']} · {row['Input']} components</b></summary>",
                                unsafe_allow_html=True,
                            )
                            for k, v in comp.items():
                                if v is None or (isinstance(v, float) and np.isnan(v)):
                                    st.markdown(f"- **{k}**: –")
                                else:
                                    st.markdown(f"- **{k}**: {float(v):,.4f}")
                            st.markdown("</details>", unsafe_allow_html=True)

                    # 📥 JSON download (scores + detail)
                    composite = int(round(sum([int(scores.get(k,0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5))
                    sf_payload = {
                        "stock": stock_name,
                        "type":  "snowflake_detail",
                        "scores": {k: int(scores.get(k,0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]},
                        "composite": composite,
                        "confidence": scores.get("_confidence", {}),
                        "detail_rows": detail,
                    }
                    _download_json_button(
                        "📥 Download Snowflake Detail (JSON)",
                        sf_payload,
                        f"{stock_name}_snowflake_detail.json",
                        key=f"dl_sf_{stock_name}"
                    )
                else:
                    st.info("Diagnostic data not available.")



            # ---- Annual — Calculated Ratios (styled)
            st.markdown('<div class="sec success"><div class="t">📐 Annual — Calculated Ratios</div></div>', unsafe_allow_html=True)
            st.markdown("#### Calculated Ratios")

            ratios = []
            for _, row in annual.iterrows():
                rowx = row.copy()

                # force valuation metrics to use today's price (P/E, P/B, etc.)
                try:
                    if cur_val is not None and np.isfinite(cur_val):
                        rowx["SharePrice"] = float(cur_val)
                except Exception:
                    pass

                # ---- helpers -------------------------------------------------------
                def _n(v):
                    try:
                        v = float(str(v).replace(",", ""))
                        return v if np.isfinite(v) else np.nan
                    except Exception:
                        return np.nan

                def getv(d, *keys):
                    """Return the first finite value found among keys/aliases (case-insensitive, substring-safe)."""
                    # direct hits
                    for k in keys:
                        if k in d:
                            vv = _n(d.get(k))
                            if np.isfinite(vv):
                                return vv
                    # fuzzy hits (handles verbose column labels)
                    lowers = {str(k).lower(): k for k in d.keys()}
                    for k in keys:
                        t = str(k).lower()
                        for lk, orig in lowers.items():
                            if t == lk or t in lk:
                                vv = _n(d.get(orig))
                                if np.isfinite(vv):
                                    return vv
                    return np.nan
                # -------------------------------------------------------------------

                # Depreciation & Amortization (prefer single field, else sum splits)
                dep_total = getv(
                    rowx,
                    "DepAmort",
                    "Depreciation & Amortization",
                    "Depreciation & Amortisation",
                    "Depreciation and amortisation",
                )
                if not np.isfinite(dep_total):
                    parts = [
                        getv(rowx, "DepPPE",     "Depreciation of property, plant and equipment"),
                        getv(rowx, "DepInvProp", "Depreciation of investment property"),
                        getv(rowx, "DepROU",     "Depreciation of right-of-use assets"),
                        getv(rowx, "AmortIntang","Amortisation of intangible assets"),
                    ]
                    parts = [p for p in parts if np.isfinite(p)]
                    dep_total = float(np.nansum(parts)) if parts else np.nan

                if np.isfinite(dep_total):
                    rowx["DepAmort"] = dep_total  # let ratio engine see it

                # EBITDA fallback:
                # 1) raw EBITDA
                # 2) EBIT/OperatingProfit + DepAmort
                # 3) NetProfit + IncomeTax + FinanceCosts + DepAmort
                ebitda = getv(rowx, "EBITDA")
                if not np.isfinite(ebitda):
                    ebit = getv(rowx, "EBIT", "OperatingProfit", "Operating profit")
                    if np.isfinite(ebit) and np.isfinite(dep_total):
                        ebitda = ebit + dep_total
                    else:
                        npf = getv(rowx, "NetProfit", "Net profit", "ProfitAfterTax", "PATMI")
                        tax = getv(rowx, "IncomeTax", "Income tax expense", "Tax")
                        fin = getv(rowx, "FinanceCosts", "Finance costs", "InterestExpense", "Finance cost")
                        parts = [v for v in (npf, tax, fin, dep_total) if np.isfinite(v)]
                        ebitda = float(np.nansum(parts)) if parts else np.nan

                if np.isfinite(ebitda):
                    rowx["EBITDA"] = ebitda

                # run your ratio engine
                r = calculations.calc_ratios(rowx)

                # convenience rows in the table (so you can see cash/debt lines)
                total_debt = np.nansum([
                    getv(rowx, "Borrowings"),
                    getv(rowx, "OtherLoans"),
                    getv(rowx, "LeaseLiabCurrent", "Current lease liabilities"),
                    getv(rowx, "LeaseLiabNonCurrent", "Non-current lease liabilities"),
                ])
                total_debt = (np.nan if (not np.isfinite(total_debt) or total_debt == 0.0) else float(total_debt))

                r.update({
                    "CapEx":                       getv(rowx, "CapEx", "CapitalExpenditure", "Purchase of property, plant and equipment"),
                    "Operating Cash Flow":         getv(rowx, "CFO", "OperatingCashFlow", "Net cash flow generated from/(used in) operating activities"),
                    "Depreciation & Amortization": (dep_total if np.isfinite(dep_total) else np.nan),
                    "EBITDA":                      (ebitda if np.isfinite(ebitda) else r.get("EBITDA")),
                    "Total Debt / Borrowings":     total_debt,
                })

                r["Year"] = row.get("Year")
                ratios.append(r)

            # build table
            ratio_df = pd.DataFrame(ratios).set_index("Year").apply(pd.to_numeric, errors="coerce").round(4)


            # keep Year as int so you don't get 2019.000000
            if not ratio_df.empty:
                ratio_df.index = pd.to_numeric(ratio_df.index, errors="coerce").astype("Int64")


            # ---- Append a TTM row to the annual RATIOS table (computed from last 4 quarters)
            ttm_detail_export = None

            try:
                q = quarterly.copy()
                q["Year"] = pd.to_numeric(q["Year"], errors="coerce")
                qnum = q["Quarter"].astype(str).str.extract(r"(\d)", expand=False)
                q["Qnum"] = pd.to_numeric(qnum, errors="coerce")
                q = q.dropna(subset=["Year", "Qnum"]).sort_values(["Year", "Qnum"])
                q4 = q.tail(4).copy()

                if not q4.empty:
                    last_year = int(q4["Year"].iloc[-1])

                    # --- compatibility for older column names ---
                    if "Q_IncomeTax" not in q4.columns and "Q_Tax" in q4.columns:
                        q4["Q_IncomeTax"] = q4["Q_Tax"]
                    # if no splits but old Q_DepAmort exists, treat it as DepPPE so DepAmort can be built
                    if ("Q_DepPPE" not in q4.columns and
                        "Q_DepInvProp" not in q4.columns and
                        "Q_DepROU" not in q4.columns and
                        "Q_DepAmort" in q4.columns):
                        q4["Q_DepPPE"] = q4["Q_DepAmort"]

                    def _sum_q(col):
                        return float(pd.to_numeric(q4.get(col), errors="coerce").dropna().sum()) \
                            if col in q4.columns else np.nan

                    def _last_q(col):
                        s = pd.to_numeric(q4.get(col) if col in q4.columns else quarterly.get(col),
                                        errors="coerce").dropna()
                        return float(s.iloc[-1]) if not s.empty else np.nan

                    # ---- build a synthetic “annual-like” row so calc_ratios() can reuse your formulas
                    synth = {
                        # 4Q sums (P&L)
                        "Revenue":          _sum_q("Q_Revenue"),
                        "GrossProfit":      _sum_q("Q_GrossProfit"),
                        "NetProfit":        _sum_q("Q_NetProfit"),
                        "CostOfSales":      _sum_q("Q_CostOfSales"),
                        "FinanceCosts":     _sum_q("Q_FinanceCosts"),
                        "AdminExpenses":    _sum_q("Q_AdminExpenses"),
                        "SellDistExpenses": _sum_q("Q_SellDistExpenses"),

                        # Cash Flow (4Q)
                        "CFO":              _sum_q("Q_CFO"),
                        "CapEx":            _sum_q("Q_CapEx"),
                        "IncomeTax":        _sum_q("Q_IncomeTax"),
                    }

                    # Depreciation total from splits (fallback to old Q_DepAmort if that’s all you have)
                    dep_parts = [_sum_q("Q_DepPPE"), _sum_q("Q_DepInvProp"), _sum_q("Q_DepROU")]
                    if any(pd.notna(v) for v in dep_parts):
                        synth["DepAmort"] = float(np.nansum([v for v in dep_parts if pd.notna(v)]))
                    elif "Q_DepAmort" in q4.columns:
                        synth["DepAmort"] = _sum_q("Q_DepAmort")

                    # Build EBITDA (no quarterly EBITDA input): NP + Tax + FinanceCosts + DepAmort
                    ebitda_ttm = np.nan
                    parts = [synth.get("NetProfit"), synth.get("IncomeTax"), synth.get("FinanceCosts"), synth.get("DepAmort")]
                    parts = [float(p) for p in parts if p is not None and np.isfinite(float(p))]
                    if parts:
                        ebitda_ttm = float(np.nansum(parts))
                        synth["EBITDA"] = ebitda_ttm

                    # latest snapshot items (balances)
                    synth.update({
                        "NumShares":          _last_q("Q_NumShares"),
                        "CurrentAsset":       _last_q("Q_CurrentAsset"),
                        "CurrentLiability":   _last_q("Q_CurrentLiability"),
                        "Inventories":        _last_q("Q_Inventories"),
                        "IntangibleAsset":    _last_q("Q_IntangibleAsset"),
                        "TotalAsset":         _last_q("Q_TotalAsset"),
                        "TotalLiability":     _last_q("Q_TotalLiability"),
                        "ShareholderEquity":  _last_q("Q_ShareholderEquity"),
                        "SharePrice":         (float(cur_val) if (cur_val is not None and np.isfinite(cur_val)) else _last_q("Q_EndQuarterPrice")),
                        "Dividend":           0.0,  # keep 0 unless you have quarterly DPS
                    })

                    # Prefer the live/current price for TTM calculations
                    try:
                        if cur_val is not None and np.isfinite(cur_val):
                            synth["SharePrice"] = float(cur_val)
                    except Exception:
                        pass

                    # Prefer a true TTM EPS from compute_ttm; else fallback to NP ÷ latest shares
                    ttm_core = calculations.compute_ttm(stock, current_price=cur_val)
                    if ttm_core and ttm_core.get("TTM EPS") is not None:
                        synth["EPS"] = float(ttm_core["TTM EPS"])
                    else:
                        if synth.get("NumShares") and synth.get("NetProfit"):
                            try:
                                synth["EPS"] = float(synth["NetProfit"]) / float(synth["NumShares"])
                            except Exception:
                                pass

                    # Total Debt snapshot (latest quarter): include borrowings + other loans + leases
                    debt_components = {
                        "Borrowings":          _last_q("Q_Borrowings"),
                        "OtherLoans":          _last_q("Q_OtherLoans"),
                        "LeaseLiabCurrent":    _last_q("Q_LeaseLiabCurrent"),
                        "LeaseLiabNonCurrent": _last_q("Q_LeaseLiabNonCurrent"),
                    }
                    total_debt_latest = float(np.nansum([v for v in debt_components.values() if v is not None and np.isfinite(v)]))
                    if total_debt_latest == 0.0:
                        total_debt_latest = np.nan

                    # run your ratio engine
                    r_ttm = calculations.calc_ratios(synth)

                    # --- same convenience fields for TTM row (mirror Annual ratios table)
                    r_ttm.update({
                        "CapEx":                       synth.get("CapEx"),
                        "Operating Cash Flow":         synth.get("CFO"),
                        "Depreciation & Amortization": synth.get("DepAmort"),
                        "EBITDA":                      (synth.get("EBITDA") if np.isfinite(synth.get("EBITDA") or np.nan) else r_ttm.get("EBITDA")),
                        "Total Debt / Borrowings":     total_debt_latest,
                    })

                    r_ttm["Year"] = f"TTM {last_year}"

                    # keep for the JSON download later (richer — includes debt parts & EBITDA build)
                    ttm_detail_export = {
                        "label": r_ttm["Year"],
                        "components": {k: (None if pd.isna(v) else float(v)) for k, v in synth.items()},
                        "debt_components_latest": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in debt_components.items()},
                        "ebitda_build": {
                            "formula": "NetProfit + IncomeTax + FinanceCosts + DepAmort",
                            "NetProfit": synth.get("NetProfit"),
                            "IncomeTax": synth.get("IncomeTax"),
                            "FinanceCosts": synth.get("FinanceCosts"),
                            "DepAmort": synth.get("DepAmort"),
                            "EBITDA_TTM": synth.get("EBITDA"),
                        },
                        "ratios": {k: (None if pd.isna(v) else float(v)) for k, v in r_ttm.items() if k != "Year"},
                    }

                    # append as a new “year” row; dtype will become object (ok)
                    ratio_df = pd.concat([ratio_df, pd.DataFrame([r_ttm]).set_index("Year")], axis=0)
                    ratio_df = ratio_df.round(4)
            except Exception:
                pass





            # ... later when you build the other orientation:
            #     disp = ratio_df.T
            # add this just after the transpose, before styling:
            if 'disp' in locals():
                try:
                    disp.columns = [int(c) for c in disp.columns]
                except Exception:
                    pass


            if ratio_df.empty:
                st.info("No ratio data available.")
            else:
                ratio_layout = st.radio(
                    "Calculated ratios table layout (annual)",
                    ["Metrics → columns (Year rows)", "Years → columns (Metric rows)"],
                    horizontal=True,
                    key=f"annual_ratio_layout_{stock_name}"
                )

                if ratio_layout.startswith("Years"):
                    disp = ratio_df.T
                    new_cols_r = drag_reorder(
                        disp.columns.tolist(),
                        key_suffix=f"ann_ratio_yearcols_{stock_name}",
                        help_text="Drag to reorder **Year** columns."
                    )
                    disp = disp[new_cols_r]
                    styled = style_ratio_table(disp)
                    _show_styled(styled, height=_auto_height(disp, row_h=34, base=96, max_h=1400))
                    st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")



                    st.session_state[f"ann_ratio_x_{stock_name}"] = [str(x) for x in disp.index.astype(str).tolist()]
                    st.session_state[f"ann_ratio_metrics_{stock_name}"] = disp.index.tolist()
                else:
                    disp = ratio_df
                    new_cols_r = drag_reorder(
                        disp.columns.tolist(),
                        key_suffix=f"ann_ratio_metriccols_{stock_name}",
                        help_text="Drag to reorder **Metric** columns."
                    )
                    disp = disp[new_cols_r]
                    styled = style_ratio_table(disp)
                    _show_styled(styled, height=_auto_height(disp, row_h=34, base=96, max_h=1400))
                    st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")



                    st.session_state[f"ann_ratio_x_{stock_name}"] = [str(x) for x in disp.index.astype(str).tolist()]
                    st.session_state[f"ann_ratio_metrics_{stock_name}"] = new_cols_r
                    
                    
            # Annual ratios — details (formulas with numbers)
            with st.expander("Show calculation details — Annual ratios", expanded=False):
                # Options = numeric years + optional TTM label we built earlier
                years_only = []
                for y in ratio_df.index:
                    try:
                        years_only.append(int(str(y)))
                    except Exception:
                        pass
                year_labels = [str(y) for y in sorted(set(years_only))]

                ttm_label = (ttm_detail_export or {}).get("label")
                if ttm_label:
                    year_labels.append(ttm_label)  # e.g. "TTM 2025"

                if not year_labels:
                    st.info("No annual history.")
                else:
                    sel_label = st.selectbox(
                        "Year / TTM",
                        options=year_labels,
                        index=len(year_labels) - 1,
                        key=f"ann_ratio_detail_year_{stock_name}",
                    )

                    def n(x):
                        try:
                            x = float(str(x).replace(",", ""))
                            return x if np.isfinite(x) else np.nan
                        except Exception:
                            return np.nan

                    # ----- TTM branch -----
                    if ttm_label and sel_label == ttm_label:
                        comp = (ttm_detail_export or {}).get("components", {})
                        npf  = n(comp.get("NetProfit"))
                        gp   = n(comp.get("GrossProfit"))
                        rev  = n(comp.get("Revenue"))
                        eq   = n(comp.get("ShareholderEquity"))
                        sh   = n(comp.get("NumShares"))
                        ca   = n(comp.get("CurrentAsset"))
                        cl   = n(comp.get("CurrentLiability"))
                        inv  = n(comp.get("Inventories"))
                        price= n(comp.get("SharePrice"))
                        dps  = n(comp.get("Dividend"))
                        inta = n(comp.get("IntangibleAsset"))

                        eps   = (npf / sh) if (np.isfinite(npf) and np.isfinite(sh) and sh != 0) else np.nan
                        roe   = (npf / eq * 100.0) if (np.isfinite(npf) and np.isfinite(eq) and eq != 0) else np.nan
                        pe    = (price / eps) if (np.isfinite(price) and np.isfinite(eps) and eps != 0) else np.nan
                        mc    = (price * sh) if (np.isfinite(price) and np.isfinite(sh)) else np.nan
                        pb    = (mc / eq) if (np.isfinite(mc) and np.isfinite(eq) and eq != 0) else np.nan
                        gpm   = (gp / rev * 100.0) if (np.isfinite(gp) and np.isfinite(rev) and rev != 0) else np.nan
                        npm   = (npf / rev * 100.0) if (np.isfinite(npf) and np.isfinite(rev) and rev != 0) else np.nan
                        curr  = (ca / cl) if (np.isfinite(ca) and np.isfinite(cl) and cl != 0) else np.nan
                        quick = ((ca - inv) / cl) if (np.isfinite(ca) and np.isfinite(inv) and np.isfinite(cl) and cl != 0) else np.nan
                        dy    = (dps / price * 100.0) if (np.isfinite(dps) and np.isfinite(price) and price != 0) else np.nan
                        bvps  = (eq / sh) if (np.isfinite(eq) and np.isfinite(sh) and sh != 0) else np.nan
                        nta   = ((eq - inta) / sh) if (np.isfinite(eq) and np.isfinite(inta) and np.isfinite(sh) and sh != 0) else np.nan

                        st.markdown(f"**{sel_label}**")
                        st.markdown(
                            f"- **EPS** = Net Profit ÷ Shares = {npf:,.0f} ÷ {sh:,.0f} = **{eps:,.4f}**  \n"
                            f"- **ROE** = Net Profit ÷ Equity = {npf:,.0f} ÷ {eq:,.0f} = **{roe:,.2f}%**  \n"
                            f"- **P/E** = Price ÷ EPS = {price:,.4f} ÷ {eps:,.4f} = **{pe:,.2f}**  \n"
                            f"- **P/B** = (Price×Shares) ÷ Equity = {(mc if np.isfinite(mc) else np.nan):,.0f} ÷ {eq:,.0f} = **{pb:,.2f}**  \n"
                            f"- **Gross Margin** = Gross Profit ÷ Revenue = {gp:,.0f} ÷ {rev:,.0f} = **{gpm:,.2f}%**  \n"
                            f"- **Net Margin** = Net Profit ÷ Revenue = {npf:,.0f} ÷ {rev:,.0f} = **{npm:,.2f}%**  \n"
                            f"- **Current Ratio** = Current Assets ÷ Current Liabilities = {ca:,.0f} ÷ {cl:,.0f} = **{curr:,.2f}×**  \n"
                            f"- **Quick Ratio** ≈ (CA − Inventories) ÷ CL = {(ca - inv) if np.isfinite(ca) and np.isfinite(inv) else np.nan:,.0f} ÷ {cl:,.0f} = **{quick:,.2f}×**  \n"
                            f"- **Dividend Yield** ≈ Dividend/share ÷ Price = {dps:,.4f} ÷ {price:,.4f} = **{dy:,.2f}%**  \n"
                            f"- **BVPS** = Equity ÷ Shares = {eq:,.0f} ÷ {sh:,.0f} = **{bvps:,.4f}**  \n"
                            + (f"- **NTA / share** = (Equity − Intangibles) ÷ Shares = {(eq - inta):,.0f} ÷ {sh:,.0f} = **{nta:,.4f}**"
                            if np.isfinite(inta) else "")
                        )

                        st.markdown("---")
                        ratio_json = {
                            "stock": stock_name,
                            "type": "annual_ratios",
                            "selected": sel_label,
                            "ratios_by_year": _records(ratio_df.reset_index()),
                        }
                        _download_json_button(
                            "📥 Download Annual Ratios (JSON)",
                            ratio_json,
                            f"{stock_name}_annual_ratios.json",
                            key=f"ann_ratios_{stock_name}_bottom",
                        )

                    # ----- normal annual-year branch (your existing calculations) -----
                    else:
                        sel_year = int(float(sel_label))

                        yr_col = pd.to_numeric(annual.get("Year", pd.Series(index=annual.index)), errors="coerce")
                        raw_rec = annual[yr_col == sel_year].tail(1).to_dict("records")
                        raw = raw_rec[0] if raw_rec else {}

                        npf = n(raw.get("NetProfit"))
                        gp  = n(raw.get("GrossProfit") or raw.get("Gross Profit"))
                        rev = n(raw.get("Revenue"))
                        eq  = n(raw.get("ShareholderEquity") or raw.get("Shareholder Equity") or raw.get("Equity"))
                        sh  = n(raw.get("NumShares") or raw.get("Number of Shares") or raw.get("Number of shares") or raw.get("ShareOutstanding"))
                        ca  = n(raw.get("CurrentAsset"))
                        cl  = n(raw.get("CurrentLiability"))
                        inv = n(raw.get("Inventories"))
                        price = n(raw.get("SharePrice") or raw.get("Current Share Price") or raw.get("End of year share price") or raw.get("Each end of year share price"))
                        dps   = n(raw.get("Dividend") or raw.get("Dividend pay cent"))
                        inta  = n(raw.get("IntangibleAsset"))

                        eps   = (npf / sh) if (np.isfinite(npf) and np.isfinite(sh) and sh != 0) else np.nan
                        roe   = (npf / eq * 100.0) if (np.isfinite(npf) and np.isfinite(eq) and eq != 0) else np.nan
                        pe    = (price / eps) if (np.isfinite(price) and np.isfinite(eps) and eps != 0) else np.nan
                        mc    = (price * sh) if (np.isfinite(price) and np.isfinite(sh)) else np.nan
                        pb    = (mc / eq) if (np.isfinite(mc) and np.isfinite(eq) and eq != 0) else np.nan
                        gpm   = (gp / rev * 100.0) if (np.isfinite(gp) and np.isfinite(rev) and rev != 0) else np.nan
                        npm   = (npf / rev * 100.0) if (np.isfinite(npf) and np.isfinite(rev) and rev != 0) else np.nan
                        curr  = (ca / cl) if (np.isfinite(ca) and np.isfinite(cl) and cl != 0) else np.nan
                        quick = ((ca - inv) / cl) if (np.isfinite(ca) and np.isfinite(inv) and np.isfinite(cl) and cl != 0) else np.nan
                        dy    = (dps / price * 100.0) if (np.isfinite(dps) and np.isfinite(price) and price != 0) else np.nan
                        bvps  = (eq / sh) if (np.isfinite(eq) and np.isfinite(sh) and sh != 0) else np.nan
                        nta   = ((eq - inta) / sh) if (np.isfinite(eq) and np.isfinite(inta) and np.isfinite(sh) and sh != 0) else np.nan

                        st.markdown(
                            f"- **EPS** = Net Profit ÷ Shares = {npf:,.0f} ÷ {sh:,.0f} = **{eps:,.4f}**  \n"
                            f"- **ROE** = Net Profit ÷ Equity = {npf:,.0f} ÷ {eq:,.0f} = **{roe:,.2f}%**  \n"
                            f"- **P/E** = Price ÷ EPS = {price:,.4f} ÷ {eps:,.4f} = **{pe:,.2f}**  \n"
                            f"- **P/B** = (Price×Shares) ÷ Equity = {(mc if np.isfinite(mc) else np.nan):,.0f} ÷ {eq:,.0f} = **{pb:,.2f}**  \n"
                            f"- **Gross Margin** = Gross Profit ÷ Revenue = {gp:,.0f} ÷ {rev:,.0f} = **{gpm:,.2f}%**  \n"
                            f"- **Net Margin** = Net Profit ÷ Revenue = {npf:,.0f} ÷ {rev:,.0f} = **{npm:,.2f}%**  \n"
                            f"- **Current Ratio** = Current Assets ÷ Current Liabilities = {ca:,.0f} ÷ {cl:,.0f} = **{curr:,.2f}×**  \n"
                            f"- **Quick Ratio** ≈ (CA − Inventories) ÷ CL = {(ca - inv) if np.isfinite(ca) and np.isfinite(inv) else np.nan:,.0f} ÷ {cl:,.0f} = **{quick:,.2f}×**  \n"
                            f"- **Dividend Yield** ≈ Dividend/share ÷ Price = {dps:,.4f} ÷ {price:,.4f} = **{dy:,.2f}%**  \n"
                            f"- **BVPS** = Equity ÷ Shares = {eq:,.0f} ÷ {sh:,.0f} = **{bvps:,.4f}**  \n"
                            + (f"- **NTA / share** = (Equity − Intangibles) ÷ Shares = {(eq - inta):,.0f} ÷ {sh:,.0f} = **{nta:,.4f}**"
                            if np.isfinite(inta) else "")
                        )

                        st.markdown("---")
                        try:
                            _yr = ratio_df.copy()
                            _mx = _yr.T
                        except Exception:
                            _yr = pd.DataFrame(); _mx = pd.DataFrame()

                        ratio_payload = {
                            "stock": stock_name,
                            "industry": industry,
                            "industry_bucket": industry_bucket,
                            "layout": ratio_layout,
                            "year_rows":   _records(_yr.reset_index()),
                            "metric_rows": _records(_mx.reset_index()),
                            "ttm_detail": (ttm_detail_export or {}),
                        }
                        _download_json_button(
                            "📥 Download Annual Ratios (JSON)",
                            ratio_payload,
                            f"{stock_name}_annual_ratios.json",
                            key=f"ann_ratios_{stock_name}_bottom",
                        )




            # ---- Undervalue/Overvalue Bar
            st.markdown("#### Undervalue/Overvalue Bar")
            pe = None
            if not ratio_df.empty:
                pe = ratio_df.iloc[-1].get("P/E", None)
            if pe is not None and pe == pe:
                if pe < 15:
                    st.success(f"P/E = {pe:.2f} (Undervalued)")
                elif pe < 25:
                    st.info(f"P/E = {pe:.2f} (Fair Value)")
                else:
                    st.error(f"P/E = {pe:.2f} (Overvalued)")
                st.progress(min(max((25 - pe) / 25, 0), 1))
            else:
                st.info("Not enough data for value bar.")

            # ---- Advanced Growth & Valuation Metrics (CAGR, PEG, Graham, MOS)
            # Toggle the lookback window (always anchors to the latest year available)
            win_label = st.radio(
                "CAGR window", ["3 years", "5 years"],
                index=0, horizontal=True, key=f"cagr_win_{stock_name}"
            )
            win = 5 if "5" in win_label else 3

            def _safe_float(x):
                try:
                    v = float(x)
                    return v if np.isfinite(v) else None
                except Exception:
                    return None

            def _fmt_num(x, d=2, commas=True):
                if x is None: return "N/A"
                try:
                    x = float(x)
                    return f"{x:,.{d}f}" if commas else f"{x:.{d}f}"
                except Exception:
                    return "N/A"

            def _fmt_pct(x, d=2):
                if x is None: return "N/A"
                try:
                    x = float(x)
                    return f"{x:.{d}f}%"
                except Exception:
                    return "N/A"

            # Figure out which years we can use (from your annual table)
            years = sorted(annual["Year"].dropna().astype(int).unique())
            if len(years) >= 2:
                last_year = years[-1]
                # earliest year within the last N years (inclusive), falling back gracefully
                first_candidates = [y for y in years if (last_year - y) <= win and y <= last_year]
                first_year = first_candidates[0] if first_candidates else years[0]
                period = max(1, last_year - first_year)  # protect against division by zero

                # Pull inputs from your prepared tables
                # ann_numeric is the numeric annual-wide table with MultiIndex (Section,Field)
                rev_first = _safe_float(ann_numeric.loc[("Income Statement", "Revenue"),     str(first_year)])
                rev_last  = _safe_float(ann_numeric.loc[("Income Statement", "Revenue"),     str(last_year)])
                np_first  = _safe_float(ann_numeric.loc[("Income Statement", "Net Profit"),  str(first_year)])
                np_last   = _safe_float(ann_numeric.loc[("Income Statement", "Net Profit"),  str(last_year)])

                # EPS & BVPS from the annual ratios table (ratio_df; index is Year)
                eps_first  = _safe_float(ratio_df.loc[first_year, "EPS"])  if first_year in ratio_df.index else None
                eps_last   = _safe_float(ratio_df.loc[last_year,  "EPS"])  if last_year  in ratio_df.index else None
                bvps_first = _safe_float(ratio_df.loc[first_year, "BVPS"]) if first_year in ratio_df.index else None
                bvps_last  = _safe_float(ratio_df.loc[last_year,  "BVPS"]) if last_year  in ratio_df.index else None

                # Gross Profit from annual table (exists in your ann_numeric)
                gp_first = _safe_float(ann_numeric.loc[("Income Statement", "Gross Profit"), str(first_year)])
                gp_last  = _safe_float(ann_numeric.loc[("Income Statement", "Gross Profit"), str(last_year)])

                # 1) CAGRs (compounded)
                cagr_rev  = ((rev_last  / rev_first)  ** (1/period) - 1) * 100 if (rev_first  and rev_first  > 0 and rev_last)  else None
                cagr_np   = ((np_last   / np_first)   ** (1/period) - 1) * 100 if (np_first   and np_first   > 0 and np_last)   else None
                cagr_eps  = ((eps_last  / eps_first)  ** (1/period) - 1) * 100 if (eps_first  and eps_first  > 0 and eps_last)  else None
                cagr_gp   = ((gp_last   / gp_first)   ** (1/period) - 1) * 100 if (gp_first   and gp_first   > 0 and gp_last)   else None
                cagr_bvps = ((bvps_last / bvps_first) ** (1/period) - 1) * 100 if (bvps_first and bvps_first > 0 and bvps_last) else None

                # 2) Est EPS (compound forward using EPS CAGR over 'win' years)
                est_eps_ny = eps_last * (1 + (cagr_eps or 0)/100) ** win if eps_last is not None and cagr_eps is not None else None

                # 3) Latest P/E and PEG
                last_pe = _safe_float(ratio_df.iloc[-1].get("P/E")) if not ratio_df.empty else None
                peg     = (last_pe / cagr_eps) if (last_pe is not None and cagr_eps and cagr_eps > 0) else None

                # 4) Graham value & Margin of Safety (use EPS CAGR as 'g' in % form)
                g_for_graham = (cagr_eps or 0)  # classic Graham uses % number (e.g., 6.5 for 6.5%)
                graham_val   = eps_last * (8.5 + 2 * g_for_graham) if eps_last is not None else None
                mos          = ((graham_val - cur_val) / graham_val * 100) if (graham_val and graham_val > 0) else None

                # ---- Render the metrics
                st.markdown("#### 🚀 Growth & Valuation Metrics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rev CAGR (%)", _fmt_pct(cagr_rev))
                c1.metric("NP  CAGR (%)", _fmt_pct(cagr_np))
                c2.metric(f"Est EPS ({win}-yr)", _fmt_num(est_eps_ny, d=4))
                c2.metric("P/E Ratio", _fmt_num(last_pe, d=2))
                c3.metric("EPS CAGR (%)", _fmt_pct(cagr_eps))
                c3.metric("PEG Ratio", _fmt_num(peg, d=2, commas=False))
                c4.metric("Graham Value", _fmt_num(graham_val, d=2))
                c4.metric("Margin of Safety", _fmt_pct(mos))

                # second row for the extra (“balance”) CAGRs
                c5, c6, _, _ = st.columns(4)
                c5.metric("Gross Profit CAGR (%)", _fmt_pct(cagr_gp))
                c6.metric("BVPS CAGR (%)",         _fmt_pct(cagr_bvps))

                # ---- Debug / Audit: show data used and formulas with numbers
                with st.expander(f"Show calculation details ({win}-year window)", expanded=False):
                    st.markdown(
                        f"""
            **Window**: {first_year} → {last_year}  &nbsp;&nbsp;|&nbsp;&nbsp;  **Years in window**: {period}  
            **Chosen horizon for forecast**: {win} year(s) (from latest year {last_year})
                        """
                    )

                    # --- Download JSON payload (built first so it's ready at the bottom)
                    gv_json = {
                        "stock": stock_name,
                        "type": "growth_valuation_window",
                        "window_years": int(win),
                        "window": {"from": int(first_year), "to": int(last_year), "period": int(period)},
                        "inputs": {
                            "Revenue_first": rev_first, "Revenue_last": rev_last,
                            "NP_first": np_first, "NP_last": np_last,
                            "EPS_first": eps_first, "EPS_last": eps_last,
                            "GP_first": gp_first, "GP_last": gp_last,                 # NEW
                            "BVPS_first": bvps_first, "BVPS_last": bvps_last,         # NEW
                            "Latest_PE": last_pe, "CurrentPrice": cur_val,
                        },
                        "outputs": {
                            "Rev_CAGR_pct": cagr_rev,
                            "NP_CAGR_pct": cagr_np,
                            "EPS_CAGR_pct": cagr_eps,
                            "GP_CAGR_pct": cagr_gp,           # NEW
                            "BVPS_CAGR_pct": cagr_bvps,       # NEW
                            "Est_EPS_forward": est_eps_ny,
                            "PEG": peg,
                            "GrahamValue": graham_val,
                            "MarginOfSafety_pct": mos,
                        },
                    }

                    # Inputs table
                    dbg_df = pd.DataFrame({
                        "Metric": [
                            "Revenue (first)", "Revenue (last)",
                            "Net Profit (first)", "Net Profit (last)",
                            "EPS (first)", "EPS (last)",
                            "Gross Profit (first)", "Gross Profit (last)",   # NEW
                            "BVPS (first)", "BVPS (last)",                   # NEW
                            "Latest P/E", "Latest Price"
                        ],
                        "Value": [
                            _fmt_num(rev_first, d=0),
                            _fmt_num(rev_last,  d=0),
                            _fmt_num(np_first,   d=0),
                            _fmt_num(np_last,    d=0),
                            _fmt_num(eps_first,  d=4),
                            _fmt_num(eps_last,   d=4),
                            _fmt_num(gp_first,   d=0),                       # NEW
                            _fmt_num(gp_last,    d=0),                       # NEW
                            _fmt_num(bvps_first, d=4),                       # NEW
                            _fmt_num(bvps_last,  d=4),                       # NEW
                            _fmt_num(last_pe,    d=2, commas=False),
                            _fmt_num(cur_val,    d=4)
                        ],
                        "Year": [
                            first_year, last_year,
                            first_year, last_year,
                            first_year, last_year,
                            first_year, last_year,   # GP
                            first_year, last_year,   # BVPS
                            last_year,  last_year
                        ]
                    })
                    st.table(dbg_df)

                    # Formulas with numbers plugged in
                    def _safe(x): return "N/A" if x is None else f"{x:,.6g}"
                    st.markdown("**Formulas (numbers plugged in):**")

                    st.markdown(
            f"""
            - **Rev CAGR** = \\(((Rev_last / Rev_first)^{{1/period}} - 1) × 100\\)  
            = (({_safe(rev_last)} / {_safe(rev_first)})^(1/{period}) - 1) × 100 = **{_fmt_pct(cagr_rev)}**
            - **NP CAGR**  = \\(((NP_last / NP_first)^{{1/period}} - 1) × 100\\)  
            = (({_safe(np_last)} / {_safe(np_first)})^(1/{period}) - 1) × 100 = **{_fmt_pct(cagr_np)}**
            - **EPS CAGR** = \\(((EPS_last / EPS_first)^{{1/period}} - 1) × 100\\)  
            = (({_safe(eps_last)} / {_safe(eps_first)})^(1/{period}) - 1) × 100 = **{_fmt_pct(cagr_eps)}**
            - **GP CAGR**  = \\(((GP_last / GP_first)^{{1/period}} - 1) × 100\\)  
            = (({_safe(gp_last)} / {_safe(gp_first)})^(1/{period}) - 1) × 100 = **{_fmt_pct(cagr_gp)}**
            - **BVPS CAGR** = \\(((BVPS_last / BVPS_first)^{{1/period}} - 1) × 100\\)  
            = (({_safe(bvps_last)} / {_safe(bvps_first)})^(1/{period}) - 1) × 100 = **{_fmt_pct(cagr_bvps)}**
            - **Est EPS ({win}-yr)** = EPS_last × (1 + EPS_CAGR/100)^{{{win}}}  
            = {_safe(eps_last)} × (1 + {(_safe(cagr_eps))}/100)^{win} = **{_fmt_num(est_eps_ny, d=4)}**
            - **PEG** = P/E ÷ EPS_CAGR = {(_safe(last_pe))} ÷ {(_safe(cagr_eps))} = **{_fmt_num(peg, d=2, commas=False)}**
            - **Graham Value** = EPS_last × (8.5 + 2 × EPS_CAGR)  
            = {_safe(eps_last)} × (8.5 + 2 × {(_safe(cagr_eps))}) = **{_fmt_num(graham_val, d=2)}**
            - **Margin of Safety** = ((Graham − Price) ÷ Graham) × 100  
            = (({_safe(graham_val)} − {_safe(cur_val)}) ÷ {_safe(graham_val)}) × 100 = **{_fmt_pct(mos)}**
            """
                    )

                    # small cross-check series for the window
                    try:
                        yrs_window = [y for y in years if first_year <= y <= last_year]
                        eps_series = ratio_df.loc[yrs_window, "EPS"] if len(yrs_window) else pd.Series([], dtype=float)
                        rev_series = pd.Series({y: ann_numeric.loc[("Income Statement","Revenue"), str(y)] for y in yrs_window})
                        np_series  = pd.Series({y: ann_numeric.loc[("Income Statement","Net Profit"), str(y)] for y in yrs_window})
                        gp_series  = pd.Series({y: ann_numeric.loc[("Income Statement","Gross Profit"), str(y)] for y in yrs_window})
                        bv_series  = pd.Series({y: ratio_df.loc[y, "BVPS"] if y in ratio_df.index else None for y in yrs_window})
                        st.markdown("**Window series (for cross-check):**")
                        st.dataframe(pd.DataFrame({
                            "EPS": eps_series, "Revenue": rev_series, "Net Profit": np_series,
                            "Gross Profit": gp_series, "BVPS": bv_series
                        }))
                    except Exception:
                        pass

                    # ----- download AT BOTTOM of the expander -----
                    _download_json_button(
                        "📥 Download Growth & Valuation (JSON)",
                        gv_json, f"{stock_name}_growth_valuation_{win}y.json",
                        key=f"gv_{stock_name}_{win}_bottom"
                    )

            else:
                st.info("Not enough annual history for CAGR/PEG/Graham.")



            # ---- Annual comparison charts …
            st.markdown('<div class="sec warning"><div class="t">📊 Annual — Comparison Charts</div></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Annual Comparison Charts (up to 4)")

            # A) Raw Data comparisons (YoY)
            st.markdown("##### Raw Data (select up to 4 series to compare across Years)")
            ann_opts = field_options(ANNUAL_SECTIONS)
            years_order = st.session_state.get(
                f"ann_raw_x_{stock_name}",
                [str(int(y)) for y in sorted(annual['Year'].dropna().unique())]
            )
            def ann_series_getter(sec_lbl):
                if ann_numeric.empty:
                    return None
                sec, lbl = sec_lbl
                if (sec, lbl) not in ann_numeric.index:
                    return None
                y = ann_numeric.loc[(sec, lbl), :]
                return pd.Series(y).reindex([str(x) for x in years_order]).values

            ann_count = st.slider("Number of raw-data charts", 1, 4, 2, key=f"ann_raw_chartcount_{stock_name}")
            multi_panel_charts(
                ann_count, ann_opts, years_order, ann_series_getter,
                key_prefix=f"annual_raw_chart_{stock_name}", chart_height=320
            )

            # B) Calculated Ratios comparisons (YoY)
            st.markdown("##### Calculated Ratios (select up to 4 ratios to compare across Years)")

            # DEFAULT X (years + optional 'TTM YYYY') — keep as strings
            ratio_x = st.session_state.get(
                f"ann_ratio_x_{stock_name}",
                [str(x) for x in ratio_df.index.tolist()]
            )
            # normalize in case session held ints previously
            ratio_x = [str(x) for x in ratio_x]

            ratio_metrics = st.session_state.get(
                f"ann_ratio_metrics_{stock_name}",
                list(ratio_df.columns)
            )

            def ratio_series_getter(metric_name):
                if metric_name not in ratio_df.columns:
                    return None
                y = ratio_df[metric_name]
                y = pd.Series(y.values, index=y.index.astype(str))
                return y.reindex([str(x) for x in ratio_x]).values

            ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"ann_ratio_chartcount_{stock_name}")
            multi_panel_charts(
                ratio_count, [(m, m) for m in ratio_metrics], ratio_x, ratio_series_getter,
                key_prefix=f"annual_ratio_chart_{stock_name}", chart_height=320
            )


            # ---- Systematic Decision (read-only) — use the same scorer as page 4 ----
            st.markdown(
                '<div class="sec danger"><div class="t">🚦 Systematic Decision</div>'
                '<div class="d">Read-only evaluation (same funnel as the Systematic Decision page)</div></div>',
                unsafe_allow_html=True
            )
            st.markdown("### 🚦 Systematic Decision")

            annual_rows_for_eval = annual.sort_values("Year")
            if annual_rows_for_eval.empty:
                st.info("No annual data to evaluate.")
            else:
                # Determine bucket (fallback to General)
                bucket = "General"
                if "IndustryBucket" in stock.columns:
                    bser = stock["IndustryBucket"].astype("string").dropna()
                    if not bser.empty:
                        bucket = bser.iloc[-1] or "General"

                # Optional momentum (if OHLC exists)
                ohlc_latest = None
                try:
                    o = io_helpers.load_ohlc(stock_name)
                    if o is not None and not o.empty and "Close" in o.columns:
                        close = pd.to_numeric(o["Close"], errors="coerce").dropna()
                        if not close.empty:
                            price = float(close.iloc[-1])
                            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
                            ret12 = float(price / close.iloc[-252] - 1.0) if len(close) >= 252 and close.iloc[-252] != 0 else None
                            ohlc_latest = {"price": price, "ma200": ma200, "ret_12m": ret12}
                except Exception:
                    pass

                # Same scorer as the Systematic Decision page
                res = calculations.compute_industry_scores(
                    stock_name=stock_name,
                    stock_df=stock,
                    bucket=bucket,
                    entry_price=cur_val,
                    fd_rate=float(getattr(config, "FD_RATE", 0.035)),
                    ohlc_latest=ohlc_latest,
                )

                gates  = res.get("gates", {}) or {}
                blocks = res.get("blocks", {}) or {}

                cA, cB = st.columns([1, 3])
                with cA:
                    st.metric("Score", res.get("composite", 0))
                    st.markdown("**Status:** " + ("✅ PASS" if res.get("decision") == "PASS" else "❌ REJECT"))
                with cB:
                    why = (res.get("why", []) + gates.get("notes", [])) or []
                    if why:
                        st.warning("Unmet / Notes: " + "; ".join([str(x) for x in why if x]))

                # ——— Simple “Show checks” (like your example), derived from TTM metrics ———
                with st.expander("Show checks", expanded=False):
                    ttm = calculations.compute_ttm(stock, current_price=cur_val) or {}

                    def ok(cond): return "✅" if cond else "❌"

                    # Mandatory (same idea as your example)
                    mand = [
                        (ok((ttm.get("TTM EPS") or 0) > 0),                    "EPS TTM > 0"),
                        (ok((ttm.get("Interest Coverage (EBITDA/Fin)") or 0) >= 3), "InterestCoverage ≥ 3"),
                        (ok(((ttm.get("Debt / FCF (yrs)") or 1e9) <= 5) or (ttm.get("TTM FCF") or 0) > 0),
                        "Debt/FCF ≤ 5 or FCF>0"),
                    ]
                    st.markdown("**Mandatory checks**")
                    for tick, label in mand:
                        st.write(f"{tick} {label}")

                    # Scored (simple 5×20pts display only)
                    scored = [
                        (ok((ttm.get("P/E (TTM)") or 1e9) <= 15),                "P/E (TTM) ≤ 15",            20),
                        (ok((ttm.get("FCF Yield (TTM) (%)") or 0) >= 5),         "FCF Yield (TTM) ≥ 5%",      20),
                        (ok((ttm.get("Cash Conversion (CFO/NP, %)") or 0) >= 80),"Cash Conversion ≥ 80%",     20),
                        (ok((ttm.get("TTM Gross Margin (%)") or 0) >= 20),       "TTM Gross Margin ≥ 20%",    20),
                        (ok((ttm.get("TTM Net Margin (%)") or 0) >= 5),          "TTM Net Margin ≥ 5%",       20),
                    ]
                    st.markdown("**Scored checks**")
                    for tick, label, pts in scored:
                        st.write(f"{tick} {label} ({pts} pts)")


# =========================
# QUARTERLY
# =========================
with tabs[1]:
    st.subheader(f"{stock_name} - Quarterly Report")

    # ---- Quarterly — Raw Data
    st.markdown('<div class="sec info"><div class="t">📄 Quarterly — Raw Data</div></div>', unsafe_allow_html=True)
    st.markdown("#### Raw Quarterly Data")
    q_numeric = build_quarter_raw_numeric(quarterly)
    q_raw_layout = st.radio(
        "Raw data layout (quarterly)",
        ["Fields → columns (Period rows)", "Periods → columns (Field rows)"],
        horizontal=True,
        key=f"quarter_raw_layout_{stock_name}"
    )
    if q_numeric.empty:
        st.info("No quarterly raw data available.")
    else:
        if q_raw_layout.startswith("Periods"):
            disp_qnum = _reorder_empty_last(q_numeric)
            new_cols = drag_reorder(
                disp_qnum.columns.tolist(),
                key_suffix=f"q_raw_periodcols_{stock_name}",
                help_text="Drag to reorder **Period** columns."
            )
            disp_qnum = disp_qnum[new_cols]
        else:
            disp_qnum = _reorder_empty_last(q_numeric.T)
            field_cols = disp_qnum.columns.tolist()
            new_cols = drag_reorder(
                field_cols,
                key_suffix=f"q_raw_fieldcols_{stock_name}",
                help_text="Drag to reorder **Field** columns."
            )
            disp_qnum = disp_qnum[new_cols]


    # ---- Quarterly Raw table (styled only)
    if q_raw_layout.startswith("Periods"):
        styled_q = style_raw_spike_table(disp_qnum, is_columns_period=True)   # periods = columns
    else:
        styled_q = style_raw_spike_table(disp_qnum, is_columns_period=False)  # periods = rows
    _show_styled(styled_q, height=_auto_height(disp_qnum, row_h=34, base=96, max_h=1400))
    st.caption("⚠️ Alert: Highlighted cells show ≥100% QoQ jump vs the previous quarter, and a material change (≥5% of median level).")

    with st.expander("Download — Quarterly Raw (JSON)", expanded=False):
        q_payload = {
            "stock":  stock_name,
            "type":   "quarterly_raw",
            "layout": "Periods→columns" if q_raw_layout.startswith("Periods") else "Fields→columns",
            "visible_table": _records(disp_qnum.reset_index() if isinstance(disp_qnum, pd.DataFrame) else pd.DataFrame()),            
        }
        _download_json_button("📥 Download Quarterly Raw (JSON)",
                            q_payload,
                            f"{stock_name}_quarterly_raw.json",
                            key=f"quarterly_raw_{stock_name}")







    # ---- Quarterly — Calculated Ratios (STYLED)  [INSIDE THE TAB]
    st.markdown('<div class="sec success"><div class="t">📐 Quarterly — Calculated Ratios</div></div>', unsafe_allow_html=True)
    st.markdown("#### Quarterly Calculated Ratios")

    qratios = []
    for _, row in quarterly.iterrows():
        rowx = row.copy()

        # let calc_ratios do the base work
        r = calculations.calc_ratios(rowx)

        # add pass-through convenience fields (mirror Annual ratios)
        import numpy as _np
        def _n(v):
            try:
                v = float(str(v).replace(",", ""))
                return v if _np.isfinite(v) else _np.nan
            except Exception:
                return _np.nan

        dep_total = _np.nan
        parts = [_n(rowx.get("Q_DepPPE")), _n(rowx.get("Q_DepInvProp")), _n(rowx.get("Q_DepROU"))]
        if any(_np.isfinite(p) for p in parts):
            dep_total = float(_np.nansum([p for p in parts if _np.isfinite(p)]))
        elif _np.isfinite(_n(rowx.get("Q_DepAmort"))):  # backward-compat
            dep_total = _n(rowx.get("Q_DepAmort"))

        total_debt = 0.0
        for k in ("Q_Borrowings", "Q_OtherLoans"):
            v = _n(rowx.get(k))
            if _np.isfinite(v):
                total_debt += v
        # (optional) include leases in "debt":
        # for k in ("Q_LeaseLiabCurrent","Q_LeaseLiabNonCurrent"):
        #     v = _n(rowx.get(k));  total_debt += (v if _np.isfinite(v) else 0.0)

        r.update({
            "CapEx":                       _n(rowx.get("Q_CapEx")),
            "Operating Cash Flow":         _n(rowx.get("Q_CFO")),
            "Depreciation & Amortization": dep_total,
            # If calc_ratios derives EBITDA, this will be filled; otherwise shows blank
            "EBITDA":                      r.get("EBITDA"),
            "Total Debt / Borrowings":     (_np.nan if total_debt == 0.0 else total_debt),
        })

        r["Year"] = rowx.get("Year")
        r["Quarter"] = rowx.get("Quarter")
        qratios.append(r)

    qratio_df = pd.DataFrame(qratios)

    # Normalize to Period index (e.g., "2024 Q3")
    if not qratio_df.empty:
        if "Quarter" in qratio_df.columns:
            qratio_df["Qnum"] = qratio_df["Quarter"].map(quarter_key_to_num)
            qratio_df = qratio_df.dropna(subset=["Year", "Qnum"])
            qratio_df["Year"] = qratio_df["Year"].astype(int)
            qratio_df = qratio_df.sort_values(["Year", "Qnum"])
            qratio_df["Period"] = qratio_df["Year"].astype(str) + " Q" + qratio_df["Qnum"].astype(int).astype(str)
            qratio_df = qratio_df.drop(columns=["Qnum"]).set_index("Period")
        elif "Year" in qratio_df.columns:
            qratio_df = qratio_df.set_index("Year")

    if qratio_df.empty:
        st.info("No quarterly ratio data available.")
    else:
        # strong numeric coercion so thresholds apply
        qratio_df = qratio_df.apply(pd.to_numeric, errors="coerce").round(4)

        qratio_layout = st.radio(
            "Calculated ratios table layout (quarterly)",
            ["Metrics → columns (Period rows)", "Periods → columns (Metric rows)"],
            horizontal=True,
            key=f"quarter_ratio_layout_{stock_name}"
        )

        if qratio_layout.startswith("Periods"):
            # Periods → columns (metrics as rows)
            disp_qratio = qratio_df.T
            new_cols = drag_reorder(
                disp_qratio.columns.tolist(),
                key_suffix=f"q_ratio_periodcols_{stock_name}",
                help_text="Drag to reorder **Period** columns."
            )
            disp_qratio = disp_qratio[new_cols]
            styled = style_ratio_table(disp_qratio)
            _show_styled(styled, height=_auto_height(disp_qratio, row_h=34, base=96, max_h=1400))
            



        else:
            # Metrics → columns (periods as rows)
            disp_qratio = qratio_df
            new_cols = drag_reorder(
                disp_qratio.columns.tolist(),
                key_suffix=f"q_ratio_metriccols_{stock_name}",
                help_text="Drag to reorder **Metric** columns."
            )
            disp_qratio = disp_qratio[new_cols]
            styled = style_ratio_table(disp_qratio)
            _show_styled(styled, height=_auto_height(disp_qratio, row_h=34, base=96, max_h=1400))
            st.caption("Legend: 🟩 Great  •  🟦 OK  •  🟥 Fails threshold")
            
            # Quarterly ratios — details (formulas with numbers)
        if not qratio_df.empty:
            with st.expander("Show calculation details — Quarterly ratios", expanded=False):
                # Let user pick a period (e.g., "2024 Q3"); default = latest
                per_opts = qratio_df.index.tolist()
                sel_idx = len(per_opts) - 1 if per_opts else 0
                sel_per = st.selectbox(
                    "Period",
                    options=per_opts,
                    index=sel_idx,
                    key=f"q_ratio_detail_per_{stock_name}",
                )

                # Parse "YYYY QN"
                year_num, q_num = None, None
                m = re.match(r"^\s*(\d{4})\s+Q(\d)\s*$", str(sel_per))
                if m:
                    year_num = int(m.group(1))
                    q_num = int(m.group(2))

                # Get raw quarterly row matching that period
                raw_q = quarterly.copy()

                # Coerce Year to numeric so equality works (string vs int problem)
                if "Year" in raw_q.columns:
                    raw_q["Year"] = pd.to_numeric(raw_q["Year"], errors="coerce")

                # Build a reliable Qnum from whatever is in "Quarter"
                if "Quarter" in raw_q.columns:
                    # Accept formats like "Q3", "2022 Q3", 3, "3"
                    qstr = raw_q["Quarter"].astype(str)
                    qnum_from_text = qstr.str.extract(r"Q\s*(\d)", expand=False)
                    qnum_from_text = pd.to_numeric(qnum_from_text, errors="coerce")

                    # If the column is already 1..4 as numbers/strings, catch that too
                    qnum_simple = pd.to_numeric(qstr, errors="coerce")

                    raw_q["Qnum"] = qnum_from_text.fillna(qnum_simple)

                # Apply filter only if we parsed a valid selection
                if (year_num is not None) and (q_num is not None) and \
                ("Year" in raw_q.columns) and ("Qnum" in raw_q.columns):
                    flt = raw_q[(raw_q["Year"] == year_num) & (raw_q["Qnum"] == q_num)]
                else:
                    flt = pd.DataFrame()

                if flt.empty:
                    st.info("No matching raw quarterly row for the selected period.")
                    rawq = {}
                else:
                    rawq = flt.tail(1).to_dict("records")[0]

                def n(x):
                    try:
                        x = float(str(x).replace(",", ""))
                        return x if np.isfinite(x) else np.nan
                    except Exception:
                        return np.nan

                # Inputs (Q_ columns)
                npf = n(rawq.get("Q_NetProfit"))
                gp  = n(rawq.get("Q_GrossProfit"))
                rev = n(rawq.get("Q_Revenue"))
                eq  = n(rawq.get("Q_ShareholderEquity"))
                sh  = n(rawq.get("Q_NumShares"))
                cur_asset = n(rawq.get("Q_CurrentAsset"))
                cur_liab  = n(rawq.get("Q_CurrentLiability"))
                inv       = n(rawq.get("Q_Inventories"))
                price = n(rawq.get("Q_EndQuarterPrice") or rawq.get("Q_SharePrice"))
                div_ps = n(rawq.get("Q_Dividend"))
                eps    = n(rawq.get("Q_EPS"))

                # Derived (same formulas, quarter-scale)
                if not np.isfinite(eps) and np.isfinite(npf) and np.isfinite(sh) and sh != 0:
                    eps = npf / sh

                roe   = (npf / eq * 100.0) if (np.isfinite(npf) and np.isfinite(eq) and eq != 0) else np.nan
                pe    = (price / eps) if (np.isfinite(price) and np.isfinite(eps) and eps != 0) else np.nan
                mc    = (price * sh) if (np.isfinite(price) and np.isfinite(sh)) else np.nan
                pb    = (mc / eq) if (np.isfinite(mc) and np.isfinite(eq) and eq != 0) else np.nan
                gpm   = (gp / rev * 100.0) if (np.isfinite(gp) and np.isfinite(rev) and rev != 0) else np.nan
                npm   = (npf / rev * 100.0) if (np.isfinite(npf) and np.isfinite(rev) and rev != 0) else np.nan
                curr  = (cur_asset / cur_liab) if (np.isfinite(cur_asset) and np.isfinite(cur_liab) and cur_liab != 0) else np.nan
                quick = ((cur_asset - inv) / cur_liab) if (np.isfinite(cur_asset) and np.isfinite(inv) and np.isfinite(cur_liab) and cur_liab != 0) else np.nan
                dy    = (div_ps / price * 100.0) if (np.isfinite(div_ps) and np.isfinite(price) and price != 0) else np.nan
                bvps  = (eq / sh) if (np.isfinite(eq) and np.isfinite(sh) and sh != 0) else np.nan
                inta_q = n(rawq.get("Q_IntangibleAsset"))
                nta   = ((eq - inta_q) / sh) if (np.isfinite(eq) and np.isfinite(inta_q) and np.isfinite(sh) and sh != 0) else np.nan

                st.markdown(
                    f"- **EPS** = Net Profit ÷ Shares = {npf:,.0f} ÷ {sh:,.0f} = **{eps:,.4f}**  \n"
                    f"- **ROE** = Net Profit ÷ Equity = {npf:,.0f} ÷ {eq:,.0f} = **{roe:,.2f}%**  \n"
                    f"- **P/E** = Price ÷ EPS = {price:,.4f} ÷ {eps:,.4f} = **{pe:,.2f}**  \n"
                    f"- **P/B** = (Price×Shares) ÷ Equity = {(mc if np.isfinite(mc) else np.nan):,.0f} ÷ {eq:,.0f} = **{pb:,.2f}**  \n"
                    f"- **Gross Margin** = Gross Profit ÷ Revenue = {gp:,.0f} ÷ {rev:,.0f} = **{gpm:,.2f}%**  \n"
                    f"- **Net Margin** = Net Profit ÷ Revenue = {npf:,.0f} ÷ {rev:,.0f} = **{npm:,.2f}%**  \n"
                    f"- **Current Ratio** = Current Assets ÷ Current Liabilities = {cur_asset:,.0f} ÷ {cur_liab:,.0f} = **{curr:,.2f}×**  \n"
                    f"- **Quick Ratio** ≈ (CA − Inventories) ÷ CL = "
                    f"{(cur_asset - inv) if np.isfinite(cur_asset) and np.isfinite(inv) else np.nan:,.0f} ÷ {cur_liab:,.0f} = **{quick:,.2f}×**  \n"
                    f"- **Dividend Yield** ≈ Dividend/share ÷ Price = {div_ps:,.4f} ÷ {price:,.4f} = **{dy:,.2f}%**  \n"
                    f"- **BVPS** = Equity ÷ Shares = {eq:,.0f} ÷ {sh:,.0f} = **{bvps:,.4f}**  \n"
                    + (f"- **NTA / share** = (Equity − Intangibles) ÷ Shares = {(eq - inta_q):,.0f} ÷ {sh:,.0f} = **{nta:,.4f}**"
                    if np.isfinite(inta_q) else "")
                )

                # --- Downloads for Quarterly Calculated Ratios ---
                # 2) All periods table
                all_payload = {
                    "stock":  stock_name,
                    "type":   "quarterly_ratios",
                    "ratios_by_period": _records(qratio_df.reset_index()),
                }
                _download_json_button(
                    "📥 Download All Quarterly Ratios (JSON)",
                    all_payload,
                    f"{stock_name}_quarterly_ratios.json",
                    key=f"dl_qr_all_{stock_name}"
                )

        

    # ---- Quarterly — Comparison Charts (under the ratios table)
    st.markdown('<div class="sec warning"><div class="t">📊 Quarterly — Comparison Charts</div></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Quarterly Comparison Charts (up to 4)")
    

    # A) Raw Quarterly comparisons
    q_opts = field_options(QUARTERLY_SECTIONS)
    period_labels = list(q_numeric.columns) if not q_numeric.empty else []
    def q_series_getter(sec_lbl):
        if q_numeric.empty:
            return None
        sec, lbl = sec_lbl
        if (sec, lbl) not in q_numeric.index:
            return None
        y = q_numeric.loc[(sec, lbl), :]
        return pd.Series(y).values

    q_raw_count = st.slider("Number of raw-data charts", 1, 4, 2, key=f"q_raw_chartcount_{stock_name}")
    multi_panel_charts(
        q_raw_count, q_opts, period_labels, q_series_getter,
        key_prefix=f"quarter_raw_chart_{stock_name}", chart_height=320
    )

    # B) Ratio comparisons
    if not qratio_df.empty:
        ratio_opts = [(m, m) for m in qratio_df.columns.tolist()]
        period_ratio = qratio_df.index.tolist()
        def q_ratio_series_getter(metric_name):
            if metric_name not in qratio_df.columns:
                return None
            y = qratio_df[metric_name]
            return pd.Series(y.values, index=period_ratio).values

        q_ratio_count = st.slider("Number of ratio charts", 1, 4, 2, key=f"q_ratio_chartcount_{stock_name}")
        multi_panel_charts(
            q_ratio_count, ratio_opts, period_ratio, q_ratio_series_getter,
            key_prefix=f"quarter_ratio_chart_{stock_name}", chart_height=320
        )



st.caption("Drag chips fixed: style args removed for streamlit-sortables to prevent React errors. Fallback drag list keeps pill styling. Charts & tables unchanged.")


