# pages/11_AI_Analyst.py
# AI Analyst — grounded on YOUR data (annual, quarterly, ratios, TTM, factor scores)
# Works with OpenAI (if key provided) and has a local fallback when no key is set.

from auth_gate import require_auth
require_auth()

# ---- Make project-root modules importable (robust like other pages) ----
import os, sys, re, json, math, itertools
from typing import Any, Dict, List, Tuple

_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
_GRANDP = os.path.abspath(os.path.join(_THIS, "..", ".."))
for p in (_PARENT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# Prefer local package, fallback to utils
try:
    import io_helpers, calculations, rules, config  # type: ignore
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules, config  # type: ignore

import streamlit as st
import pandas as pd
import numpy as np

# =========================
# --------- UI ------------
# =========================
st.set_page_config(layout="wide")
st.title("🤖 AI Analyst")

# Minimal CSS to match the app’s look-and-feel
st.markdown("""
<style>
:root{
  --border:#e5e7eb; --surface:#ffffff; --hover:#f8fafc; --shadow:0 8px 24px rgba(15,23,42,.06);
}
.wrap{border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); background:#fff; padding:10px;}
</style>
""", unsafe_allow_html=True)

# =========================
# ---- Load the dataset ---
# =========================
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please upload data on the Add/Edit page first.")
    st.stop()

# Ensure columns exist (compat)
if "IsQuarter" not in df.columns: df["IsQuarter"] = False
if "Quarter" not in df.columns: df["Quarter"] = pd.NA

# =========================
# --- Helpers / builders --
# =========================
def _to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    try:
        s = str(x).replace(",", "").strip()
        return float(s) if s != "" else np.nan
    except Exception:
        return np.nan

def quarter_key_to_num(q):
    if pd.isna(q): return np.nan
    s = str(q).upper().strip()
    m = re.search(r"(\d+)", s)
    if not m: return np.nan
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 4 else np.nan
    except Exception:
        return np.nan

ANNUAL_SECTIONS = [
    ("Income Statement", [
        ("Net Profit", "NetProfit"),
        ("Gross Profit", "GrossProfit"),
        ("Revenue", "Revenue"),
        ("Cost Of Sales", "CostOfSales"),
        ("Finance Costs", "FinanceCosts"),
        ("Administrative Expenses", "AdminExpenses"),
        ("Selling & Distribution Expenses", "SellDistExpenses"),
    ]),
    ("Balance Sheet", [
        ("Number of Shares", "NumShares"),
        ("Current Asset", "CurrentAsset"),
        ("Other Receivables", "OtherReceivables"),
        ("Trade Receivables", "TradeReceivables"),
        ("Biological Assets", "BiologicalAssets"),
        ("Inventories", "Inventories"),
        ("Prepaid Expenses", "PrepaidExpenses"),
        ("Intangible Asset", "IntangibleAsset"),
        ("Current Liability", "CurrentLiability"),
        ("Total Asset", "TotalAsset"),
        ("Total Liability", "TotalLiability"),
        ("Shareholder Equity", "ShareholderEquity"),
        ("Reserves", "Reserves"),
        ("Cash & Cash Equivalents", "Cash"),
        ("Total Debt / Borrowings", "TotalDebt"),
    ]),
    ("Other Data", [
        ("Dividend pay cent", "Dividend"),
        ("End of year share price", "SharePrice"),
    ]),
    ("Cash Flow Statement", [
        ("Operating Cash Flow", "CFO"),
        ("Capital Expenditure", "CapEx"),
        ("Income Tax Expense", "IncomeTax"),
        ("Depreciation & Amortization", "DepAmort"),
        ("EBITDA (optional)", "EBITDA"),
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
        ("Current Asset", "Q_CurrentAsset"),
        ("Other Receivables", "Q_OtherReceivables"),
        ("Trade Receivables", "Q_TradeReceivables"),
        ("Biological Assets", "Q_BiologicalAssets"),
        ("Inventories", "Q_Inventories"),
        ("Prepaid Expenses", "Q_PrepaidExpenses"),
        ("Intangible Asset", "Q_IntangibleAsset"),
        ("Current Liability", "Q_CurrentLiability"),
        ("Total Asset", "Q_TotalAsset"),
        ("Total Liability", "Q_TotalLiability"),
        ("Shareholder Equity", "Q_ShareholderEquity"),
        ("Reserves", "Q_Reserves"),
    ]),
    ("Quarterly Cash Flow Statement", [
        ("Quarterly Operating Cash Flow", "Q_CFO"),
        ("Quarterly Capital Expenditure", "Q_CapEx"),
        ("Quarterly Income Tax", "Q_Tax"),
        ("Quarterly Depreciation & Amortization", "Q_DepAmort"),
        ("Quarterly EBITDA (optional)", "Q_EBITDA"),
    ]),
    ("Quarterly Other Data", [
        ("Each end per every quarter price", "Q_EndQuarterPrice"),
    ]),
]

def build_annual_raw_numeric(annual_df: pd.DataFrame) -> pd.DataFrame:
    if annual_df is None or annual_df.empty:
        return pd.DataFrame()
    years = sorted([int(y) for y in pd.to_numeric(annual_df["Year"], errors="coerce").dropna().unique()])
    rows = []
    for sec, items in ANNUAL_SECTIONS:
        for label, key in items:
            rows.append((sec, label, key))
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Section", "Field"])
    out = pd.DataFrame(index=idx, columns=[str(y) for y in years], dtype=float)
    ann_coerced = annual_df.copy()
    for col in annual_df.columns:
        if col == "Year": continue
        ann_coerced[col] = pd.to_numeric(annual_df[col], errors="coerce")
    ann_by_year = {int(r["Year"]): r for _, r in ann_coerced.iterrows() if pd.notna(r.get("Year"))}
    for (sec, label, key), (i_sec, i_field) in zip(rows, out.index):
        for y in years:
            val = np.nan
            row = ann_by_year.get(y)
            if row is not None and key in row:
                val = _to_float(row[key])
            out.loc[(i_sec, i_field), str(y)] = val
    return out

def build_quarter_raw_numeric(quarter_df: pd.DataFrame) -> pd.DataFrame:
    if quarter_df is None or quarter_df.empty:
        return pd.DataFrame()
    q = quarter_df.copy()
    q["Qnum"] = q["Quarter"].map(quarter_key_to_num)
    q = q.dropna(subset=["Year", "Qnum"])
    q["Year"] = pd.to_numeric(q["Year"], errors="coerce")
    q = q.dropna(subset=["Year"])
    q["Year"] = q["Year"].astype(int)
    q = q.sort_values(["Year", "Qnum"])
    periods = [f"{int(r['Year'])} Q{int(r['Qnum'])}" for _, r in q.iterrows()]
    seen, cols, row_by_period = set(), [], {}
    for period, (_, r) in zip(periods, q.iterrows()):
        if period in seen: continue
        seen.add(period); cols.append(period); row_by_period[period] = r
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

def _spike_flags(series: pd.Series, alert_pct=1.0, materiality_ratio=0.05) -> List[Tuple[int, float, float, float]]:
    """
    Return list of spike tuples: (idx, prev, cur, pct_change)
    A spike = (cur-prev)/max(|prev|, 5%*median_abs(series)) >= alert_pct
    """
    vals = pd.to_numeric(series, errors="coerce").astype(float).values
    if len(vals) < 2:
        return []
    med = np.nanmedian(np.abs(vals))
    floor = max(1e-12, materiality_ratio * (0 if np.isnan(med) else med))
    spikes = []
    for i in range(1, len(vals)):
        prev, cur = vals[i-1], vals[i]
        if np.isnan(prev) or np.isnan(cur): 
            continue
        base = max(abs(prev), floor)
        pct = (cur - prev) / base
        if np.isfinite(pct) and pct >= alert_pct and abs(cur - prev) >= floor:
            spikes.append((i, prev, cur, pct))
    return spikes

def spike_report_annual(ann_numeric: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    if ann_numeric is None or ann_numeric.empty:
        return out
    years = list(ann_numeric.columns)
    for (sec, field) in ann_numeric.index:
        s = ann_numeric.loc[(sec, field)]
        hits = _spike_flags(s)
        for idx, prev, cur, pct in hits:
            out.append({
                "where": "annual", "section": sec, "field": field,
                "from_year": years[idx-1], "to_year": years[idx],
                "prev": float(prev), "cur": float(cur), "pct_change": float(pct)
            })
    return out

def spike_report_quarterly(q_numeric: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    if q_numeric is None or q_numeric.empty:
        return out
    periods = list(q_numeric.columns)
    for (sec, field) in q_numeric.index:
        s = q_numeric.loc[(sec, field)]
        hits = _spike_flags(s)
        for idx, prev, cur, pct in hits:
            out.append({
                "where": "quarterly", "section": sec, "field": field,
                "from_period": periods[idx-1], "to_period": periods[idx],
                "prev": float(prev), "cur": float(cur), "pct_change": float(pct)
            })
    return out

def _last_non_null(serieslike):
    s = pd.to_numeric(serieslike, errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def _industry_for(stock_df: pd.DataFrame) -> Tuple[str, str]:
    industry = ""; bucket = "General"
    if "Industry" in stock_df.columns and stock_df["Industry"].notna().any():
        industry = str(stock_df["Industry"].dropna().astype(str).iloc[-1])
    if "IndustryBucket" in stock_df.columns and stock_df["IndustryBucket"].notna().any():
        bucket = str(stock_df["IndustryBucket"].dropna().astype(str).iloc[-1])
    return industry, bucket

def _current_price(stock_df: pd.DataFrame) -> float:
    cur_val = np.nan
    if "CurrentPrice" in stock_df.columns:
        s = pd.to_numeric(stock_df["CurrentPrice"], errors="coerce").dropna()
        if not s.empty:
            cur_val = float(s.iloc[-1])
    return cur_val

def _compact_annual_ratios(annual_df: pd.DataFrame, cur_price: float) -> pd.DataFrame:
    rows = []
    for _, row in annual_df.iterrows():
        r = row.copy()
        # force latest price for valuation metrics
        try:
            if np.isfinite(cur_price):
                r["SharePrice"] = float(cur_price)
        except Exception:
            pass
        rs = calculations.calc_ratios(r)
        rs["Year"] = row.get("Year")
        rows.append(rs)
    out = pd.DataFrame(rows)
    if not out.empty and "Year" in out.columns:
        out = out.set_index("Year").sort_index()
    return out

@st.cache_data(show_spinner=False)
def build_stock_context(stock_name: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a rich but compact description of one stock:
      • latest price
      • TTM block (from your calculations.compute_ttm)
      • factor scores (Value/Quality/Growth/Cash/Momentum)
      • last N years (ratios) and last M quarters (ratios)
      • spike anomalies from annual & quarterly raw matrices
    """
    stock = data[data["Name"] == stock_name].sort_values("Year")
    annual = stock[stock["IsQuarter"] != True].copy()
    quarterly = stock[stock["IsQuarter"] == True].copy()

    industry, bucket = _industry_for(stock)
    cur_val = _current_price(stock)

    # --- TTM (your function)
    try:
        ttm = calculations.compute_ttm(stock, current_price=cur_val)
    except Exception:
        ttm = {}

    # --- Factor scores (your function)
    try:
        scores = calculations.compute_factor_scores(
            stock_name, stock, ttm, ohlc_latest=None, industry=industry or None
        )
    except Exception:
        scores = {}

    # --- Ratios (annual) — compact
    ratio_df = _compact_annual_ratios(annual, cur_val)
    # Keep only a handful of core columns if huge
    core_cols = ["EPS","ROE (%)","P/E","P/B","Gross Profit Margin (%)","Net Profit Margin (%)",
                 "Current Ratio","Quick Ratio","Dividend Yield (%)","BVPS","NTA per share"]
    keep_cols = [c for c in core_cols if c in ratio_df.columns]
    ratio_compact = ratio_df[keep_cols].tail(8) if not ratio_df.empty else pd.DataFrame()

    # --- Build raw matrices & spike reports
    ann_num = build_annual_raw_numeric(annual)
    q_num   = build_quarter_raw_numeric(quarterly)
    ann_spikes = spike_report_annual(ann_num) if not ann_num.empty else []
    q_spikes   = spike_report_quarterly(q_num) if not q_num.empty else []

    # --- JSON-friendly
    def _records_df(df):
        if df is None or df.empty: return []
        # ensure numeric -> floats
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").round(4)
        df2.index = [str(i) for i in df2.index]
        return df2.reset_index().to_dict("records")

    context = {
        "name": stock_name,
        "industry": industry,
        "industry_bucket": bucket,
        "current_price": (float(cur_val) if np.isfinite(cur_val) else None),
        "ttm": {k: (None if v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v))) else float(v))
                for k, v in (ttm or {}).items()
                if not isinstance(v, (dict, list))},  # keep simple numeric ttm fields
        "factor_scores": {k: int(scores.get(k, 0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]},
        "factor_detail_confidence": scores.get("_confidence", {}),
        "annual_ratios_recent": _records_df(ratio_compact),
        "data_spikes": {
            "annual": ann_spikes[:20],      # cap to keep tokens reasonable
            "quarterly": q_spikes[:20],
        }
    }
    return context

@st.cache_data(show_spinner=False)
def list_stock_names(data: pd.DataFrame) -> List[str]:
    return sorted(data["Name"].dropna().astype(str).unique().tolist())

ALL_NAMES = list_stock_names(df)

def _match_names_in_text(q: str, names: List[str]) -> List[str]:
    ql = q.lower()
    hits = []
    for n in names:
        if n and str(n).lower() in ql:
            hits.append(n)
    return list(dict.fromkeys(hits))  # preserve order / unique

def _rank_stocks_locally(data: pd.DataFrame, names: List[str], topk: int = 5) -> List[str]:
    """
    Heuristic ranking using your calc functions.
    Composite = mean(Value,Quality,Growth,Cash,Momentum). If tie, prefer lower P/E (TTM).
    """
    scored = []
    for nm in names:
        stock = data[data["Name"] == nm]
        cur_val = _current_price(stock)
        try:
            ttm = calculations.compute_ttm(stock, current_price=cur_val) or {}
        except Exception:
            ttm = {}
        try:
            s = calculations.compute_factor_scores(nm, stock, ttm, ohlc_latest=None, industry=None) or {}
        except Exception:
            s = {}
        comp = np.mean([int(s.get(k,0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]]) if s else 0
        pe = ttm.get("P/E (TTM)")
        pe = float(pe) if (pe is not None and isinstance(pe,(int,float)) and math.isfinite(pe)) else 1e9
        scored.append((nm, comp, pe))
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [s[0] for s in scored[:topk]]

def _choose_relevant_stocks(question: str, data: pd.DataFrame, max_stocks: int) -> List[str]:
    mentioned = _match_names_in_text(question or "", ALL_NAMES)
    if mentioned:
        base = mentioned
    else:
        # use entire universe for ranking
        base = ALL_NAMES
    return _rank_stocks_locally(data, base, topk=max_stocks)

def _build_context_pack(names: List[str], data: pd.DataFrame, detail: str = "compact") -> Dict[str, Any]:
    """
    Build the JSON that will be shown to the model (or the fallback).
    detail: compact | detailed
    Compact keeps recent ratios + spikes; Detailed also adds last 8 quarterly ratios & extra TTM fields.
    """
    items = []
    for nm in names:
        ctx = build_stock_context(nm, data)
        if detail == "detailed":
            # add a few more TTM items if present
            extra = ["TTM Revenue", "TTM Net Profit", "TTM EBITDA", "TTM CFO", "TTM CapEx",
                     "Cash Conversion (CFO/NP, %)", "FCF Yield (TTM) (%)", "Debt / FCF (yrs)"]
            extras = {}
            for k in extra:
                v = ctx["ttm"].get(k) if isinstance(ctx.get("ttm"), dict) else None
                if isinstance(v, (int,float)) and math.isfinite(float(v)):
                    extras[k] = float(v)
            if extras:
                ctx["ttm_extra"] = extras
        items.append(ctx)
    return {"universe_size": len(ALL_NAMES), "selected": names, "stocks": items}

# =========================
# ---- OpenAI plumbing ----
# =========================
def _get_openai_client(api_key: str):
    try:
        from openai import OpenAI  # requires openai>=1.0
    except Exception as e:
        raise RuntimeError("The 'openai' package is not installed. Add 'openai>=1.35.0' to requirements.txt") from e
    return OpenAI(api_key=api_key)

SYS_PROMPT = """You are an equity research assistant grounded strictly on the provided JSON data.
Tasks you can perform:
- Evaluate which stock(s) look best now (Value, Quality, Growth, Cash, Momentum) using the data.
- Explain your reasoning with numbers (TTM, ratios, trends).
- Audit the data: call out anomalies or potential key-in errors (we provide spike reports).
Rules:
- Do NOT invent data—only use the JSON you were given.
- If data looks wrong, say why and point to the exact fields/periods.
- Prefer concise bullets and include equations when useful.
- If multiple choices tie, state trade-offs and list top 3 with reasons.
"""

def _ask_openai(api_key: str, model: str, temperature: float, user_q: str, context_json: Dict[str, Any]) -> str:
    client = _get_openai_client(api_key)
    # Keep payload human-readable but bounded
    context_str = json.dumps(context_json, indent=2, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question:\n{user_q}\n\nData:\n{context_str}"}
    ]
    try:
        # Chat Completions keeps compatibility; you can switch to Responses if you prefer
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=1,
            max_tokens=1400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# =========================
# ------- Sidebar ---------
# =========================
with st.sidebar:
    st.header("⚙️ AI Settings")
    default_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", type="password", value=default_key, help="Stored only in your session.")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.2, 0.1)
    detail = st.radio("Context detail", ["compact", "detailed"], horizontal=True, index=0)
    max_stocks = st.slider("Max stocks to analyze per question", 1, 12, 6)
    offline = st.toggle("Offline fallback (no API)", value=(api_key.strip() == ""))

    st.caption("Tip: mention tickers/names in your question to force-include them.")

# =========================
# ------- Chat UI ---------
# =========================
if "ai_chat" not in st.session_state:
    st.session_state.ai_chat = []
if "last_context" not in st.session_state:
    st.session_state.last_context = {}

for m in st.session_state.ai_chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about your universe (e.g., “Which looks best now?” or “Audit for data errors”).")
if q:
    st.session_state.ai_chat.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Building context…"):
            chosen = _choose_relevant_stocks(q, df, max_stocks=max_stocks)
            ctx = _build_context_pack(chosen, df, detail=detail)
            st.session_state.last_context = ctx

        # Local heuristic fallback
        def _local_answer() -> str:
            lines = []
            lines.append(f"**Local (no-API) analyst — {len(ctx['stocks'])} stock(s) considered**")
            # Rank again with composite and include quick reasons
            ranked = _rank_stocks_locally(df, chosen, topk=min(3, len(chosen)))
            for i, nm in enumerate(ranked, 1):
                item = next((x for x in ctx["stocks"] if x["name"] == nm), None)
                if not item: 
                    continue
                fs  = item.get("factor_scores", {})
                ttm = item.get("ttm", {})
                gpm = ttm.get("TTM Gross Margin (%)")
                npm = ttm.get("TTM Net Margin (%)")
                pe  = ttm.get("P/E (TTM)")
                fy  = ttm.get("FCF Yield (TTM) (%)")
                lines.append(f"{i}. **{nm}** — Scores: V{fs.get('Value',0)}/Q{fs.get('Quality',0)}/G{fs.get('Growth',0)}/C{fs.get('Cash',0)}/M{fs.get('Momentum',0)}"
                             + (f" · P/E {pe:.2f}" if isinstance(pe,(int,float)) and math.isfinite(pe) else "")
                             + (f" · GM {gpm:.1f}%" if isinstance(gpm,(int,float)) and math.isfinite(gpm) else "")
                             + (f" · NM {npm:.1f}%" if isinstance(npm,(int,float)) and math.isfinite(npm) else "")
                             + (f" · FCFY {fy:.1f}%" if isinstance(fy,(int,float)) and math.isfinite(fy) else ""))
                # data anomalies summary
                spikes = (item.get("data_spikes") or {})
                ann_cnt = len(spikes.get("annual", []))
                q_cnt   = len(spikes.get("quarterly", []))
                if ann_cnt or q_cnt:
                    lines.append(f"   ↳ ⚠️ Data flags: {ann_cnt} annual spikes, {q_cnt} quarterly spikes (≥100% jump & ≥5% materiality).")
            if not ranked:
                lines.append("No stocks found in the question. Try mentioning a name.")
            lines.append("\n_Provide an API key in the sidebar to get a deeper LLM rationale._")
            return "\n".join(lines)

        if offline or (not api_key.strip()):
            ans = _local_answer()
        else:
            with st.spinner("Asking OpenAI…"):
                ans = _ask_openai(api_key.strip(), model, temperature, q, ctx)

        st.markdown(ans)
        st.session_state.ai_chat.append({"role":"assistant","content":ans})

# =========================
# ------ Tools row --------
# =========================
st.divider()
c1, c2, c3 = st.columns([1.2,1,1])
with c1:
    if st.button("⬇️ Download last LLM context (JSON)", use_container_width=True, type="secondary",
                 disabled=(not st.session_state.get("last_context"))):
        st.download_button(
            "Download context.json",
            data=json.dumps(st.session_state.get("last_context", {}), indent=2),
            file_name="ai_context.json",
            mime="application/json",
            use_container_width=True,
            key="dl_ctx_btn_inner"
        )
with c2:
    if st.button("🧹 Reset chat", use_container_width=True, type="secondary"):
        st.session_state.ai_chat = []
        st.rerun()
with c3:
    show_audit = st.toggle("Show data anomaly audit (quick scan)", value=False)

if show_audit:
    st.subheader("📋 Quick anomaly audit (programmatic)")
    st.caption("Rule: highlight ≥100% change vs previous period AND absolute change ≥5% of the series median.")
    names = st.multiselect("Stocks to scan", ALL_NAMES, default=ALL_NAMES[:min(8,len(ALL_NAMES))])
    for nm in names:
        st.markdown(f"**{nm}**")
        stock = df[df["Name"] == nm]
        annual = stock[stock["IsQuarter"] != True]
        quarterly = stock[stock["IsQuarter"] == True]
        ann_num = build_annual_raw_numeric(annual)
        q_num   = build_quarter_raw_numeric(quarterly)
        ann_sp = spike_report_annual(ann_num)
        q_sp   = spike_report_quarterly(q_num)
        if not ann_sp and not q_sp:
            st.success("No spikes detected.")
        else:
            if ann_sp:
                st.markdown("**Annual spikes**")
                ast = pd.DataFrame(ann_sp)
                st.dataframe(ast, use_container_width=True, height=min(260, 48+28*len(ast)))
            if q_sp:
                st.markdown("**Quarterly spikes**")
                qst = pd.DataFrame(q_sp)
                st.dataframe(qst, use_container_width=True, height=min(260, 48+28*len(q_sp)))
