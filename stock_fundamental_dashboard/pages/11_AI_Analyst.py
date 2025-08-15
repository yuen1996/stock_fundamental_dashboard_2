# pages/11_AI_Analyst.py

from __future__ import annotations

# =========================
# Path setup (robust)
# =========================
import os, sys, re, json, math, time
from typing import Any, Dict, List, Tuple

_THIS = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))           # project root (parent of /pages)
_GRANDP = os.path.abspath(os.path.join(_ROOT, ".."))         # repo root
for p in (_ROOT, _GRANDP):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try both import layouts
try:
    from utils import io_helpers, calculations, rules, config  # type: ignore
except Exception:
    import io_helpers, calculations, rules                    # type: ignore
    try:
        import config                                         # type: ignore
    except Exception:
        class _DummyConfig:                                   # safe defaults
            FD_RATE = 0.035
        config = _DummyConfig()                               # type: ignore

# =========================
# Libs
# =========================
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config + CSS (match the rest of the app)
# =========================
st.set_page_config(page_title="AI Analyst", layout="wide")

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
.sec::before{ content:""; display:inline-block; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }

/* Chat bubbles */
.msg{ border:1px solid var(--border); border-radius:14px; background:#fff; padding:.75rem .9rem; box-shadow:var(--shadow); }
.msg.user{ border-left:5px solid var(--primary); }
.msg.ai{ border-left:5px solid #111827; }
.small{ color:var(--muted); font-size:.9rem; }
.kv{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:.85rem; }

/* Dataframes */
div[data-testid="stDataFrame"] {
  overflow-x:auto;
}
div[data-testid="stDataFrame"] table {
  border-collapse: collapse !important;
  width: max-content;
  min-width: 100%;
}
div[data-testid="stDataFrame"] table th,
div[data-testid="stDataFrame"] table td {
  border: 1px solid var(--border) !important;
}
div[data-testid="stDataFrame"] table thead th {
  position: sticky !important;
  top: 0 !important;
  background: #f9fafb !important;
  z-index: 10 !important;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("🤖 AI Analyst")

# =========================
# Utilities
# =========================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _qnum(q):
    if pd.isna(q): return np.nan
    s = str(q).upper().strip()
    m = re.search(r"(\d)", s)
    return float(m.group(1)) if m else np.nan

def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    df2 = df.copy()
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [" • ".join([str(x) for x in c]) for c in df2.columns.to_list()]
    if isinstance(df2.index, pd.MultiIndex):
        df2.index = [" • ".join([str(x) for x in tup]) for tup in df2.index]
    df2 = df2.replace({pd.NA: None, np.nan: None})
    return df2.to_dict("records")

def _download_json_button(label: str, obj, filename: str, key: str | None = None) -> None:
    try:
        payload = json.dumps(obj, default=str, indent=2)
    except Exception:
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

# Add canonical columns to annual so selections never KeyError
def _add_canonical_annual_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    alias_map = {
        "Revenue": ["Revenue","Sales","TotalRevenue","Total Revenue"],
        "GrossProfit": ["GrossProfit","Gross Profit"],
        "OperatingProfit": ["OperatingProfit","Operating Profit","Operating_Profit","EBIT","OperatingIncome","Operating Income"],
        "NetProfit": ["NetProfit","Net Profit","Profit","NetIncome","Net Income"],
        "EPS": ["EPS","Basic EPS","Diluted EPS","EPS (Basic)","EPS (Diluted)"],
        "NumShares": ["NumShares","Number of Shares","Number of shares","SharesOutstanding","ShareOutstanding","ShareCount","BasicShares"],
        "ShareholderEquity": ["ShareholderEquity","Shareholder Equity","TotalEquity","Total Equity","Equity"],
        "TotalDebt": ["TotalDebt","Debt","Borrowings"],
        "Cash": ["Cash","CashAndEquivalents","Cash & Equivalents","Cash & Cash Equivalents"],
        "Dividend": ["Dividend","Dividend pay cent","Dividend per share","DPS"],
        "SharePrice": ["SharePrice","End of year share price","Current Share Price","Each end of year share price"],
    }
    for canon, cands in alias_map.items():
        if canon not in df2.columns:
            for c in cands:
                if c in df2.columns:
                    df2[canon] = df2[c]
                    break
    return df2

def compute_momentum_bits(name: str) -> dict | None:
    """Read uploaded OHLC (Momentum page) and compute price, 200-DMA, 12m return."""
    try:
        oh = io_helpers.load_ohlc(name)
        if oh is None or oh.empty: return None
        df = oh.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df.get("Close"), errors="coerce")
        df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        if df.empty: return None
        price = float(df["Close"].iloc[-1])
        ma200 = float(df["Close"].rolling(200, min_periods=200).mean().iloc[-1]) if len(df) >= 200 else None
        ret12 = None
        if len(df) >= 252 and df["Close"].iloc[-252] not in (0, np.nan):
            base = float(df["Close"].iloc[-252])
            if base != 0:
                ret12 = float(price / base - 1.0)
        else:
            cutoff = df["Date"].iloc[-1] - pd.Timedelta(days=365)
            win = df[df["Date"] >= cutoff]
            if len(win) >= 2:
                base = float(win["Close"].iloc[0])
                if base != 0:
                    ret12 = float(price / base - 1.0)
        return {"price": price, "ma200": ma200, "ret_12m": ret12}
    except Exception:
        return None

def collect_bundle(name: str, full: bool = False) -> dict[str, Any]:
    df = st.session_state.get("_MASTER_DF")
    if df is None or df.empty:
        return {}
    s = df[df["Name"].astype(str) == str(name)].copy()
    if s.empty:
        return {}

    # helpers
    s["_Year"] = _to_num(s["Year"])
    s["_Q"] = s["Quarter"].apply(_qnum)
    s = s.sort_values(["_Year","_Q"])

    # metadata
    industry = ""
    bucket = "General"
    if "Industry" in s.columns and s["Industry"].notna().any():
        try: industry = str(s["Industry"].dropna().astype(str).iloc[-1])
        except Exception: pass
    if "IndustryBucket" in s.columns and s["IndustryBucket"].notna().any():
        try: bucket = str(s["IndustryBucket"].dropna().astype(str).iloc[-1]) or "General"
        except Exception: pass

    # annual
    annual = s[s["IsQuarter"] == False].copy()
    annual = _add_canonical_annual_cols(annual)
    latest_annual_dict = {}
    if not annual.empty:
        latest_annual_dict = annual.sort_values(["_Year"]).tail(1).iloc[0].to_dict()

    # TTM
    try:
        ttm = calculations.compute_ttm(s) or {}
    except Exception:
        ttm = {}

    # factor scores (uses price/ohlc if available)
    try:
        ohlc = compute_momentum_bits(name)
        factor = calculations.compute_factor_scores(
            stock_name=name, stock=s, ttm=ttm, ohlc_latest=ohlc, industry=None
        ) or {}
    except Exception:
        factor = {}

    # decision using rules (best effort)
    pass_fail = {"pass": None, "score": None, "reasons": []}
    try:
        metrics = {}
        metrics.update(ttm)
        for k in ["ROE (%)","P/E","Current Ratio","Debt-Asset Ratio (%)",
                  "Dividend Yield (%)","Dividend Payout Ratio (%)",
                  "Interest Coverage (EBITDA/Fin)"]:
            if k in s.columns:
                try:
                    metrics[k] = float(_to_num(s[k]).dropna().iloc[-1])
                except Exception:
                    pass
        res = rules.evaluate(metrics, "VQGM")  # type: ignore
        pass_fail = {
            "pass": res.get("pass"),
            "score": res.get("score"),
            "reasons": res.get("reasons", [])
        }
    except Exception:
        if isinstance(factor, dict) and factor:
            pass_fail["score"] = sum([int(factor.get(k,0) or 0)
                                     for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5

    bundle: Dict[str, Any] = {
        "name": name,
        "industry": industry,
        "bucket": bucket,
        "latest_annual": {
            k: latest_annual_dict.get(k)
            for k in [
                "Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS",
                "ShareholderEquity","NumShares","TotalDebt","Cash","SharePrice","CurrentPrice"
            ]
            if k in latest_annual_dict
        },
        "ttm": {k: ttm.get(k) for k in [
            "TTM Revenue","TTM Gross Profit","TTM Operating Profit","TTM Net Profit","TTM EBITDA",
            "TTM EPS","TTM Gross Margin (%)","TTM Net Margin (%)","P/E (TTM)","P/S (TTM)","P/B (TTM)",
            "EV/EBITDA (TTM)","FCF Yield (TTM) (%)","Cash Conversion (CFO/NP, %)","Debt / FCF (yrs)",
            "Interest Coverage (EBITDA/Fin)","MarketCap","Net Cash (Debt)"
        ] if k in ttm},
        "momentum": compute_momentum_bits(name),
        "scorecard": {
            "Value": int(factor.get("Value", 0) or 0),
            "Quality": int(factor.get("Quality", 0) or 0),
            "Growth": int(factor.get("Growth", 0) or 0),
            "Cash": int(factor.get("Cash", 0) or 0),
            "Momentum": int(factor.get("Momentum", 0) or 0),
            "Composite": int(round(sum([int(factor.get(k,0) or 0)
                              for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5)) if factor else None
        },
        "decision": pass_fail,
    }

    if full:
        # tail quarters
        q = s[s["IsQuarter"] == True].copy()
        if not q.empty:
            q["_Year"] = _to_num(q["Year"])
            q["_Q"] = q["Quarter"].apply(_qnum)
            q = q.dropna(subset=["_Year","_Q"]).sort_values(["_Year","_Q"]).tail(8)
            keep_q = ["Year","Quarter"] + [c for c in q.columns if str(c).startswith("Q_")]
            bundle["quarters_tail"] = (
                q[keep_q].replace({pd.NA: None, np.nan: None}).to_dict("records")
            )

        # annual rows (safe intersection)
        if not annual.empty:
            desired = ["Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS","NumShares",
                       "ShareholderEquity","TotalDebt","Cash","Dividend","SharePrice","CurrentPrice"]
            cols = [c for c in desired if c in annual.columns]
            bundle["annual_rows"] = (
                annual[cols]
                .replace({pd.NA: None, np.nan: None})
                .to_dict("records")
            )

    return bundle

def build_global_catalog(df: pd.DataFrame, hard_limit: int | None = None) -> list[dict]:
    """Small summaries per stock for the LLM context."""
    stocks = sorted([s for s in df["Name"].dropna().astype(str).unique().tolist()])
    if hard_limit is not None:
        stocks = stocks[:hard_limit]
    packs = []
    for n in stocks:
        packs.append(collect_bundle(n, full=False))
    return packs

# =========================
# Data load
# =========================
st.markdown('<div class="sec"><div class="t">📦 Data</div><div class="d">Loaded from your app storage</div></div>', unsafe_allow_html=True)

df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No master data found. Go to **Add/Edit** to import.")
    st.stop()

# Normalize a few essentials
if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

st.session_state["_MASTER_DF"] = df

stocks = sorted([s for s in df["Name"].dropna().astype(str).unique().tolist()])
st.caption(f"Loaded **{len(stocks)}** stock(s).")

# =========================
# OpenAI settings
# =========================
st.markdown('<div class="sec info"><div class="t">🔐 OpenAI</div><div class="d">Provide your API key to enable analysis</div></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([2,1,1])
with c1:
    api_key = st.text_input("OpenAI API Key", type="password", key="ai_api_key",
                            placeholder="sk-…", help="Key is kept only in your session.")
with c2:
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0, help="Use a lightweight model for speed; switch to gpt-4o for better reasoning.")
with c3:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

if api_key:
    st.success("API key set.")
else:
    st.info("Enter your API key to chat.")

# =========================
# Context controls
# =========================
st.markdown('<div class="sec success"><div class="t">🧠 Context</div><div class="d">What the AI can see</div></div>', unsafe_allow_html=True)
left, right = st.columns([2,1])
with left:
    default_q = "Which looks best now and why? Rank top-5 with key metrics."
    user_q = st.text_area("Ask the AI", value=default_q, height=90, help="Examples: 'Rank top-5', 'Tell me about HUP SENG', 'Build a thesis comparing A vs B'.")
with right:
    scan_all = st.checkbox("Include all stocks catalog", value=True, help="Send a compact bundle of every stock (names, TTM highlights, factor scores).")
    include_full_for_mentions = st.checkbox("Include full details for mentioned stocks", value=True, help="If you mention tickers/names in your question, the AI receives deeper rows for them.")

# Detect mentioned names (simple contains)
def _mentioned_names(text: str) -> list[str]:
    t = (text or "").lower()
    found = []
    for n in stocks:
        if n.lower() in t:
            found.append(n)
    return list(dict.fromkeys(found))  # unique, keep order

mentioned = _mentioned_names(user_q)

# Build payloads
catalog = build_global_catalog(df) if scan_all else []
packs_for_prompt = [collect_bundle(n, full=True) for n in mentioned] if include_full_for_mentions and mentioned else []

st.markdown("**Context preview**")
st.write(f"- Mentioned: {mentioned if mentioned else '—'}")
colp, cold = st.columns(2)
with colp:
    _download_json_button("📥 Download context — mentioned (JSON)", packs_for_prompt, "ai_context_mentioned.json", key="dl_ctx_m")
with cold:
    _download_json_button("📥 Download context — catalog (JSON)", catalog, "ai_context_catalog.json", key="dl_ctx_c")

# =========================
# Chat runtime
# =========================
if "chat" not in st.session_state:
    st.session_state["chat"] = []

def _push(role: str, content: str):
    st.session_state["chat"].append({"role": role, "content": content})

# system prompt keeps the AI grounded to your schema
SYSTEM = """You are an equity analyst working inside a user’s private dashboard.
Use ONLY the data provided in the context. If data is missing, say so clearly.
When ranking or recommending, give short bullet rationales referencing specific fields (e.g., “TTM Net Margin 12.4%”, “P/E (TTM) 9.2”, “Interest Coverage < 3”).
Prefer TTM figures for profitability and valuation; fall back to latest annual when TTM not available.
NEVER invent values that are not present in the context."""

def _render_chat():
    for m in st.session_state["chat"]:
        klass = "msg ai" if m["role"] == "assistant" else "msg user"
        st.markdown(f'<div class="{klass}">{m["content"]}</div>', unsafe_allow_html=True)

_render_chat()

def call_openai_chat(api_key: str, model: str, system: str, messages: List[Dict[str,str]], temperature: float = 0.2, max_tokens: int = 1200) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            messages=[{"role":"system","content":system}] + messages
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def build_prompt_messages(question: str, catalog: list[dict], packs: list[dict]) -> List[Dict[str,str]]:
    ctx_bits = []
    if catalog:
        ctx_bits.append({"type":"catalog", "stocks": catalog})
    if packs:
        ctx_bits.append({"type":"focus_stocks", "stocks": packs})
    context_str = json.dumps(ctx_bits, ensure_ascii=False)
    messages = [
        {
            "role": "user",
            "content": (
                "CONTEXT JSON BELOW.\n"
                "1) Read it carefully.\n"
                "2) Then answer the user question.\n"
                "3) Be concise and cite concrete fields (TTM/ratios) when you justify.\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"QUESTION: {question}"
            )
        }
    ]
    return messages

# =========================
# Action buttons
# =========================
st.markdown('<div class="sec warning"><div class="t">💬 Chat</div><div class="d">Ask your question and let the model rank or analyze</div></div>', unsafe_allow_html=True)
cA, cB, cC = st.columns([1,1,1])
with cA:
    ask = st.button("Ask AI", type="primary", use_container_width=True)
with cB:
    clear = st.button("Clear chat", use_container_width=True)
with cC:
    show_payload = st.toggle("Show raw payload in chat", value=False, help="Echo the JSON context back in the chat (useful for debugging).")

if clear:
    st.session_state["chat"] = []
    st.rerun()

if ask:
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
    else:
        # Push user message
        _push("user", user_q)

        # Build messages & call
        msgs = build_prompt_messages(user_q, catalog, packs_for_prompt)

        if show_payload:
            debug_blob = json.loads(msgs[0]["content"].split("CONTEXT:\n",1)[1].split("\n\nQUESTION:",1)[0])
            _push("assistant", f"<div class='small'><b>Context sent to model:</b></div>\n<pre class='kv'>{json.dumps(debug_blob, indent=2, ensure_ascii=False)}</pre>")

        answer = call_openai_chat(api_key, model, SYSTEM, msgs, temperature=temperature)
        _push("assistant", answer)

        st.experimental_rerun()

# =========================
# Quick helpers: best-5 (local)
# =========================
st.markdown('<div class="sec danger"><div class="t">⚙️ Quick Local Ranking (no AI)</div><div class="d">Use your computed scores to preview</div></div>', unsafe_allow_html=True)

def _quick_rank_top5() -> pd.DataFrame:
    rows = []
    for n in stocks:
        b = collect_bundle(n, full=False)
        if not b: continue
        sc = b.get("scorecard", {})
        ttm = b.get("ttm", {})
        rows.append({
            "Name": n,
            "Score": sc.get("Composite"),
            "TTM Revenue": ttm.get("TTM Revenue"),
            "TTM Net Profit": ttm.get("TTM Net Profit"),
            "TTM EPS": ttm.get("TTM EPS"),
            "TTM Net Margin (%)": ttm.get("TTM Net Margin (%)"),
            "P/E (TTM)": ttm.get("P/E (TTM)"),
            "Interest Coverage": ttm.get("Interest Coverage (EBITDA/Fin)"),
            "Pass?": (b.get("decision", {}) or {}).get("pass")
        })
    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Simple ordering: pass first, higher Score next, then Net Margin
    out["_pass"] = out["Pass?"].fillna(False).astype(int)
    out = out.sort_values(["_pass","Score","TTM Net Margin (%)"], ascending=[False, False, False]).head(5)
    out = out.drop(columns=["_pass"])
    return out

preview = _quick_rank_top5()
if preview.empty:
    st.info("No ranking available yet.")
else:
    st.dataframe(preview, use_container_width=True, height=230)
    _download_json_button("📥 Download Quick Top-5 (JSON)", _records(preview), "ai_quick_top5.json", key="dl_quick5")
