# pages/11_AI_Analyst.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

# =========================
# Path setup (robust)
# =========================
import os, sys, re, json
from typing import Any, Dict, List, Tuple

_THIS = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))           # project root (parent of /pages)
_GRANDP = os.path.abspath(os.path.join(_ROOT, ".."))         # repo root
for p in (_ROOT, _GRANDP, os.path.join(_ROOT, "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------
# Super-robust imports
# -------------------------
import importlib

def _try_import(name_variants: List[str]):
    last_err = None
    for modname in name_variants:
        try:
            return importlib.import_module(modname)
        except Exception as e:
            last_err = e
    raise last_err if last_err else ModuleNotFoundError(", ".join(name_variants))

PKG = (__package__.split(".")[0] if __package__ else None)

# candidates for each helper
_io_candidates = ["io_helpers", "utils.io_helpers"]
_calc_candidates = ["calculations", "utils.calculations"]
_rules_candidates = ["rules", "utils.rules"]
_conf_candidates = ["config", "utils.config"]

if PKG:
    _io_candidates[:0] = [f"{PKG}.io_helpers", f"{PKG}.utils.io_helpers"]
    _calc_candidates[:0] = [f"{PKG}.calculations", f"{PKG}.utils.calculations"]
    _rules_candidates[:0] = [f"{PKG}.rules", f"{PKG}.utils.rules"]
    _conf_candidates[:0] = [f"{PKG}.config", f"{PKG}.utils.config"]

try:
    io_helpers = _try_import(_io_candidates)
    calculations = _try_import(_calc_candidates)
    rules = _try_import(_rules_candidates)
    try:
        config = _try_import(_conf_candidates)
    except Exception:
        class _DummyConfig:
            FD_RATE = 0.035
        config = _DummyConfig()
except Exception as e:
    import streamlit as st
    st.error(
        "Could not import helper modules. Make sure your project has either:\n\n"
        "• helpers in the project root (io_helpers.py, calculations.py, rules.py, config.py)\n"
        "• OR in a 'utils/' folder (utils/io_helpers.py, ...)\n\n"
        f"Importer error: {e}"
    )
    st.stop()

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
  --bg:#f6f7fb;
  --surface:#ffffff;
  --text:#0f172a;
  --muted:#475569;
  --border:#e5e7eb;
  --shadow:0 8px 24px rgba(15, 23, 42, .06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"]{ font-size:16px !important; color:var(--text); }
.stApp{ background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg); }
h1, h2, h3, h4{ color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px; }
.sec{
  background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:var(--shadow);
  padding:.65rem .9rem; margin:1rem 0 .6rem 0; display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; display:inline-block; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before    { background:var(--info); }
.sec.success::before { background:var(--success); }
.sec.warning::before { background:var(--warning); }
.sec.danger::before  { background:var(--danger); }
.msg{ border:1px solid var(--border); border-radius:14px; background:#fff; padding:.75rem .9rem; box-shadow:var(--shadow); }
.msg.user{ border-left:5px solid var(--primary); }
.msg.ai{ border-left:5px solid #111827; }
.small{ color:var(--muted); font-size:.9rem; }
.kv{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:.85rem; }
div[data-testid="stDataFrame"] { overflow-x:auto; }
div[data-testid="stDataFrame"] table { border-collapse: collapse !important; width: max-content; min-width: 100%; }
div[data-testid="stDataFrame"] table th, div[data-testid="stDataFrame"] table td { border: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] table thead th { position: sticky !important; top: 0 !important; background: #f9fafb !important; z-index: 10 !important; }
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

# Canonicalize annual columns so selections never KeyError
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
        "CurrentPrice": ["CurrentPrice","Current Price","Price"],
    }
    for canon, cands in alias_map.items():
        if canon not in df2.columns:
            for c in cands:
                if c in df2.columns:
                    df2[canon] = df2[c]
                    break
    return df2

def compute_momentum_bits(name: str) -> dict | None:
    """Read uploaded OHLC and compute price, 200-DMA, 12m return."""
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

def _safe_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _is_missing(v) -> bool:
    try:
        if v is None: return True
        if isinstance(v, float) and np.isnan(v): return True
        return False
    except Exception:
        return True

def collect_bundle(name: str, full: bool = False) -> dict[str, Any]:
    df = st.session_state.get("_MASTER_DF")
    if df is None or df.empty:
        return {}
    s = df[df["Name"].astype(str) == str(name)].copy()
    if s.empty:
        return {}

    s["_Year"] = pd.to_numeric(s["Year"], errors="coerce")
    s["_Q"] = s["Quarter"].apply(lambda q: float(re.search(r"(\d)", str(q).upper().strip()).group(1)) if pd.notna(q) and re.search(r"(\d)", str(q).upper().strip()) else np.nan)
    s = s.sort_values(["_Year","_Q"])

    industry = ""
    bucket = "General"
    if "Industry" in s.columns and s["Industry"].notna().any():
        try: industry = str(s["Industry"].dropna().astype(str).iloc[-1])
        except Exception: pass
    if "IndustryBucket" in s.columns and s["IndustryBucket"].notna().any():
        try: bucket = (str(s["IndustryBucket"].dropna().astype(str).iloc[-1]) or "General")
        except Exception: pass

    annual = s[s["IsQuarter"] == False].copy()
    annual = _add_canonical_annual_cols(annual)
    latest_annual_dict = {}
    if not annual.empty:
        latest_annual_dict = annual.sort_values(["_Year"]).tail(1).iloc[0].to_dict()

    # Momentum / last price
    ohlc = compute_momentum_bits(name)

    # If CurrentPrice missing in annual, fill from OHLC price
    if ohlc and (_is_missing(latest_annual_dict.get("CurrentPrice"))):
        latest_annual_dict["CurrentPrice"] = ohlc.get("price")

    # Compute TTM from your calculations
    try:
        ttm = calculations.compute_ttm(s) or {}
    except Exception:
        ttm = {}

    # Derive a few common valuation ratios if inputs exist (price, shares, equity, EBITDA, rev, eps)
    try:
        latest_price = _safe_float((ohlc or {}).get("price")) \
                       or _safe_float(latest_annual_dict.get("CurrentPrice")) \
                       or _safe_float(latest_annual_dict.get("SharePrice"))
        shares = _safe_float(latest_annual_dict.get("NumShares"))
        equity = _safe_float(latest_annual_dict.get("ShareholderEquity"))
        debt = _safe_float(latest_annual_dict.get("TotalDebt"))
        cash = _safe_float(latest_annual_dict.get("Cash"))
        eps_ttm = _safe_float(ttm.get("TTM EPS"))
        rev_ttm = _safe_float(ttm.get("TTM Revenue"))
        ebitda_ttm = _safe_float(ttm.get("TTM EBITDA"))

        if latest_price is not None:
            ttm["Current Price"] = latest_price

        marketcap = None
        if latest_price and shares:
            marketcap = latest_price * shares
            ttm["MarketCap"] = marketcap

        if latest_price and eps_ttm and eps_ttm > 0:
            ttm["P/E (TTM)"] = latest_price / eps_ttm
        if marketcap and rev_ttm and rev_ttm > 0:
            ttm["P/S (TTM)"] = marketcap / rev_ttm
        if marketcap and equity and equity > 0:
            ttm["P/B (TTM)"] = marketcap / equity
        if marketcap and ebitda_ttm and ebitda_ttm > 0:
            ev = marketcap + (debt or 0.0) - (cash or 0.0)
            ttm["EV/EBITDA (TTM)"] = ev / ebitda_ttm
    except Exception:
        pass

    # Factor scores (your existing method)
    try:
        factor = calculations.compute_factor_scores(
            stock_name=name, stock=s, ttm=ttm, ohlc_latest=ohlc, industry=None
        ) or {}
    except Exception:
        factor = {}

    # Pass/fail via rules
    pass_fail = {"pass": None, "score": None, "reasons": []}
    try:
        metrics = {}
        metrics.update(ttm)
        for k in ["ROE (%)","P/E","Current Ratio","Debt-Asset Ratio (%)",
                  "Dividend Yield (%)","Dividend Payout Ratio (%)",
                  "Interest Coverage (EBITDA/Fin)"]:
            if k in s.columns:
                try:
                    metrics[k] = float(pd.to_numeric(s[k], errors="coerce").dropna().iloc[-1])
                except Exception:
                    pass
        res = rules.evaluate(metrics, "VQGM")  # type: ignore
        pass_fail = {"pass": res.get("pass"), "score": res.get("score"), "reasons": res.get("reasons", [])}
    except Exception:
        if isinstance(factor, dict) and factor:
            pass_fail["score"] = sum([int(factor.get(k,0) or 0)
                                     for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5

    latest_annual_out = {
        k: latest_annual_dict.get(k)
        for k in [
            "Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS",
            "ShareholderEquity","NumShares","TotalDebt","Cash","SharePrice","CurrentPrice"
        ]
        if k in latest_annual_dict
    }
    # Ensure CurrentPrice shows up even if not originally in the CSV
    if ohlc and _is_missing(latest_annual_out.get("CurrentPrice")):
        latest_annual_out["CurrentPrice"] = ohlc.get("price")

    bundle: Dict[str, Any] = {
        "name": name,
        "industry": industry,
        "bucket": bucket,
        "latest_annual": latest_annual_out,
        "ttm": {k: ttm.get(k) for k in [
            "TTM Revenue","TTM Gross Profit","TTM Operating Profit","TTM Net Profit","TTM EBITDA",
            "TTM EPS","TTM Gross Margin (%)","TTM Net Margin (%)","P/E (TTM)","P/S (TTM)","P/B (TTM)",
            "EV/EBITDA (TTM)","FCF Yield (TTM) (%)","Cash Conversion (CFO/NP, %)","Debt / FCF (yrs)",
            "Interest Coverage (EBITDA/Fin)","MarketCap","Current Price","Net Cash (Debt)"
        ] if k in ttm},
        "momentum": ohlc,
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
        q = s[s["IsQuarter"] == True].copy()
        if not q.empty:
            q["_Year"] = pd.to_numeric(q["Year"], errors="coerce")
            q["_Q"] = q["Quarter"].apply(lambda x: float(re.search(r"(\d)", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d)", str(x)) else np.nan)
            q = q.dropna(subset=["_Year","_Q"]).sort_values(["_Year","_Q"]).tail(8)
            keep_q = ["Year","Quarter"] + [c for c in q.columns if str(c).startswith("Q_")]
            bundle["quarters_tail"] = (q[keep_q].replace({pd.NA: None, np.nan: None}).to_dict("records"))

        if not annual.empty:
            desired = ["Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS","NumShares",
                       "ShareholderEquity","TotalDebt","Cash","Dividend","SharePrice","CurrentPrice"]
            cols = [c for c in desired if c in annual.columns]
            bundle["annual_rows"] = (annual[cols].replace({pd.NA: None, np.nan: None}).to_dict("records"))

    return bundle

def build_global_catalog(df: pd.DataFrame, hard_limit: int | None = None) -> list[dict]:
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

if "IsQuarter" not in df.columns:
    df["IsQuarter"] = False
if "Quarter" not in df.columns:
    df["Quarter"] = pd.NA

st.session_state["_MASTER_DF"] = df

stocks = sorted([s for s in df["Name"].dropna().astype(str).unique().tolist()])
st.caption(f"Loaded **{len(stocks)}** stock(s).")

# =========================
# Sidebar — OpenAI + Context
# =========================
sb = st.sidebar
sb.header("🔐 OpenAI")

# Load key from session/secrets/env; keep in session so it survives reruns
_default_key = st.session_state.get("ai_api_key") or os.getenv("OPENAI_API_KEY") or ""
try:
    if not _default_key:
        _default_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass

api_key = sb.text_input("API Key", type="password", value=_default_key, help="Stored only in your session.")
if api_key:
    st.session_state["ai_api_key"] = api_key

model = sb.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
temperature = sb.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

sb.markdown("---")
sb.subheader("🧠 Context to send")
scan_all = sb.checkbox("Include ALL stocks (catalog)", value=True)
include_full_for_mentions = sb.checkbox("Full details for mentioned names", value=True)
show_payload = sb.checkbox("Show raw payload in chat", value=False, help="Echo the JSON context (for debugging).")

# =========================
# Main — Ask + Preview
# =========================
def _mentioned_names(text: str) -> list[str]:
    t = (text or "").lower()
    found = []
    for n in stocks:
        if n.lower() in t:
            found.append(n)
    return list(dict.fromkeys(found))

st.markdown('<div class="sec success"><div class="t">🧠 Context</div><div class="d">What the AI can see</div></div>', unsafe_allow_html=True)

default_q = "Which looks best now and why? Rank top-5 with key metrics."
user_q = st.text_area("Ask the AI", value=default_q, height=90,
                      help="Examples: 'Rank top-5', 'Tell me about HUP SENG', 'Build a thesis comparing A vs B', 'Should I buy HUP SENG now?'.")
mentioned = _mentioned_names(user_q)

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

SYSTEM = """You are an equity analyst working inside a user’s private dashboard.
Use ONLY the data provided in the context. If data is missing, say so clearly.
When ranking or recommending, give short bullet rationales referencing specific fields (e.g., “TTM Net Margin 12.4%”, “P/E (TTM) 9.2”, “Interest Coverage < 3”).
Prefer TTM figures for profitability and valuation; fall back to latest annual when TTM not available.
If latest_annual.CurrentPrice is null, treat momentum.price (if present) as the current price.
If the user asks “Should I buy X now?” produce a concise verdict (Buy / Watch / Avoid) with 3–5 bullets citing concrete metrics from the context, then 1–2 key risks.
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
    pass  # spacer

if clear:
    st.session_state["chat"] = []
    st.rerun()

if ask:
    if not st.session_state.get("ai_api_key"):
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        _push("user", user_q)
        msgs = build_prompt_messages(user_q, catalog, packs_for_prompt)

        if show_payload:
            debug_blob = json.loads(msgs[0]["content"].split("CONTEXT:\n",1)[1].split("\n\nQUESTION:",1)[0])
            _push("assistant", f"<div class='small'><b>Context sent to model:</b></div>\n<pre class='kv'>{json.dumps(debug_blob, indent=2, ensure_ascii=False)}</pre>")

        answer = call_openai_chat(st.session_state["ai_api_key"], model, SYSTEM, msgs, temperature=temperature)
        _push("assistant", answer)
        st.rerun()

# =========================
# Quick helpers: best-5 (local, no AI)
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

