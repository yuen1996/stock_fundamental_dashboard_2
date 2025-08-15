# pages/11_AI_Analyst.py
# 🤖 AI Analyst — grounded on your app's data (Annual, Quarterly, TTM, Momentum, Scores)
from __future__ import annotations

import os, sys, re, json
from typing import Any, Iterable
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Optional auth ----------------
try:
    from auth_gate import require_auth
    require_auth()
except Exception:
    pass

# ---------------- Robust imports ----------------
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
for p in (_PARENT, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import io_helpers, calculations, rules  # type: ignore
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # type: ignore

# Do NOT import config here (prevents your previous ImportError).

# ---------------- OpenAI client ----------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def get_client(key: str | None):
    """Return OpenAI client or None if not configured."""
    if OpenAI is None:
        return None
    key = key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

# ---------------- Theme (same CSS as rest) -----
st.set_page_config(layout="wide", page_title="🤖 AI Analyst")

BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --text:#0f172a; --muted:#475569; --border:#e5e7eb; --shadow:0 8px 24px rgba(15,23,42,.06);
  --primary:#4f46e5; --info:#0ea5e9; --success:#10b981; --warning:#f59e0b; --danger:#ef4444;
}
html, body, [class*="css"]{ font-size:16px !important; color:var(--text); }
.stApp{ background: radial-gradient(1000px 500px at 10% -10%, #f0f3fb 0%, var(--bg) 60%), var(--bg); }
h1, h2, h3, h4{ color:var(--text) !important; font-weight:800 !important; letter-spacing:.2px; }

/* Section header card */
.sec{
  background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:var(--shadow);
  padding:.65rem .9rem; margin:1rem 0 .6rem 0; display:flex; align-items:center; gap:.6rem;
}
.sec .t{ font-size:1.05rem; font-weight:800; margin:0; padding:0; }
.sec .d{ color:var(--muted); font-size:.95rem; margin-left:.25rem; }
.sec::before{ content:""; display:inline-block; width:8px; height:26px; border-radius:6px; background:var(--primary); }
.sec.info::before{ background:var(--info); } .sec.success::before{ background:var(--success); }
.sec.warning::before{ background:var(--warning); } .sec.danger::before{ background:var(--danger); }

/* Tables */
.stDataFrame, .stDataEditor{ font-size:15px !important; }
div[data-testid="stDataFrame"] table{ border-collapse:separate !important; border-spacing:0; }
div[data-testid="stDataFrame"] table tbody tr:hover td{ background:#f8fafc !important; }
div[data-testid="stDataFrame"] td{ border-bottom:1px solid var(--border) !important; }

/* Inputs */
div[data-baseweb="input"] input, textarea, .stNumberInput input{ font-size:15px !important; }

/* Buttons */
.stButton>button{ border-radius:12px !important; padding:.55rem 1.1rem !important; font-weight:700; }

/* Tabs */
.stTabs [role="tab"]{ font-size:15px !important; font-weight:600 !important; }

/* Sidebar dark */
[data-testid="stSidebar"]{ background:linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important; }
[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------- Title ------------------------
st.title("🤖 AI Analyst")
st.caption("Ask about your fundamentals. The model sees *only* the data in this app.")

# ---------------- Sidebar ----------------------
st.sidebar.subheader("OpenAI Settings")
api_key = st.sidebar.text_input("API Key (sk-…)", type="password", help="Or set OPENAI_API_KEY env var / st.secrets")
model   = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
max_stocks  = st.sidebar.slider("Max stocks in context", 5, 200, 50, 5)
send_full_bundles = st.sidebar.checkbox("Attach full bundles (deeper, larger prompt)", value=False)
strict_mode = st.sidebar.checkbox("Strict analyst tone (less fluffy)", value=True)

# ---------------- Load data --------------------
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please upload data on the Add/Edit page first.")
    st.stop()

if "IsQuarter" not in df.columns: df["IsQuarter"] = False
if "Quarter" not in df.columns: df["Quarter"] = pd.NA
if "Year" not in df.columns: df["Year"] = pd.NA

# Normalize helper
def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _qnum(q) -> float:
    if pd.isna(q): return np.nan
    m = re.search(r"(\d+)", str(q))
    if not m: return np.nan
    try:
        n = int(m.group(1));  return n if 1 <= n <= 4 else np.nan
    except Exception:
        return np.nan

# ---------------- Momentum helpers -------------
def compute_momentum_bits(name: str) -> dict[str, Any]:
    """Read /data/ohlc/<name>.csv (if present) and compute price, 200-DMA, 12m return."""
    try:
        oh = io_helpers.load_ohlc(name)
        if oh is None or oh.empty: return {}
        dfp = oh.copy()
        if "Date" in dfp.columns:
            dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
        dfp["Close"] = pd.to_numeric(dfp.get("Close"), errors="coerce")
        dfp = dfp.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
        if dfp.empty: return {}

        close = dfp["Close"]
        price = float(close.iloc[-1])
        ma200 = float(close.rolling(200, min_periods=200).mean().iloc[-1]) if len(close) >= 200 else None

        ret_12m = None
        if len(close) >= 252 and float(close.iloc[-252]) != 0.0:
            ret_12m = float(price / float(close.iloc[-252]) - 1.0)
        else:
            cutoff = dfp["Date"].iloc[-1] - pd.Timedelta(days=365)
            win = dfp[dfp["Date"] >= cutoff]
            if len(win) >= 2:
                base = float(win["Close"].iloc[0])
                if base != 0.0: ret_12m = float(price / base - 1.0)

        return {
            "price": price,
            "ma200": ma200,
            "ret_12m": ret_12m,
            "price_above_ma200": (price > ma200) if (ma200 is not None) else None
        }
    except Exception:
        return {}

# ---------------- Build per-stock bundle -------
def collect_bundle(name: str, full: bool = False) -> dict[str, Any]:
    """All key bits for one stock; 'full' optionally includes annual rows and last quarters."""
    s = df[df["Name"].astype(str) == str(name)].copy()
    s["_Year"] = _to_num(s["Year"])
    s["_Q"] = s["Quarter"].apply(_qnum)
    s = s.sort_values(["_Year","_Q"])

    # Industry & bucket (best-effort)
    industry = ""
    bucket = "General"
    try:
        if "Industry" in s.columns and s["Industry"].notna().any():
            industry = str(s["Industry"].dropna().astype(str).iloc[-1])
        if "IndustryBucket" in s.columns and s["IndustryBucket"].notna().any():
            bucket = str(s["IndustryBucket"].dropna().astype(str).iloc[-1]) or "General"
    except Exception:
        pass

    # Latest annual row
    annual = s[s["IsQuarter"] == False].copy()
    latest_annual = {}
    if not annual.empty:
        latest_annual = annual.sort_values(["_Year"]).iloc[-1].to_dict()

    # TTM + factor scores
    ttm = {}
    factor = {}
    try:
        ttm = calculations.compute_ttm(s) or {}
    except Exception:
        ttm = {}
    try:
        ohlc = compute_momentum_bits(name)
        factor = calculations.compute_factor_scores(
            stock_name=name, stock=s, ttm=ttm, ohlc_latest=ohlc, industry=None
        ) or {}
    except Exception:
        factor = {}

    # Pass/fail via rules if available
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
        # fallback to factor score if present
        if isinstance(factor, dict) and factor:
            pass_fail["score"] = sum([int(factor.get(k,0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5

    bundle = {
        "name": name,
        "industry": industry,
        "bucket": bucket,
        "latest_annual": {
            k: latest_annual.get(k) for k in [
                "Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS",
                "ShareholderEquity","NumShares","TotalDebt","Cash","SharePrice","CurrentPrice"
            ] if k in latest_annual
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
            "Composite": int(round(sum([int(factor.get(k,0) or 0) for k in ["Value","Quality","Growth","Cash","Momentum"]]) / 5)) if factor else None
        },
        "decision": pass_fail,
    }

    if full:
        # Last 8 quarters and all annual rows (compact numeric)
        q = s[s["IsQuarter"] == True].copy()
        if not q.empty:
            q = q.sort_values(["_Year","_Q"]).tail(8)
            keep_q = ["Year","Quarter"] + [c for c in q.columns if c.startswith("Q_")]
            bundle["quarters_tail"] = q[keep_q].replace({pd.NA: None, np.nan: None}).to_dict("records")
        if not annual.empty:
            keep_a = ["Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS","NumShares",
                      "ShareholderEquity","TotalDebt","Cash","Dividend","SharePrice","CurrentPrice"]
            bundle["annual_rows"] = (annual[keep_a]
                                     .apply(pd.to_numeric, errors="ignore")
                                     .replace({pd.NA: None, np.nan: None})
                                     .to_dict("records"))
    return bundle

# ---------------- Build packs for context -------
@st.cache_data(show_spinner=True, ttl=600)
def build_context(max_n: int, attach_full: bool) -> list[dict]:
    names = (df["Name"].dropna().astype(str).str.strip()
                .replace({"": np.nan}).dropna().unique().tolist())
    bundles = [collect_bundle(n, full=attach_full) for n in names]
    # sort by decision score (desc), then composite, then name
    def score_key(b):
        s1 = b.get("decision", {}).get("score")
        s2 = b.get("scorecard", {}).get("Composite")
        return (float(s1 if s1 is not None else -1e9),
                float(s2 if s2 is not None else -1e9),
                b["name"])
    bundles.sort(key=score_key, reverse=True)
    return bundles[:max_n]

context_packs = build_context(max_stocks, send_full_bundles)

# ---------------- Preview + local ranking -------
st.markdown('<div class="sec info"><div class="t">📦 Data Context</div><div class="d">What will be sent to the model (trimmed)</div></div>', unsafe_allow_html=True)
c1, c2 = st.columns([0.55, 0.45])
with c1:
    st.subheader("Top locally (by rules score / composite)")
    top_table = []
    for b in context_packs[:10]:
        top_table.append({
            "Name": b["name"],
            "Score": b.get("decision",{}).get("score"),
            "Pass": b.get("decision",{}).get("pass"),
            "Composite": b.get("scorecard",{}).get("Composite"),
            "Gross M% (TTM)": b.get("ttm",{}).get("TTM Gross Margin (%)"),
            "Net M% (TTM)": b.get("ttm",{}).get("TTM Net Margin (%)"),
            "PE (TTM)": b.get("ttm",{}).get("P/E (TTM)"),
            "ICov (EBITDA/Fin)": b.get("ttm",{}).get("Interest Coverage (EBITDA/Fin)"),
            "FCF Yld % (TTM)": b.get("ttm",{}).get("FCF Yield (TTM) (%)"),
            "12m Ret": (b.get("momentum",{}).get("ret_12m"))
        })
    st.dataframe(pd.DataFrame(top_table), use_container_width=True, height=360)
with c2:
    st.subheader("Prompt payload (preview)")
    st.caption(f"{len(context_packs)} stock(s); toggle 'full bundles' in the sidebar for deeper analysis.")
    st.json(context_packs, expanded=False)

# ---------------- Chat UX -----------------------
st.markdown('<div class="sec"><div class="t">💬 Chat & Insights</div><div class="d">Ask questions about your stocks</div></div>', unsafe_allow_html=True)

if "ai_msgs" not in st.session_state:
    sys_prompt = (
        "You are an equity analyst. Use ONLY the JSON context I give you; do not invent data. "
        "If data is missing, say 'N/A'.\n"
        "When ranking 'best', evaluate profitability (TTM margins, NP), valuation (P/E TTM, P/S TTM, P/B TTM, EV/EBITDA), "
        "balance-sheet strength (Net Cash, Debt/FCF, Interest Coverage), cash generation (FCF Yield, Cash Conversion), "
        "and momentum (12m return, price vs 200DMA) if provided. Prefer firms that PASS the rules screen.\n"
        "Output should be concise and metric-based:\n"
        "• For rankings: numbered list with 1-line reasons and key metrics like `NP margin xx% · P/E yy · ICov zz× · FCF yld aa% · 12m bb%`.\n"
        "• For a single stock: short profile sections: 'Snapshot', 'Valuation & Quality', 'Cash & Leverage', 'Momentum', 'Risks'.\n"
        "Never use external info. Never speculate."
    )
    if strict_mode:
        sys_prompt += "\nMaintain a neutral, precise tone. Avoid adjectives without numbers."
    st.session_state["ai_msgs"] = [{"role":"system","content":sys_prompt}]

# Render past exchanges (except system)
for m in st.session_state["ai_msgs"]:
    if m["role"] != "system":
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# --------------- Ask input ----------------------
question = st.chat_input("Ask e.g. 'Which looks best now and why? Rank top-5' or 'Tell me about HUP SENG'")
if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["ai_msgs"].append({"role":"user","content":question})

    # Detect if user asked about a single stock
    names_all = [b["name"] for b in context_packs]
    lower_q = question.lower()
    mentioned: list[str] = [n for n in names_all if n.lower() in lower_q]

    # Build context: if specific stock mentioned, restrict to that/those bundles
    if mentioned:
        packs_for_prompt = [collect_bundle(n, full=True) for n in mentioned]
    else:
        packs_for_prompt = context_packs

    # Local fallback answer (no API key)
    client = get_client(api_key)
    if client is None:
        # Simple local ranking response
        def _fmt(x, d=2):
            try:
                return f"{float(x):,.{d}f}"
            except Exception:
                return "N/A"
        lines = []
        if mentioned:
            for n in mentioned:
                b = collect_bundle(n, full=True)
                t = b.get("ttm", {})
                m = b.get("momentum", {})
                sc = b.get("scorecard", {})
                dec = b.get("decision", {})
                lines.append(
                    f"**{b['name']}** — "
                    f"Net M {_fmt(t.get('TTM Net Margin (%)'))}% · PE {_fmt(t.get('P/E (TTM)'))} · "
                    f"ICov {_fmt(t.get('Interest Coverage (EBITDA/Fin)'))}× · FCF yld {_fmt(t.get('FCF Yield (TTM) (%)'))}% · "
                    f"12m {(_fmt((m.get('ret_12m') or 0)*100) + '%') if m.get('ret_12m') is not None else 'N/A'} · "
                    f"Pass {dec.get('pass')} · Score {_fmt(dec.get('score'))} · Composite {sc.get('Composite')}"
                )
        else:
            ranked = packs_for_prompt[:5]
            for i, b in enumerate(ranked, 1):
                t = b.get("ttm", {})
                m = b.get("momentum", {})
                dec = b.get("decision", {})
                lines.append(
                    f"{i}. **{b['name']}** — "
                    f"NP M {_fmt(t.get('TTM Net Margin (%)'))}% · PE {_fmt(t.get('P/E (TTM)'))} · "
                    f"ICov {_fmt(t.get('Interest Coverage (EBITDA/Fin)'))}× · FCF yld {_fmt(t.get('FCF Yield (TTM) (%)'))}% · "
                    f"12m {(_fmt((m.get('ret_12m') or 0)*100) + '%') if m.get('ret_12m') is not None else 'N/A'} · "
                    f"Pass {dec.get('pass')} · Score {dec.get('score')}"
                )
            if not ranked:
                lines.append("_No stocks available to rank._")

        local_answer = "\n".join(lines) if lines else "No data available."
        with st.chat_message("assistant"):
            st.markdown(local_answer)
        st.session_state["ai_msgs"].append({"role":"assistant","content":local_answer})

    else:
        # Build strict prompt payload
        payload = {
            "dataset": packs_for_prompt,
            "instruction": (
                "Use ONLY the fields present. If a metric is missing, say 'N/A'. "
                "For rankings: produce a numbered list with short metric strings. "
                "For single-stock queries: give Snapshot, Valuation & Quality, Cash & Leverage, Momentum, Risks."
            )
        }
        user_prompt = (
            "CONTEXT (JSON):\n" + json.dumps(payload, ensure_ascii=False) +
            "\n\nQUESTION:\n" + question
        )

        # Call OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=st.session_state["ai_msgs"] + [{"role":"user","content":user_prompt}],
                        temperature=float(temperature),
                        max_tokens=1200,
                    )
                    answer = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    # Robust fallback: local ranking if API call fails
                    answer = f"_OpenAI error: {e}_\n\n"
                    ranked = packs_for_prompt[:5]
                    for i, b in enumerate(ranked, 1):
                        t = b.get("ttm", {})
                        answer += f"{i}. {b['name']} — PE {t.get('P/E (TTM)')} · NetM {t.get('TTM Net Margin (%)')}%\n"

            st.markdown(answer)
        st.session_state["ai_msgs"].append({"role":"assistant","content":answer})

# --------------- Utilities ----------------------
st.divider()
cA, cB, cC = st.columns(3)
with cA:
    if st.button("🔍 Rank best now", use_container_width=True):
        st.session_state["ai_msgs"].append({"role":"user","content":"Which looks best now and why? Rank top-5 with key metrics."})
        st.rerun()
with cB:
    if st.button("📄 Explain HUP SENG", use_container_width=True):
        # Quick helper to demo single-stock deep dive
        st.session_state["ai_msgs"].append({"role":"user","content":"Tell me about HUP SENG. Use only the dataset and show key metrics."})
        st.rerun()
with cC:
    if st.button("🧼 Clear chat", use_container_width=True):
        st.session_state.pop("ai_msgs", None)
        st.rerun()
