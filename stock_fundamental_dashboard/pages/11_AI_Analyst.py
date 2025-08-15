# pages/11_AI_Analyst.py
# Streamlit “AI Analyst” — chat with your dataset and ask “what’s best now?”
from __future__ import annotations
import os, json, math
import pandas as pd
import numpy as np
import streamlit as st

# --- import your local modules (works whether utils/ exists or files are top-level) ---
try:
    from utils import io_helpers, calculations, rules
except Exception:
    import io_helpers, calculations, rules  # fallback

# --- OpenAI client (modern SDK) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled below

PAGE_TITLE = "🤖 AI Analyst"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# ╭─────────────────────────────────────────────────────────────────────────────╮
# │ 1) API key + model                                                         │
# ╰─────────────────────────────────────────────────────────────────────────────╯
st.sidebar.subheader("OpenAI Settings")
api_key = st.sidebar.text_input("API Key (sk-…)", type="password", help="Will fall back to env var OPENAI_API_KEY or st.secrets['OPENAI_API_KEY']")
model   = st.sidebar.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"], index=0)

def get_client():
    if OpenAI is None:
        st.error("Missing `openai` package. Add `openai>=1.40.0` to requirements.txt and reinstall.")
        return None
    key = api_key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        st.warning("Provide an API key in the sidebar (or set OPENAI_API_KEY).")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"OpenAI init failed: {e}")
        return None

# ╭─────────────────────────────────────────────────────────────────────────────╮
# │ 2) Build a compact “knowledge pack” per stock                              │
# ╰─────────────────────────────────────────────────────────────────────────────╯
@st.cache_data(show_spinner=True, ttl=600)
def build_pack() -> dict[str, dict]:
    """Read your master dataframe and compress to small, LLM-friendly summaries."""
    df = io_helpers.load_data()  # your existing loader
    if df is None or df.empty:
        return {}

    out: dict[str, dict] = {}
    names = sorted([x for x in df["Name"].dropna().unique().tolist() if str(x).strip()])

    for name in names:
        s = df[df["Name"] == name].copy()
        s_year = pd.to_numeric(s.get("Year"), errors="coerce")
        s_qtr  = pd.to_numeric(s.get("Quarter"), errors="coerce")

        # Latest annual row
        annual = s[s.get("IsQuarter", False) == False].copy()
        if not annual.empty:
            annual = annual.assign(_Y=pd.to_numeric(annual.get("Year"), errors="coerce"))
            latest_annual = annual.sort_values("_Y").iloc[-1].to_dict()
        else:
            latest_annual = {}

        # Compute TTM + derived valuation/quality metrics via your calculations module
        ttm = {}
        try:
            # compute_ttm() picks best-effort aliases for EBIT/EBITDA/FCF etc. as per your code
            ttm = calculations.compute_ttm(s)  # returns dict of TTM figures
        except Exception:
            ttm = {}

        derived = {}
        try:
            derived = calculations.compute_derived_metrics(s, ttm)  # EV/EBITDA, FCF yield, cash conversion, etc.
        except Exception:
            pass

        # Minimal ratios snapshot from “latest annual” (if present)
        # Keep only a few numerics that help ranking & guard token size.
        def _pick(d, keys):
            return {k: d.get(k) for k in keys if k in d}
        ann_keys = [
            "Revenue","GrossProfit","OperatingProfit","NetProfit","OperatingCashFlow","FreeCashFlow",
            "TotalAssets","TotalLiabilities","ShareholderEquity","Dividend","DividendYield (%)",
            "Gross Profit Margin (%)","Net Profit Margin (%)","ROE (%)","Current Ratio","P/E","P/B"
        ]
        ann = _pick(latest_annual, ann_keys)

        # Optional: factor signals / confidence (lightweight) → helps ranking
        # If you have OHLC, you can pass latest close & momentum into the scorer
        factor = {}
        try:
            factor = calculations.compute_factor_scores(
                name, s, ttm, ohlc_latest=None, industry=latest_annual.get("IndustryBucket") or latest_annual.get("Industry")
            )
        except Exception:
            factor = {}

        # Compact textual bullet summary (keeps context lean)
        bullets = []
        def add(label, val, fmt="{:.1f}"):
            try:
                if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    return
                if isinstance(val, (int, float)):
                    bullets.append(f"{label}: {fmt.format(float(val))}")
                else:
                    bullets.append(f"{label}: {val}")
            except Exception:
                pass

        add("FCF Yield (TTM, %)", derived.get("FCF Yield (TTM) (%)"))
        add("EV/EBITDA (TTM)",   derived.get("EV/EBITDA (TTM)"), "{:.2f}")
        add("Cash Conversion %", derived.get("Cash Conversion (CFO/NP, %)"))
        add("Interest Cover",    derived.get("Interest Coverage (EBITDA/Fin)"), "{:.2f}")
        add("Debt/FCF (yrs)",    derived.get("Debt / FCF (yrs)"), "{:.2f}")
        add("TTM Gross Margin %", derived.get("TTM Gross Margin (%)"))
        add("TTM Net Margin %",   derived.get("TTM Net Margin (%)"))
        add("Dividend Yield %",   ann.get("DividendYield (%)"))
        add("P/E",                ann.get("P/E"), "{:.1f}")
        add("P/B",                ann.get("P/B"), "{:.1f}")
        # Factor headline if present
        try:
            fscore = factor.get("overall", {}).get("score")
            if fscore is None:
                fscore = factor.get("composite")
            if fscore is not None:
                add("Factor Score (0-100)", fscore, "{:.0f}")
        except Exception:
            pass

        out[name] = {
            "name": name,
            "industry": latest_annual.get("IndustryBucket") or latest_annual.get("Industry"),
            "latest_year": latest_annual.get("Year"),
            "ann": ann,
            "ttm": {k: v for k, v in (ttm or {}).items() if isinstance(v, (int, float)) or v is None},
            "derived": {k: v for k, v in (derived or {}).items() if isinstance(v, (int, float)) or v is None},
            "factor": factor if isinstance(factor, dict) else {},
            "summary": "; ".join(bullets)[:400],  # keep short for token budget
        }

    return out

with st.spinner("Building knowledge pack…"):
    pack = build_pack()
if not pack:
    st.info("No data loaded yet. Fill your dataset first.")
    st.stop()

# Quick peek
st.caption(f"Knowledge pack ready for {len(pack)} stocks (compact summaries + TTM/ratios).")

# ╭─────────────────────────────────────────────────────────────────────────────╮
# │ 3) Helpers to route user questions                                         │
# ╰─────────────────────────────────────────────────────────────────────────────╯
def _stock_subset_from_question(q: str) -> list[str]:
    q = (q or "").lower()
    names = list(pack.keys())
    hits = [n for n in names if n.lower() in q]
    # If the user didn’t mention a name, default to all (but we’ll cap N)
    return hits or names

def _render_context_json(names: list[str], cap: int = 25) -> str:
    rows = []
    for nm in names[:cap]:
        d = pack[nm]
        rows.append({
            "Name": d["name"],
            "Industry": d.get("industry"),
            "LatestYear": d.get("latest_year"),
            "Summary": d.get("summary"),
            "Derived": d.get("derived"),     # EV/EBITDA, FCF Yield, CashConv, IntCover, Debt/FCF …
            "TTM": d.get("ttm"),             # raw TTM figures (Revenue, EBITDA, CFO, FCF etc. if present)
            "Factor": {
                "score": d.get("factor", {}).get("overall", {}).get("score") or d.get("factor", {}).get("composite"),
                "details": d.get("factor", {}).get("details", None),
            },
        })
    return json.dumps({"stocks": rows}, ensure_ascii=False)

SYSTEM = (
    "You are an equity analyst. Answer ONLY using the provided JSON context. "
    "Be precise, cite numbers from the context, avoid inventing data. "
    "If user asks for 'best stock now', rank using Value/Quality/CashFlow solvency "
    "and momentum hints in the summaries; prefer positive FCF yield, sustainable leverage "
    "(Debt/FCF low), adequate interest cover, and reasonable EV/EBITDA/P-E. "
    "If context lacks data, say so and ask for missing fields."
)

def ask_model(prompt: str, names: list[str]) -> str:
    client = get_client()
    if client is None:
        return "No OpenAI client. Add your API key in the sidebar."
    ctx = _render_context_json(names)
    try:
        # Responses API (official, current) — single turn grounded on our JSON
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context JSON:\n{ctx}\n\nUser question: {prompt}"},
            ],
            max_output_tokens=800,
        )
        # Pull the text
        parts = resp.output_text if hasattr(resp, "output_text") else None
        if not parts and getattr(resp, "output", None):
            # safety for SDK variants
            for item in resp.output:
                if getattr(item, "content", None):
                    for c in item.content:
                        if c.type == "output_text" and getattr(c, "text", ""):
                            parts = c.text
                            break
        return parts or "(No text returned.)"
    except Exception as e:
        return f"OpenAI error: {e}"

# ╭─────────────────────────────────────────────────────────────────────────────╮
# │ 4) UI — Ask anything; “Best stock now?” button                              │
# ╰─────────────────────────────────────────────────────────────────────────────╯
with st.container():
    colA, colB = st.columns([3,1])
    with colB:
        if st.button("🔥 Ask: Which is best right now?"):
            q = "From the context, pick the best 3 stocks right now and explain briefly why."
            names = _stock_subset_from_question("")  # all
            st.session_state.setdefault("chat", [])
            st.session_state["chat"].append(("user", q))
            ans = ask_model(q, names)
            st.session_state["chat"].append(("assistant", ans))

    # Chat box
    st.subheader("Chat with your data")
    st.caption("Grounded on your pack (TTM, derived ratios, factor hints).")

    st.session_state.setdefault("chat", [])
    for role, msg in st.session_state["chat"]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)

    user_q = st.chat_input("Ask a question about these stocks…")
    if user_q:
        names = _stock_subset_from_question(user_q)
        st.session_state["chat"].append(("user", user_q))
        ans = ask_model(user_q, names)
        st.session_state["chat"].append(("assistant", ans))
        st.rerun()

# ╭─────────────────────────────────────────────────────────────────────────────╮
# │ 5) Optional: show the raw compact pack (debug)                              │
# ╰─────────────────────────────────────────────────────────────────────────────╯
with st.expander("Show compact knowledge pack (debug)"):
    st.json({k: {"industry": v.get("industry"), "summary": v.get("summary")} for k, v in pack.items()})
