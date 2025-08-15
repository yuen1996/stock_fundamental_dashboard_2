# pages/11_AI_Analyst.py
# AI Analyst — grounded on YOUR data (annuals, quarterlies, TTM, factor score)
from __future__ import annotations

import os, sys, re, json, math
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import streamlit as st

# ---------------- Auth (optional) ----------------
try:
    from auth_gate import require_auth  # your app's optional guard
    require_auth()
except Exception:
    pass

# ---------------- Imports: robust ----------------
# Make project-root modules importable whether or not you have a utils/ package.
_THIS = os.path.dirname(__file__)
_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
for p in (_PARENT, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try direct imports first, then fallback to utils/
try:
    import io_helpers, calculations, rules  # type: ignore
except ModuleNotFoundError:
    from utils import io_helpers, calculations, rules  # type: ignore

# (Do NOT import config from utils; not required here and causes your error.)

# ---------------- OpenAI client ------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # We'll show a friendly message in UI.

def get_client(key: str | None):
    if OpenAI is None:
        st.error("Missing `openai` package. Add `openai>=1.40.0` to requirements.txt and reinstall.")
        return None
    key = key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"OpenAI init failed: {e}")
        return None

# ---------------- UI Layout ----------------------
st.set_page_config(layout="wide", page_title="🤖 AI Analyst")
st.title("🤖 AI Analyst")

st.sidebar.subheader("OpenAI Settings")
api_key = st.sidebar.text_input("API Key (sk-…)", type="password", help="Or set OPENAI_API_KEY env var / st.secrets")
model   = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
max_stocks = st.sidebar.slider("Max stocks to include in AI context", 5, 200, 50, 5)
temp = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

# ---------------- Load data ----------------------
df = io_helpers.load_data()
if df is None or df.empty or "Name" not in df.columns:
    st.warning("No data found. Please upload data on the Add/Edit page first.")
    st.stop()

# Ensure compatibility columns exist
if "IsQuarter" not in df.columns: df["IsQuarter"] = False
if "Quarter" not in df.columns: df["Quarter"] = pd.NA

# ---------------- Helpers ------------------------
def _to_float(x) -> float:
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    try:
        s = str(x).replace(",", "").strip()
        return float(s) if s != "" else np.nan
    except Exception:
        return np.nan

def _quarter_to_num(q):
    if pd.isna(q): return np.nan
    s = str(q).upper().strip()
    m = re.search(r"(\d+)", s)
    if not m: return np.nan
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 4 else np.nan
    except Exception:
        return np.nan

def _safe_get(d: dict, k: str, default=np.nan):
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return v

# ---------------- Build knowledge packs ----------
@st.cache_data(show_spinner=True, ttl=600)
def build_packs() -> list[dict]:
    # Group by stock name and compute compact summary (latest + TTM + score)
    out: list[dict] = []

    # Clean types for sort
    if "Year" in df.columns:
        df["_Year"] = pd.to_numeric(df["Year"], errors="coerce")
    else:
        df["_Year"] = np.nan
    if "Quarter" in df.columns:
        df["_Q"] = df["Quarter"].apply(_quarter_to_num)
    else:
        df["_Q"] = np.nan

    names = (
        df["Name"].dropna().astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
    )

    for name in names:
        s = df[df["Name"].astype(str) == str(name)].copy()

        # latest annual row
        annual = s[s["IsQuarter"] == False].copy()
        latest_annual: dict[str, Any] = {}
        if not annual.empty:
            latest_annual = annual.sort_values(["_Year"]).iloc[-1].to_dict()

        # compute TTM via your calculations module if available
        ttm: dict[str, Any] = {}
        try:
            # Provide the whole stock slice; your function should pick relevant fields.
            ttm = calculations.compute_ttm(s)  # type: ignore
        except Exception:
            # minimal fallback: sum last 4 quarters for some common fields
            q = s[s["IsQuarter"] == True].copy()
            if not q.empty and "Year" in q.columns and "Quarter" in q.columns:
                q = q.sort_values(["_Year", "_Q"]).tail(4)
                def sum_col(cand: list[str]):
                    for c in cand:
                        if c in q.columns:
                            vals = pd.to_numeric(q[c], errors="coerce")
                            if vals.notna().any():
                                return float(vals.sum(skipna=True))
                    return np.nan
                rev4  = sum_col(["Q_Revenue","Revenue"])
                gp4   = sum_col(["Q_GrossProfit","GrossProfit"])
                op4   = sum_col(["Q_OperatingProfit","OperatingProfit"])
                np4   = sum_col(["Q_NetProfit","NetProfit"])
                ebitda4 = sum_col(["Q_EBITDA","EBITDA"])
                eps4  = sum_col(["Q_EPS","EPS"])
                ttm = {
                    "TTM Revenue": rev4, "TTM Gross Profit": gp4,
                    "TTM Operating Profit": op4, "TTM Net Profit": np4,
                    "TTM EBITDA": ebitda4, "TTM EPS": eps4
                }

        # derive a few valuation / quality bits if present
        price = np.nan
        try:
            price = float(s.sort_values(["_Year","_Q"]).get("Price", pd.Series([np.nan])) .iloc[-1])
        except Exception:
            pass

        # Put together a compact pack
        pack = {
            "name": name,
            "latest_annual": {k: latest_annual.get(k) for k in [
                "Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS","Shares",
                "ShareholderEquity","TotalDebt","Cash","Price"
            ] if k in latest_annual},
            "ttm": {k: ttm.get(k) for k in [
                "TTM Revenue","TTM Gross Profit","TTM Operating Profit","TTM Net Profit",
                "TTM EBITDA","TTM EPS","P/E (TTM)","FCF Yield (TTM) (%)","TTM Gross Margin (%)",
                "TTM Net Margin (%)","Debt / FCF (yrs)","Cash Conversion (CFO/NP, %)",
            ] if k in ttm},
        }

        # evaluate with your rules (VQGM if available)
        score_block = None
        try:
            metrics_for_rules = {}
            metrics_for_rules.update(pack["ttm"])
            # map some common aliases if available in latest_annual
            for k, alias in [
                ("ROE (%)", "ROE (%)"),
                ("P/E", "P/E"),
                ("Current Ratio", "Current Ratio"),
                ("Debt-Asset Ratio (%)", "Debt-Asset Ratio (%)"),
                ("Dividend Yield (%)", "Dividend Yield (%)"),
                ("Dividend Payout Ratio (%)", "Dividend Payout Ratio (%)"),
                ("Interest Coverage (EBITDA/Fin)", "Interest Coverage (EBITDA/Fin)"),
            ]:
                if alias in s.columns:
                    try:
                        metrics_for_rules[k] = float(pd.to_numeric(s[alias], errors="coerce").iloc[-1])
                    except Exception:
                        pass

            ruleset_key = "VQGM"
            score_block = rules.evaluate(metrics_for_rules, ruleset_key)  # type: ignore
            pack["ruleset"] = ruleset_key
            pack["score"] = score_block["score"]
            pack["pass"] = score_block["pass"]
            pack["fail_reasons"] = score_block.get("reasons", [])
        except Exception:
            pack["ruleset"] = None

        out.append(pack)

    # sort by score if present, else by name
    out.sort(key=lambda x: (-(x.get("score") or -1), str(x["name"])))
    return out

packs = build_packs()
if not packs:
    st.info("No usable rows found.")
    st.stop()

# ----------------- Best now (local) --------------
def local_best(n=5) -> list[dict]:
    have_score = [p for p in packs if isinstance(p.get("score"), (int,float))]
    have_score.sort(key=lambda x: float(x["score"]), reverse=True)
    return have_score[:n]

# ----------------- Chat UI -----------------------
st.caption("Ask questions about your uploaded fundamentals. Optionally, provide an OpenAI key to use GPT. Without a key, I’ll answer using local scoring.")

# Left: chat; Right: reference table
col_chat, col_ref = st.columns([0.62, 0.38], gap="large")

with col_ref:
    st.subheader("Top by score (local)")
    top_local = local_best(10)
    st.dataframe(pd.DataFrame([{
        "Name": p["name"], "Score": p.get("score"), "Pass": p.get("pass"),
        "Fail Reasons": ", ".join(p.get("fail_reasons", []))
    } for p in top_local]), use_container_width=True)

    st.subheader("What I’ll send to the model")
    trimmed = packs[:max_stocks]
    st.caption(f"{len(trimmed)} stocks included (adjust in sidebar)")
    st.json(trimmed, expanded=False)

with col_chat:
    # Message store
    if "ai_msgs" not in st.session_state:
        st.session_state["ai_msgs"] = [
            {"role": "system", "content": (
                "You are an equity analyst. Answer ONLY using the dataset provided. "
                "Prefer objective comparisons, explain tradeoffs, and if asked to pick the best stock, "
                "give a ranked list with brief justifications tied to the metrics."
            )}
        ]

    # Render history (excluding system)
    for m in st.session_state["ai_msgs"]:
        if m["role"] != "system":
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    q = st.chat_input("Ask about your stocks (e.g., “Which looks best right now and why?”)")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        st.session_state["ai_msgs"].append({"role": "user", "content": q})

        # Decide: use OpenAI if key present, else local answer
        client = get_client(api_key)
        if client is None:
            # Local fallback: simple answer using rules ranking
            top = local_best(5)
            lines = ["I don’t have an API key, so here’s a local ranking by your rules score:\n"]
            for i, p in enumerate(top, 1):
                lines.append(f"{i}. **{p['name']}** — score {p.get('score')}; pass={p.get('pass')}; reasons: {', '.join(p.get('fail_reasons', [])) or '—'}")
            local_answer = "\n".join(lines)
            with st.chat_message("assistant"):
                st.markdown(local_answer)
            st.session_state["ai_msgs"].append({"role": "assistant", "content": local_answer})
        else:
            # Build compact context for the model
            context_json = json.dumps(packs[:max_stocks], ensure_ascii=False)
            prompt = (
                "Here is the full dataset (JSON list of stocks, each with latest_annual, ttm, score, and pass flags). "
                "Use ONLY this data—no outside info:\n\n"
                f"{context_json}\n\n"
                "Now answer the user's question. If asked to choose 'best', give a ranked top-5 with short metric-based reasons."
            )
            messages = st.session_state["ai_msgs"] + [{"role": "user", "content": prompt}]
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=float(temp),
                            max_tokens=900,
                        )
                        answer = resp.choices[0].message.content.strip()
                    except Exception as e:
                        answer = f"OpenAI error: {e}\nFalling back to local ranking.\n\n"
                        top = local_best(5)
                        for i, p in enumerate(top, 1):
                            answer += f"{i}. {p['name']} — score {p.get('score')}; pass={p.get('pass')}\n"
                st.markdown(answer)
            st.session_state["ai_msgs"].append({"role": "assistant", "content": answer})

# --------------- Convenience buttons -------------
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("🔍 Suggest the best stock now", use_container_width=True):
        if get_client(api_key) is None:
            # local print
            top = local_best(5)
            st.info("Local (no API key) best now:\n\n" + "\n".join([f"- {p['name']} (score {p.get('score')})" for p in top]))
        else:
            st.session_state["ai_msgs"].append({"role":"user","content":"Which looks best now and why? Rank top-5 with key metrics."})
            st.rerun()
with c2:
    if st.button("🧼 Clear chat history", use_container_width=True):
        st.session_state.pop("ai_msgs", None)
        st.rerun()

