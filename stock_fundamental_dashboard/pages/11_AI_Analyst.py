# pages/11_AI_Analyst.py
from __future__ import annotations

from auth_gate import require_auth
require_auth()

# =========================
# Path setup (robust)
# =========================
import os, sys, re, json, math
from typing import Any, Dict, List, Tuple

_THIS = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_THIS, "."))           # project root (parent of /pages)
_GRANDP = os.path.abspath(os.path.join(_ROOT, "."))         # repo root
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
_io_candidates   = ["io_helpers", "utils.io_helpers"]
_calc_candidates = ["calculations", "utils.calculations"]
_rules_candidates= ["rules", "utils.rules"]
_conf_candidates = ["config", "utils.config"]

if PKG:
    _io_candidates[:0]   = [f"{PKG}.io_helpers", f"{PKG}.utils.io_helpers"]
    _calc_candidates[:0] = [f"{PKG}.calculations", f"{PKG}.utils.calculations"]
    _rules_candidates[:0]= [f"{PKG}.rules", f"{PKG}.utils.rules"]
    _conf_candidates[:0] = [f"{PKG}.config", f"{PKG}.utils.config"]

try:
    io_helpers   = _try_import(_io_candidates)
    calculations = _try_import(_calc_candidates)
    rules        = _try_import(_rules_candidates)
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
# Page config + CSS
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
.kv{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size:.85rem; }

/* sticky chat input + scrollable chat */
.chat-wrap{ display:flex; flex-direction:column; gap:.6rem; height:70vh; }
.chat-scroll{ flex:1 1 auto; overflow:auto; padding-right:.25rem; }
.chat-input{ position: sticky; bottom: 0; background:transparent; padding-top:.25rem; }

/* sticky table headers for dataframes */
div[data-testid="stDataFrame"] table thead th { position: sticky; top: 0; background: #f9fafb !important; z-index: 10; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

st.header("🤖 AI Analyst")

# =========================
# Helpers
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
        payload = json.dumps(obj, default=str, indent=2, ensure_ascii=False)
    except Exception:
        try:
            if isinstance(obj, pd.DataFrame):
                obj = obj.replace({pd.NA: None, np.nan: None}).to_dict("records")
            elif isinstance(obj, pd.Series):
                obj = obj.replace({pd.NA: None, np.nan: None}).to_dict()
            payload = json.dumps(obj, default=str, indent=2, ensure_ascii=False)
        except Exception:
            payload = json.dumps({"error": "failed to serialize"}, indent=2, ensure_ascii=False)
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
        "ROE (%)": ["ROE (%)","ROE","Return on equity (%)","Return On Equity"],
    }
    for canon, cands in alias_map.items():
        if canon not in df2.columns:
            for c in cands:
                if c in df2.columns:
                    df2[canon] = df2[c]
                    break
    return df2

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

# ---------- Growth helpers (CAGR & summary) ----------
def _cagr_percent(old: float, new: float, years_span: int) -> float | None:
    try:
        if old is None or new is None: return None
        if old <= 0 or new <= 0 or years_span < 1: return None
        return ((new / old) ** (1.0 / years_span) - 1.0) * 100.0
    except Exception:
        return None

def _series_window_cagr(annual: pd.DataFrame, col: str, win_years: int) -> float | None:
    """Anchor at the latest year, look back up to N years (mirrors View Stock)."""
    try:
        if annual.empty or col not in annual.columns: return None
        ax = annual[["Year", col]].dropna().copy()
        ax["Year"] = pd.to_numeric(ax["Year"], errors="coerce")
        ax[col] = pd.to_numeric(ax[col], errors="coerce")
        ax = ax.dropna().astype({"Year": int}).sort_values("Year")
        if ax.empty or len(ax["Year"].unique()) < 2: return None
        years = sorted(ax["Year"].unique())
        last_y = years[-1]
        first_candidates = [y for y in years if (last_y - y) <= win_years and y <= last_y]
        first_y = first_candidates[0] if first_candidates else years[0]
        span = max(1, last_y - first_y)
        first_val = float(ax.loc[ax["Year"] == first_y, col].iloc[0])
        last_val  = float(ax.loc[ax["Year"] == last_y, col].iloc[0])
        return _cagr_percent(first_val, last_val, span)
    except Exception:
        return None

def _bvps_col(annual: pd.DataFrame) -> pd.Series:
    try:
        eq  = pd.to_numeric(annual.get("ShareholderEquity"), errors="coerce")
        sh  = pd.to_numeric(annual.get("NumShares"), errors="coerce")
        bvps = (eq / sh).replace([np.inf, -np.inf], np.nan)
        return bvps
    except Exception:
        return pd.Series(dtype=float)

def _series_last_n(annual: pd.DataFrame, col: str, n: int = 5) -> list[dict]:
    try:
        if col not in annual.columns or annual.empty: return []
        df = annual[["Year", col]].dropna()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna().astype({"Year": int}).sort_values("Year").tail(n)
        return [{"Year": int(r.Year), col: (None if pd.isna(r[col]) else float(r[col]))} for r in df.itertuples()]
    except Exception:
        return []

# ---------- Momentum ----------
def compute_momentum_bits(name: str) -> dict | None:
    """Read uploaded OHLC and compute price, MAs, returns, 52w stats."""
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
        ma50  = float(df["Close"].rolling(50,  min_periods=50 ).mean().iloc[-1]) if len(df) >= 50  else None
        ma200 = float(df["Close"].rolling(200, min_periods=200).mean().iloc[-1]) if len(df) >= 200 else None

        def _ret_back(days: int) -> float | None:
            if len(df) < days+1: return None
            base = float(df["Close"].iloc[-(days+1)])
            if base == 0: return None
            return float(price / base - 1.0)

        ret_3m  = _ret_back(63)
        ret_6m  = _ret_back(126)
        ret_12m = _ret_back(252)

        cutoff = df["Date"].iloc[-1] - pd.Timedelta(days=365)
        win = df[df["Date"] >= cutoff]
        if win.empty:
            win = df.tail(min(len(df), 252))
        hi52 = float(win["Close"].max()) if not win.empty else None
        lo52 = float(win["Close"].min()) if not win.empty else None
        off_high = (price / hi52 - 1.0) if (hi52 and hi52 != 0) else None

        return {
            "price": price,
            "ma50": ma50,
            "ma200": ma200,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "high_52w": hi52,
            "low_52w": lo52,
            "off_high_52w": off_high,
            "above_ma200": (ma200 is not None and price >= ma200) or None,
        }
    except Exception:
        return None

# ---------- Core bundler ----------
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

    # Annual block
    annual = s[s["IsQuarter"] == False].copy()
    annual = _add_canonical_annual_cols(annual)

    # Derive BVPS column (for CAGR & download)
    if "BVPS" not in annual.columns:
        annual["BVPS"] = _bvps_col(annual)

    latest_annual_dict = {}
    if not annual.empty:
        latest_annual_dict = annual.sort_values(["_Year"]).tail(1).iloc[0].to_dict()

    # Momentum / last price
    ohlc = compute_momentum_bits(name)

    # If CurrentPrice missing in annual, fill from OHLC price
    if ohlc and (_is_missing(latest_annual_dict.get("CurrentPrice"))):
        latest_annual_dict["CurrentPrice"] = ohlc.get("price")

    # Compute TTM using your calculations (pass price when possible for valuation)
    try:
        curr_price = _safe_float((ohlc or {}).get("price")) \
                     or _safe_float(latest_annual_dict.get("CurrentPrice")) \
                     or _safe_float(latest_annual_dict.get("SharePrice"))
        ttm = calculations.compute_ttm(s, current_price=curr_price)  # type: ignore
    except Exception:
        try:
            ttm = calculations.compute_ttm(s)  # back-compat
        except Exception:
            ttm = {}

    # Fill useful TTM-derived ratios if inputs exist
    try:
        latest_price = _safe_float((ohlc or {}).get("price")) \
                       or _safe_float(latest_annual_dict.get("CurrentPrice")) \
                       or _safe_float(latest_annual_dict.get("SharePrice"))
        shares = _safe_float(latest_annual_dict.get("NumShares"))
        equity = _safe_float(latest_annual_dict.get("ShareholderEquity"))
        debt   = _safe_float(latest_annual_dict.get("TotalDebt"))
        cash   = _safe_float(latest_annual_dict.get("Cash"))
        eps_ttm= _safe_float(ttm.get("TTM EPS"))
        rev_ttm= _safe_float(ttm.get("TTM Revenue"))
        ebitda = _safe_float(ttm.get("TTM EBITDA"))

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
            ttm["P/B (TTM)"]  = marketcap / equity
        if marketcap and ebitda and ebitda > 0:
            ev = marketcap + (debt or 0.0) - (cash or 0.0)
            ttm["EV/EBITDA (TTM)"] = ev / ebitda
    except Exception:
        pass

    # Factor scores (existing method)
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

    # ---- Growth/CAGR block (anchor latest year over 3y/5y; matches View Stock intent) ----
    growth: Dict[str, Any] = {}
    if not annual.empty:
        try:
            ax = annual.copy()
            ax["Year"] = pd.to_numeric(ax["Year"], errors="coerce")
            ax = ax.dropna(subset=["Year"]).astype({"Year": int}).sort_values("Year")

            # Ensure these numeric columns exist
            for col in ["Revenue","GrossProfit","NetProfit","EPS","BVPS","ROE (%)"]:
                if col in ax.columns:
                    ax[col] = pd.to_numeric(ax[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

            def c(col, n):
                return _series_window_cagr(ax, col, n)

            # 3y & 5y windows (in %)
            growth.update({
                "Revenue CAGR (3y) (%)":      c("Revenue",     3),
                "Revenue CAGR (5y) (%)":      c("Revenue",     5),
                "Net Profit CAGR (3y) (%)":   c("NetProfit",   3),
                "Net Profit CAGR (5y) (%)":   c("NetProfit",   5),
                "EPS CAGR (3y) (%)":          c("EPS",         3),
                "EPS CAGR (5y) (%)":          c("EPS",         5),
                "Gross Profit CAGR (3y) (%)": c("GrossProfit", 3),
                "Gross Profit CAGR (5y) (%)": c("GrossProfit", 5),
                "BVPS CAGR (3y) (%)":         c("BVPS",        3),
                "BVPS CAGR (5y) (%)":         c("BVPS",        5),
            })

            # ROE stability (median & stdev over 5y)
            if "ROE (%)" in ax.columns:
                roe5 = pd.to_numeric(ax["ROE (%)"], errors="coerce").dropna().tail(5)
                growth["ROE median (5y)"] = float(roe5.median()) if len(roe5) else None
                growth["ROE stdev (5y)"]  = float(roe5.std(ddof=0)) if len(roe5) else None
        except Exception:
            pass

    # Latest-annual fields sent
    latest_annual_out = {
        k: latest_annual_dict.get(k)
        for k in [
            "Year","Revenue","GrossProfit","OperatingProfit","NetProfit","EPS",
            "ShareholderEquity","NumShares","TotalDebt","Cash","Dividend","SharePrice","CurrentPrice"
        ] if k in latest_annual_dict
    }
    if ohlc and _is_missing(latest_annual_out.get("CurrentPrice")):
        latest_annual_out["CurrentPrice"] = ohlc.get("price")

    # Assemble bundle
    ttm_keep = [
        "TTM Revenue","TTM Gross Profit","TTM Operating Profit","TTM Net Profit","TTM EBITDA",
        "TTM CFO","TTM CapEx","TTM FCF",
        "TTM EPS","TTM Gross Margin (%)","TTM Net Margin (%)",
        "P/E (TTM)","P/S (TTM)","P/B (TTM)","EV/EBITDA (TTM)",
        "FCF Yield (TTM) (%)","Cash Conversion (CFO/NP, %)","Debt / FCF (yrs)",
        "Interest Coverage (EBITDA/Fin)","MarketCap","Current Price",
        "Net Cash (Debt)","Net Cash / MC (%)","DataHealth"
    ]

    bundle: Dict[str, Any] = {
        "name": name,
        "industry": industry,
        "bucket": bucket,
        "latest_annual": latest_annual_out,
        "annual_last_5": {
            "Revenue":   _series_last_n(annual, "Revenue", 5),
            "NetProfit": _series_last_n(annual, "NetProfit", 5),
            "EPS":       _series_last_n(annual, "EPS", 5),
            "Dividend":  _series_last_n(annual, "Dividend", 5),
            "ROE (%)":   _series_last_n(annual, "ROE (%)", 5),
        },
        "growth": growth,
        "ttm": {k: ttm.get(k) for k in ttm_keep if k in ttm},
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
                       "ShareholderEquity","TotalDebt","Cash","Dividend","SharePrice","CurrentPrice","BVPS"]
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

_default_key = st.session_state.get("ai_api_key") or os.getenv("OPENAI_API_KEY") or ""
try:
    if not _default_key:
        _default_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass

api_key = sb.text_input("API Key", type="password", value=_default_key, help="Stored only in your session.")
if api_key:
    st.session_state["ai_api_key"] = api_key

# >>> Added gpt-4.1 model option <<<
model = sb.selectbox("Model", ["gpt-4.1", "gpt-4o", "gpt-4o-mini"], index=0)
temperature = sb.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

sb.markdown("---")
sb.subheader("🧠 Context to send")
scan_all = sb.checkbox("Include ALL stocks (catalog)", value=True)
include_full_for_mentions = sb.checkbox("Full details for mentioned names", value=True)
show_payload = sb.checkbox("Show raw payload in chat", value=False, help="Echo the JSON context (for debugging).")

# =========================
# Context preview
# =========================
def _mentioned_names(text: str) -> list[str]:
    t = (text or "").lower()
    found = []
    for n in stocks:
        if n.lower() in t:
            found.append(n)
    return list(dict.fromkeys(found))

if "chat_inited" not in st.session_state:
    st.session_state["chat_inited"] = False

default_q = "Which looks best now and why? Rank top-5 with key metrics."
seed_text = default_q if not st.session_state["chat_inited"] else ""

last_user_content = None
for m in st.session_state.get("chat", []):
    if m.get("role") == "user":
        last_user_content = m.get("content")
mentioned = _mentioned_names(last_user_content or seed_text)

catalog = build_global_catalog(df) if scan_all else []
packs_for_prompt = [collect_bundle(n, full=True) for n in mentioned] if include_full_for_mentions and mentioned else []

st.markdown('<div class="sec success"><div class="t">🧠 Context</div><div class="d">What the AI can see</div></div>', unsafe_allow_html=True)
st.write(f"- Mentioned: {mentioned if mentioned else '—'}")
colp, cold = st.columns(2)
with colp:
    _download_json_button("📥 Download context — mentioned (JSON)", packs_for_prompt, "ai_context_mentioned.json", key="dl_ctx_m")
with cold:
    _download_json_button("📥 Download context — catalog (JSON)", catalog, "ai_context_catalog.json", key="dl_ctx_c")

# =========================
# Chat runtime (sticky input + Enter to send)
# =========================
if "chat" not in st.session_state:
    st.session_state["chat"] = []

SYSTEM = """You are an equity analyst working inside a user’s private dashboard.
Use ONLY the data provided in the context. If data is missing, say so clearly.
When ranking or recommending, give short bullet rationales referencing specific fields (e.g., “TTM Net Margin 12.4%”, “P/E (TTM) 9.2”, “Interest Coverage < 3”).
Prefer TTM figures for profitability and valuation; fall back to latest annual when TTM not available.
Use provided growth metrics (e.g., 3y/5y Revenue & EPS CAGR) and momentum (ret_3m/6m/12m, MA50/MA200, 52w off-high) where relevant.
If latest_annual.CurrentPrice is null, treat momentum.price (if present) as the current price.
If the user asks “Should I buy X now?” produce a concise verdict (Buy / Watch / Avoid) with 3–5 bullets citing concrete metrics from the context, then 1–2 key risks.
NEVER invent values that are not present in the context."""

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
    # Sanitize NaN/inf robustly (keeps your “JSON function to prevent error” spirit)
    ctx_bits = []
    if catalog:
        ctx_bits.append({"type":"catalog", "stocks": catalog})
    if packs:
        ctx_bits.append({"type":"focus_stocks", "stocks": packs})
    try:
        context_str = json.dumps(ctx_bits, ensure_ascii=False)
    except Exception:
        def _sanitize(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k:_sanitize(v) for k,v in o.items()}
            if isinstance(o, list):
                return [_sanitize(x) for x in o]
            return o
        context_str = json.dumps(_sanitize(ctx_bits), ensure_ascii=False)

    return [{
        "role": "user",
        "content": (
            "CONTEXT JSON BELOW.\n"
            "1) Read it carefully.\n"
            "2) Then answer the user question.\n"
            "3) Be concise and cite concrete fields (TTM/ratios) when you justify.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {question}"
        )
    }]

# --- Chat UI (sticky) ---
st.markdown('<div class="sec warning"><div class="t">💬 Chat</div><div class="d">Ask your question and let the model rank or analyze</div></div>', unsafe_allow_html=True)

chat_wrap = st.container()
with chat_wrap:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    chat_scroll = st.container()
    with chat_scroll:
        for m in st.session_state["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)

    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    prompt = st.chat_input("Type your question and press Enter…", key="chat_input")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Action row
cA, cB = st.columns([1,1])
with cA:
    if st.button("Clear chat", use_container_width=True):
        st.session_state["chat"] = []
        st.session_state["chat_inited"] = False
        st.rerun()
with cB:
    pass

# First render tip
if "chat_inited" not in st.session_state or not st.session_state["chat_inited"]:
    st.info("Tip: ask things like **'Rank top 5 by composite + valuation'**, or **'Should I buy HUP SENG now?'**")
    st.session_state["chat_inited"] = True

# Handle submit
if prompt is not None:
    if not st.session_state.get("ai_api_key"):
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        mentioned_now = _mentioned_names(prompt)
        packs_for_prompt = [collect_bundle(n, full=True) for n in mentioned_now] if include_full_for_mentions and mentioned_now else []
        catalog = build_global_catalog(df) if scan_all else []

        st.session_state["chat"].append({"role":"user", "content": prompt})

        msgs = build_prompt_messages(prompt, catalog, packs_for_prompt)

        if show_payload:
            debug_blob = json.loads(msgs[0]["content"].split("CONTEXT:\n",1)[1].split("\n\nQUESTION:",1)[0])
            st.session_state["chat"].append({"role":"assistant", "content": f"<div class='small'><b>Context sent to model:</b></div>\n<pre class='kv'>{json.dumps(debug_blob, indent=2, ensure_ascii=False)}</pre>"})

        answer = call_openai_chat(st.session_state["ai_api_key"], model, SYSTEM, msgs, temperature=temperature)
        st.session_state["chat"].append({"role":"assistant","content": answer})
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
