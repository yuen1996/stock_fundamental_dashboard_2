# pages/10_Quant_Tech_Charts.py
import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Try to use your existing io_helpers for persistence
try:
    import io_helpers as _ioh
except Exception:
    try:
        from utils import io_helpers as _ioh  # fallback to utils/io_helpers.py
    except Exception:
        _ioh = None  # We'll fall back to direct filesystem writes

st.set_page_config(page_title="Quant Technical Charts", layout="wide")


# ---------- small helpers ----------
def _safe_upper(s):
    return (s or "").strip().upper()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _list_existing_ohlc_names() -> list[str]:
    """Prefer io_helpers if it exposes a lister, else scan ./data/ohlc/*.csv"""
    if _ioh and hasattr(_ioh, "list_ohlc_names"):
        try:
            return list(_ioh.list_ohlc_names())  # type: ignore[attr-defined]
        except Exception:
            pass
    base = os.path.join(os.getcwd(), "data", "ohlc")
    if not os.path.isdir(base):
        return []
    names = []
    for fn in os.listdir(base):
        if fn.lower().endswith(".csv"):
            names.append(os.path.splitext(fn)[0])
    return sorted(set(names))


def _normalize_ohlc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts TradingView CSV or Yahoo CSV (or similar) and returns a normalized
    OHLCV dataframe with columns: Date, Open, High, Low, Close, Volume.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    # Case-insensitive column picking
    lower_map = {str(c).strip().lower(): c for c in df_raw.columns}

    def pick(*names):
        for n in names:
            if n in lower_map:
                return lower_map[n]
        return None

    # Date/time
    c_date = pick("date", "time", "timestamp")
    # OHLCV (support variants from TradingView and Yahoo)
    c_open = pick("open", "o")
    c_high = pick("high", "h")
    c_low = pick("low", "l")
    # prefer 'close' / 'c', otherwise 'adj close'
    c_close = pick("close", "c", "adj close", "adj_close", "adjclose")
    c_vol = pick("volume", "v")

    out = pd.DataFrame()
    if c_date is None or c_close is None:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    out["Date"] = pd.to_datetime(df_raw[c_date], errors="coerce")

    # Handle epoch (ms or s) commonly seen in TradingView exports
    if pd.api.types.is_numeric_dtype(df_raw[c_date]):
        try:
            vals = pd.to_numeric(df_raw[c_date], errors="coerce")
            if vals.dropna().median() > 10_000_000_000:  # looks like milliseconds
                out["Date"] = pd.to_datetime(vals, unit="ms", errors="coerce")
            else:
                out["Date"] = pd.to_datetime(vals, unit="s", errors="coerce")
        except Exception:
            pass

    def to_num(col):
        if col is None:
            return np.nan
        return pd.to_numeric(df_raw[col], errors="coerce")

    out["Open"] = to_num(c_open)
    out["High"] = to_num(c_high)
    out["Low"] = to_num(c_low)
    out["Close"] = to_num(c_close)
    out["Volume"] = to_num(c_vol) if c_vol is not None else np.nan

    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return out[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ---- indicators (no external libs needed) ----
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int):
    return s.rolling(window, min_periods=window).mean()


def rsi(s: pd.Series, window: int = 14):
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, window: int = 14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def bollinger(s: pd.Series, window=20, num_std=2):
    mid = sma(s, window)
    std = s.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200)
    d["EMA12"] = ema(d["Close"], 12)
    d["EMA26"] = ema(d["Close"], 26)
    d["RSI14"] = rsi(d["Close"], 14)
    m_line, m_sig, m_hist = macd(d["Close"], 12, 26, 9)
    d["MACD"] = m_line
    d["MACDsig"] = m_sig
    d["MACDhist"] = m_hist
    d["ATR14"] = atr(d, 14)
    bb_mid, bb_up, bb_low = bollinger(d["Close"], 20, 2)
    d["BB_M"] = bb_mid
    d["BB_U"] = bb_up
    d["BB_L"] = bb_low
    # Simple momentum: 12m rate-of-change (approx 252 trading days)
    d["ROC_12m"] = d["Close"].pct_change(252)
    return d


# ---- Signal helpers ---------------------------------------------------------
def _rolling_max_prev(s: pd.Series, window: int):
    # Rolling max of the *previous* window (shift by 1 so today doesn't look at itself)
    return s.shift(1).rolling(window=window, min_periods=window).max()


def _rolling_min_prev(s: pd.Series, window: int):
    return s.shift(1).rolling(window=window, min_periods=window).min()


def _last_true_date(mask: pd.Series, dates: pd.Series):
    mask = mask.fillna(False)
    if not mask.any():
        return None
    idx = mask[mask].index[-1]
    return pd.to_datetime(dates.loc[idx]).date().isoformat()


def _last_cross_dates(a: pd.Series, b: pd.Series, dates: pd.Series):
    up = (a > b) & (a.shift(1) <= b.shift(1))
    dn = (a < b) & (a.shift(1) >= b.shift(1))
    return _last_true_date(up, dates), _last_true_date(dn, dates)


def build_signals(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Return (signals_timeseries_df, summary_dict)
    timeseries columns are booleans (or floats for ATR%), with a Date column.
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(), {}

    d = df_in.copy().reset_index(drop=True)
    # booleans
    d["Price_gt_SMA200"] = d["Close"] > d["SMA200"]
    d["SMA20_gt_SMA50"] = d["SMA20"] > d["SMA50"]
    d["MACD_gt_Signal"] = d["MACD"] > d["MACDsig"]
    d["RSI_gt_50"] = d["RSI14"] > 50
    d["ROC12m_pos"] = d["ROC_12m"] > 0

    # 52-week breakout / breakdown (≈252 trading days)
    d["Breakout_252"] = d["Close"] > _rolling_max_prev(d["Close"], 252)
    d["Breakdown_252"] = d["Close"] < _rolling_min_prev(d["Close"], 252)

    # 20-day Donchian breakouts (use High/Low channels)
    d["Donchian20_Up"] = d["Close"] > _rolling_max_prev(d["High"], 20)
    d["Donchian20_Down"] = d["Close"] < _rolling_min_prev(d["Low"], 20)

    # Cross events
    gc_up, gc_dn = _last_cross_dates(d["SMA50"], d["SMA200"], d["Date"])  # Golden/Death cross
    p200_up, p200_dn = _last_cross_dates(d["Close"], d["SMA200"], d["Date"])  # Price vs 200SMA

    # ATR% of price (volatility proxy)
    d["ATR_pct"] = (d["ATR14"] / d["Close"]).replace([np.inf, -np.inf], np.nan)

    last = d.iloc[-1]
    summary = {
        "now": pd.to_datetime(last["Date"]).date().isoformat(),
        "Price_gt_SMA200": bool(last["Price_gt_SMA200"]),
        "SMA20_gt_SMA50": bool(last["SMA20_gt_SMA50"]),
        "MACD_gt_Signal": bool(last["MACD_gt_Signal"]),
        "RSI_gt_50": bool(last["RSI_gt_50"]),
        "ROC12m_pos": bool(last["ROC12m_pos"]),
        "Breakout_252": bool(last["Breakout_252"]),
        "Breakdown_252": bool(last["Breakdown_252"]),
        "Donchian20_Up": bool(last["Donchian20_Up"]),
        "Donchian20_Down": bool(last["Donchian20_Down"]),
        "ATR_pct": float(last["ATR_pct"]) if pd.notna(last["ATR_pct"]) else np.nan,
        # last trigger dates
        "last_golden_cross": gc_up,
        "last_death_cross": gc_dn,
        "last_price>200_up": p200_up,
        "last_price>200_dn": p200_dn,
        "last_breakout_252": _last_true_date(d["Breakout_252"], d["Date"]),
        "last_breakdown_252": _last_true_date(d["Breakdown_252"], d["Date"]),
        "last_dc20_up": _last_true_date(d["Donchian20_Up"], d["Date"]),
        "last_dc20_dn": _last_true_date(d["Donchian20_Down"], d["Date"]),
    }

    # A simple composite score: count of bullish booleans
    bullish_keys = [
        "Price_gt_SMA200",
        "SMA20_gt_SMA50",
        "MACD_gt_Signal",
        "RSI_gt_50",
        "ROC12m_pos",
        "Breakout_252",
        "Donchian20_Up",
    ]
    summary["composite_score"] = int(sum(1 for k in bullish_keys if summary[k]))
    summary["composite_max"] = len(bullish_keys)

    ts_cols = [
        "Date",
        "Price_gt_SMA200",
        "SMA20_gt_SMA50",
        "MACD_gt_Signal",
        "RSI_gt_50",
        "ROC12m_pos",
        "Breakout_252",
        "Breakdown_252",
        "Donchian20_Up",
        "Donchian20_Down",
        "ATR_pct",
    ]
    return d[ts_cols].copy(), summary


# ---------- UI ----------
st.markdown(
    '<div class="sec"><div class="t">📈 Quant Technical Charts</div>'
    '<div class="d">Upload TradingView/Yahoo CSV or fetch via Yahoo, compute indicators, and chart</div></div>',
    unsafe_allow_html=True,
)

c0, c1, c2 = st.columns([1.2, 1, 1])
with c0:
    stock_name_raw = st.text_input("Stock Name (label only)", value="", placeholder="E.g., HUP SENG")
    stock_name = _safe_upper(stock_name_raw)
with c1:
    source = st.selectbox("Data source", ["Upload CSV", "Yahoo Finance", "Existing OHLC"], index=0)
with c2:
    period = st.selectbox("Period (Yahoo)", ["1y", "2y", "5y", "max"], index=0)

df_ohlc = pd.DataFrame()

if source == "Upload CSV":
    up = st.file_uploader("CSV file from TradingView or Yahoo", type=["csv"], accept_multiple_files=False)
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception:
            up.seek(0)
            raw = pd.read_csv(io.StringIO(up.getvalue().decode("utf-8")), engine="python")
        df_ohlc = _normalize_ohlc(raw)

elif source == "Yahoo Finance":
    ticker = st.text_input("Yahoo ticker (e.g., 0003.HK, AAPL, 1295.KL)", value="")
    if ticker.strip():
        try:
            import yfinance as yf  # import only when needed
            data = yf.download(ticker.strip(), period=period, auto_adjust=False, progress=False)
            data = data.reset_index()  # 'Date','Open','High','Low','Close','Adj Close','Volume'
            df_ohlc = _normalize_ohlc(data)
            if not stock_name:
                stock_name = _safe_upper(ticker)
        except Exception as e:
            st.error(f"Yahoo download failed: {e}")

else:  # Existing OHLC
    names = _list_existing_ohlc_names()
    if names:
        pick = st.selectbox("Select existing OHLC dataset", names)
        if pick:
            if _ioh and hasattr(_ioh, "load_ohlc"):
                oh = _ioh.load_ohlc(pick)  # type: ignore[attr-defined]
            else:
                base = os.path.join(os.getcwd(), "data", "ohlc")
                try:
                    oh = pd.read_csv(os.path.join(base, f"{pick}.csv"))
                except Exception:
                    oh = None
            if oh is not None and not getattr(oh, "empty", True):
                df_ohlc = _normalize_ohlc(oh)
                if not stock_name:
                    stock_name = _safe_upper(pick)
    else:
        st.info("No saved OHLC CSVs found yet.")

# Show a preview
if df_ohlc is not None and not df_ohlc.empty:
    st.dataframe(df_ohlc.tail(8), use_container_width=True, height=220)
else:
    st.info("Load a CSV or Yahoo series to begin.")
    st.stop()

# Controls
st.markdown("### Chart options")
cA, cB, cC, cD, cE = st.columns(5)
with cA:
    show_sma = st.checkbox("SMA 20/50/200", value=True)
with cB:
    show_bb = st.checkbox("Bollinger (20,2)", value=False)
with cC:
    show_ema = st.checkbox("EMA 12/26", value=False)
with cD:
    show_rsi = st.checkbox("RSI (14)", value=True)
with cE:
    show_macd = st.checkbox("MACD (12,26,9)", value=True)

# Date filter
min_d, max_d = df_ohlc["Date"].min(), df_ohlc["Date"].max()
d1, d2 = st.slider(
    "Date window",
    min_value=min_d.to_pydatetime(),
    max_value=max_d.to_pydatetime(),
    value=(
        max(min_d.to_pydatetime(), max_d.to_pydatetime() - pd.Timedelta(days=365)),
        max_d.to_pydatetime(),
    ),
    format="YYYY-MM-DD",
    key="qt_date_window",
)

# Compute indicators & chart
df_ind = enrich_indicators(df_ohlc)
m = (df_ind["Date"] >= pd.to_datetime(d1)) & (df_ind["Date"] <= pd.to_datetime(d2))
dfv = df_ind.loc[m].copy()

# --- Plotly: candles + overlays + volume + RSI/MACD ---
rows = 3 if (show_rsi or show_macd) else 2
fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.60, 0.15] + ([0.25] if rows == 3 else []),
)

# Row 1: Candlesticks
fig.add_trace(
    go.Candlestick(
        x=dfv["Date"], open=dfv["Open"], high=dfv["High"], low=dfv["Low"], close=dfv["Close"], name="Price"
    ),
    row=1,
    col=1,
)

if show_sma:
    for col, nm in [("SMA20", "SMA20"), ("SMA50", "SMA50"), ("SMA200", "SMA200")]:
        fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv[col], mode="lines", name=nm), row=1, col=1)
if show_ema:
    for col, nm in [("EMA12", "EMA12"), ("EMA26", "EMA26")]:
        fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv[col], mode="lines", name=nm), row=1, col=1)
if show_bb:
    fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["BB_U"], mode="lines", name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["BB_M"], mode="lines", name="BB Mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["BB_L"], mode="lines", name="BB Lower"), row=1, col=1)

# Row 2: Volume (bars)
fig.add_trace(go.Bar(x=dfv["Date"], y=dfv["Volume"], name="Volume"), row=2, col=1)

# Row 3: RSI or MACD (or both stacked in same row)
if rows == 3:
    if show_rsi:
        fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["RSI14"], mode="lines", name="RSI14"), row=3, col=1)
        # 70/30 guide
        fig.add_hrect(y0=70, y1=70, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hrect(y0=30, y1=30, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)
    if show_macd:
        fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["MACD"], mode="lines", name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=dfv["Date"], y=dfv["MACDsig"], mode="lines", name="Signal"), row=3, col=1)
        fig.add_trace(go.Bar(x=dfv["Date"], y=dfv["MACDhist"], name="Hist"), row=3, col=1)

fig.update_layout(
    title=f"{stock_name or 'STOCK'} — Technicals",
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis_rangeslider_visible=False,
    height=780 if rows == 3 else 620,
)

# Unique key avoids Streamlit duplicate-ID issues
st.plotly_chart(fig, use_container_width=True, key=f"qt_plot_{stock_name}_{source}")

# ---------- Signals (量化) ----------
st.markdown("### ⚙️ Signals (量化交易)")
sig_ts, sig_summary = build_signals(dfv)

if not sig_ts.empty:
    # Top badges / metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Composite (bullish)", f"{sig_summary['composite_score']} / {sig_summary['composite_max']}")
    with c2:
        st.metric(
            "Price > SMA200",
            "✅" if sig_summary["Price_gt_SMA200"] else "❌",
            help=f"↑ last: {sig_summary['last_price>200_up'] or '—'}  ·  ↓ last: {sig_summary['last_price>200_dn'] or '—'}",
        )
    with c3:
        st.metric(
            "SMA20 > SMA50",
            "✅" if sig_summary["SMA20_gt_SMA50"] else "❌",
            help=f"Golden cross last: {sig_summary['last_golden_cross'] or '—'}  ·  Death: {sig_summary['last_death_cross'] or '—'}",
        )
    with c4:
        atrpct = sig_summary["ATR_pct"]
        st.metric("ATR % of price", f"{(atrpct if pd.notna(atrpct) else 0):.2%}")

    # Table of statuses & last triggers
    rows_view = [
        ["Price > SMA200", sig_summary["Price_gt_SMA200"], sig_summary["last_price>200_up"]],
        ["SMA20 > SMA50", sig_summary["SMA20_gt_SMA50"], sig_summary["last_golden_cross"]],
        ["MACD > Signal", sig_summary["MACD_gt_Signal"], None],
        ["RSI > 50", sig_summary["RSI_gt_50"], None],
        ["ROC 12m > 0", sig_summary["ROC12m_pos"], None],
        ["52w Breakout", sig_summary["Breakout_252"], sig_summary["last_breakout_252"]],
        ["52w Breakdown", sig_summary["Breakdown_252"], sig_summary["last_breakdown_252"]],
        ["Donchian 20 Up", sig_summary["Donchian20_Up"], sig_summary["last_dc20_up"]],
        ["Donchian 20 Down", sig_summary["Donchian20_Down"], sig_summary["last_dc20_dn"]],
    ]
    view = pd.DataFrame(rows_view, columns=["Signal", "Now", "Last Trigger"])
    view["Now"] = view["Now"].map(lambda b: "✅" if bool(b) else "❌")
    st.dataframe(view, use_container_width=True, height=300)

    # Optional: show last N days of time-series (for auditing)
    with st.expander("Show recent signal time-series (audit)", expanded=False):
        n = st.slider("Rows", 50, 400, 200, 10, key="qt_sig_rows")
        st.dataframe(sig_ts.tail(n), use_container_width=True, height=240)

    # Export time-series
    csv_sig = sig_ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export signals (CSV)",
        data=csv_sig,
        file_name=f"{stock_name or 'STOCK'}_signals.csv",
        mime="text/csv",
        key=f"dl_sig_{stock_name or 'STOCK'}",
    )

st.markdown("### Save / Export")
cS, cE = st.columns([1, 1])
with cS:
    if st.button("💾 Save to OHLC store", disabled=(not stock_name)):
        try:
            if _ioh and hasattr(_ioh, "save_ohlc"):
                _ioh.save_ohlc(stock_name, df_ohlc)  # type: ignore[attr-defined]
            else:
                base = os.path.join(os.getcwd(), "data", "ohlc")
                _ensure_dir(base)
                df_ohlc.to_csv(os.path.join(base, f"{stock_name}.csv"), index=False)
            st.success(f"Saved normalized OHLC CSV for {stock_name}.")
        except Exception as e:
            st.error(f"Save failed: {e}")

with cE:
    # Export enriched indicators
    export_name = f"{stock_name or 'STOCK'}_indicators.csv"
    csv_bytes = df_ind.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export indicators CSV",
        data=csv_bytes,
        file_name=export_name,
        mime="text/csv",
        key=f"dl_ind_{stock_name or 'STOCK'}",
    )
