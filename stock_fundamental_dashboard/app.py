import streamlit as st

st.set_page_config(page_title="Stock Fundamental Dashboard", layout="wide", page_icon="📈")

# ---------- Global CSS ----------
BASE_CSS = """
<style>
html, body, [class*="css"] { font-size: 16px !important; }
h1, h2, h3, h4 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: .2px; }
p, label, span, div { color: #0b132a; }
.stApp {
  background: radial-gradient(1100px 600px at 18% -10%, #f7fbff 0%, #eef4ff 45%, #ffffff 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #1f2937 100%) !important;
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.stTabs [role="tab"] { font-size: 15px !important; font-weight: 600 !important; }
.stDataFrame, .stDataEditor, .dataframe { font-size: 15px !important; }
div[data-baseweb="input"] input, textarea, .stNumberInput input { font-size: 15px !important; }
.stButton>button { border-radius: 12px !important; padding: .6rem 1.5rem !important; font-weight: 600; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------- Hide Streamlit's default sidebar nav ----------
HIDE_DEFAULT_NAV_CSS = """
<style>
/* Hide the built-in multipage page list */
section[data-testid="stSidebarNav"] { display: none !important; }
</style>
"""
st.markdown(HIDE_DEFAULT_NAV_CSS, unsafe_allow_html=True)

# ---------- Landing ----------
st.title("📈 Stock Fundamental Dashboard")
st.markdown(
    """
Welcome to the **Stock Fundamental Dashboard**!  
Manage, analyze and visualize your stock financial data.

- **📊 Dashboard** — overview, latest annual/quarter summaries.
- **✏️ Add / Edit** — enter annual & quarterly fundamentals.
- **🔍 View Stock** — deep-dive ratios, charts, details per stock.
- **🧭 Systematic Decision** — evaluate/value-investing funnel.
- **📐 Risk / Reward Planner** — entries, stops, targets, size.
- **🧾 Queue Audit Log** — every add/update/delete.
- **📈 Ongoing Trades** & **📘 Trade History** — manage PnL.
- **⚡ Momentum Data** — import/attach OHLC CSVs.
- **🧪 Quant Tech & Signals** — candlesticks + indicators (SMA/EMA/RSI/MACD/Bollinger/ATR) + signal panel & CSV export.

---
"""
)
st.info("Choose a page from the sidebar to begin.")

# ---------- Custom Sidebar Navigation ----------
st.sidebar.title("Navigation")

try:
    # Fundamentals
    st.sidebar.subheader("Fundamentals")
    st.sidebar.page_link("pages/1_Dashboard.py",           label="📊 Dashboard")
    st.sidebar.page_link("pages/2_Add_or_Edit.py",         label="✏️ Add / Edit")
    st.sidebar.page_link("pages/3_View_Stock.py",          label="🔍 View Stock")
    st.sidebar.page_link("pages/4_Systematic_Decision.py", label="🧭 Systematic Decision")

    st.sidebar.divider()

    # Trading & Logs
    st.sidebar.subheader("Trading & Logs")
    st.sidebar.page_link("pages/5_Risk_Reward_Planner.py", label="📐 Risk / Reward Planner")
    st.sidebar.page_link("pages/6_Queue_Audit_Log.py",     label="🧾 Queue Audit Log")
    st.sidebar.page_link("pages/7_Ongoing_Trades.py",      label="📈 Ongoing Trades")
    st.sidebar.page_link("pages/8_Trade_History.py",       label="📘 Trade History")

    st.sidebar.divider()

    # Momentum & Quant
    st.sidebar.subheader("Momentum & Quant")
    st.sidebar.page_link("pages/9_Momentum_Data.py",       label="⚡ Momentum Data")
    st.sidebar.page_link("pages/10_Quant_Tech_Charts.py",  label="🧪 Quant Tech & Signals")

except Exception:
    # Fallback for older Streamlit versions (no page_link)
    st.sidebar.markdown(
        """
**Fundamentals**
- [📊 Dashboard](./pages/1_Dashboard.py)
- [✏️ Add / Edit](./pages/2_Add_or_Edit.py)
- [🔍 View Stock](./pages/3_View_Stock.py)
- [🧭 Systematic Decision](./pages/4_Systematic_Decision.py)

**Trading & Logs**
- [📐 Risk / Reward Planner](./pages/5_Risk_Reward_Planner.py)
- [🧾 Queue Audit Log](./pages/6_Queue_Audit_Log.py)
- [📈 Ongoing Trades](./pages/7_Ongoing_Trades.py)
- [📘 Trade History](./pages/8_Trade_History.py)

**Momentum & Quant**
- [⚡ Momentum Data](./pages/9_Momentum_Data.py)
- [🧪 Quant Tech & Signals](./pages/10_Quant_Tech_Charts.py)
"""
    )




