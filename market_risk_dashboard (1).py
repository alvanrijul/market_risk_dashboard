"""
Market Risk Dashboard
----------------------
Interactive Streamlit app for stock performance & risk analytics:
returns, volatility, VaR, CVaR (Expected Shortfall), and drawdown.

Run locally:
    pip install streamlit yfinance pandas numpy scipy plotly
    streamlit run market_risk_dashboard.py

Deploy free (recommended for a portfolio link):
    1. Push this file (+ a requirements.txt) to a public GitHub repo
    2. Go to https://share.streamlit.io -> "New app" -> point at the repo/file
    3. You get a shareable https://<name>.streamlit.app link
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Risk Dashboard", layout="wide")

# ----------------------------- Sidebar controls -----------------------------
st.sidebar.header("Controls")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

default_start = datetime.date(2016, 4, 30)
default_end = datetime.date.today()

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
    min_value=datetime.date(1990, 1, 1),
    max_value=default_end,
)
start_date, end_date = date_range if len(date_range) == 2 else (default_start, default_end)

confidence = st.sidebar.select_slider(
    "VaR / CVaR confidence level",
    options=[0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x * 100)}%",
)

var_method = st.sidebar.radio(
    "VaR method",
    options=["Historical", "Parametric (Normal)"],
    index=0,
)

st.sidebar.caption("Data source: Yahoo Finance via yfinance")

# ----------------------------- Data loading -----------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tkr: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    df = yf.download(tkr, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


st.title("📊 Market Risk Dashboard")
st.caption(f"{ticker} · {start_date} to {end_date}")

with st.spinner(f"Fetching {ticker}..."):
    data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error(f"No data found for '{ticker}'. Check the ticker symbol and date range.")
    st.stop()

data = data.copy()
data["returns"] = data["Close"].pct_change()
returns = data["returns"].dropna()

if returns.empty:
    st.warning("Not enough data points to compute risk metrics for this range.")
    st.stop()

# ----------------------------- Metric calculations -----------------------------

cum_returns = (1 + returns).cumprod() - 1
annual_return = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)

alpha = 1 - confidence

if var_method == "Historical":
    var = np.percentile(returns, 100 * alpha)
else:
    from scipy.stats import norm

    var = returns.mean() + returns.std() * norm.ppf(alpha)

cvar = returns[returns <= var].mean()
if pd.isna(cvar):
    cvar = var  # fallback if no observations breach VaR (e.g. very high confidence, short window)

wealth_index = (1 + returns).cumprod()
rolling_max = wealth_index.cummax()
drawdown = (wealth_index - rolling_max) / rolling_max
max_drawdown = drawdown.min()
max_dd_date = drawdown.idxmin()

# ----------------------------- KPI row -----------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annualized Return", f"{annual_return:.2%}")
c2.metric("Annualized Volatility", f"{annual_vol:.2%}")
c3.metric(f"VaR ({int(confidence*100)}%, daily)", f"{var:.2%}")
c4.metric(f"CVaR ({int(confidence*100)}%, daily)", f"{cvar:.2%}")
c5.metric("Max Drawdown", f"{max_drawdown:.2%}", help=f"Trough on {max_dd_date.date()}")

st.divider()

# ----------------------------- Charts -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
    fig_price.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("Cumulative Returns")
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(x=cum_returns.index, y=cum_returns * 100, mode="lines", name="Cumulative Return (%)")
    )
    fig_cum.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, yaxis_title="%")
    st.plotly_chart(fig_cum, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown (%)",
            line=dict(color="crimson"),
        )
    )
    fig_dd.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, yaxis_title="%")
    st.plotly_chart(fig_dd, use_container_width=True)

with col4:
    st.subheader("Return Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=returns * 100, nbinsx=80, name="Daily Returns (%)"))
    fig_hist.add_vline(x=var * 100, line_color="orange", line_dash="dash",
                        annotation_text=f"VaR {int(confidence*100)}%", annotation_position="top")
    fig_hist.add_vline(x=cvar * 100, line_color="crimson", line_dash="dash",
                        annotation_text=f"CVaR {int(confidence*100)}%", annotation_position="bottom")
    fig_hist.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, xaxis_title="Daily return (%)")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

with st.expander("Show raw price data"):
    st.dataframe(data.tail(200), use_container_width=True)

st.caption(
    "VaR/CVaR are computed on daily returns and are not annualized. "
    "Historical VaR uses the empirical percentile of past returns; "
    "Parametric VaR assumes normally distributed returns."
)
