"""
OmniQuant — Layer 4: Decision Logic & Dashboard
================================================
Streamlit application that:
  1. Accepts a stock ticker input
  2. Runs the full data pipeline (Layer 1)
  3. Runs the AutoML Race — backtests all three models (Layer 2 + 3)
  4. Crowns the lowest-RMSE model as the winner
  5. Renders an interactive Plotly chart with confidence interval
"""

import urllib.request
urllib.request.getproxies = lambda: {}   # macOS proxy-hang fix

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta

from data_orchestrator import DataOrchestrator
from evaluator import run_backtest
from model_zoo import TreeForecaster, ForestForecaster, StatisticalForecaster

# ── Future Forecasting Helper ─────────────────────────────────────────────────

def _recompute_features(close_s: pd.Series, feature_cols: list) -> pd.DataFrame:
    """Recalculate all TA features from a Close price series."""
    tmp = pd.DataFrame({"Close": close_s})
    tmp["SMA_20"]       = ta.trend.sma_indicator(tmp["Close"], window=20)
    tmp["RSI_14"]       = ta.momentum.rsi(tmp["Close"], window=14)
    macd = ta.trend.MACD(tmp["Close"])
    tmp["MACD_12_26_9"]  = macd.macd()
    tmp["MACDh_12_26_9"] = macd.macd_diff()
    tmp["MACDs_12_26_9"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(tmp["Close"], window=20, window_dev=2)
    tmp["BBL_20_2"] = bb.bollinger_lband()
    tmp["BBM_20_2"] = bb.bollinger_mavg()
    tmp["BBU_20_2"] = bb.bollinger_hband()
    tmp["BBB_20_2"] = bb.bollinger_wband()
    tmp["BBP_20_2"] = bb.bollinger_pband()
    tmp["Log_Returns"] = np.log(tmp["Close"] / tmp["Close"].shift(1))
    
    # Handle external features like NLP Sentiment by filling with neutral value
    for col in feature_cols:
        if col not in tmp.columns:
            tmp[col] = 0.0
            
    return tmp[feature_cols]


def generate_future_forecast(
    model,
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Close",
    n_days: int = 7,
) -> pd.DataFrame:
    """
    Retrain the winning model on 100 % of available data, then generate
    an n_days out-of-sample forecast using a recursive strategy:

      For each future step t:
        1. Compute feature vector from the extended Close price history
        2. Ask the model to predict the next Close
        3. Append the predicted Close to the history and repeat

    StatisticalForecaster (ARIMA) natively supports multi-step forecast,
    so it takes the direct path without the recursive loop.

    Returns a DataFrame indexed by future business dates with columns
    ['Predicted_Close'].
    """
    from pandas.tseries.offsets import BDay

    X_full = df[feature_cols]
    y_full = df[target_col]

    # ── Retrain on 100 % of data ──────────────────────────────────────────────
    model.train(X_full, y_full)

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + BDay(1), periods=n_days)

    if isinstance(model, StatisticalForecaster):
        # ── Direct path for ARIMA ─────────────────────────────────────────────
        preds = model.predict(pd.DataFrame(index=future_dates))
        return pd.DataFrame({"Predicted_Close": preds.round(4)}, index=future_dates)

    # ── Recursive loop for Tree forecasters ────────────────────────────
    # Seed the rolling history with the real Close prices
    close_history = df[target_col].tolist()
    future_preds  = []

    for _ in range(n_days):
        # Rebuild the full Close series (real + already-predicted days)
        close_series = pd.Series(close_history, dtype=float)

        # Recompute all TA indicators on the extended series
        feature_df = _recompute_features(close_series, feature_cols)

        # Use the very last (most recent) clean feature row
        last_row = feature_df.dropna().iloc[[-1]]  # shape (1, n_features)

        # Predict the next closing price
        pred_close = float(model.predict(last_row)[0])
        future_preds.append(round(pred_close, 4))

        # Extend history so the next iteration's indicators include this step
        close_history.append(pred_close)

    return pd.DataFrame({"Predicted_Close": future_preds}, index=future_dates)



@st.cache_data(ttl=3600)
def load_data(ticker: str, start_str: str, end_str: str, display_currency: str) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Cached data pipeline — downloads OHLCV data and engineers all TA features.
    Result is stored in Streamlit's memory cache for 1 hour (ttl=3600 s).
    Subsequent UI interactions reuse the cached DataFrame instantly.
    """
    orch = DataOrchestrator()
    return orch.process(ticker, start_str, end_str, display_currency)

st.set_page_config(
    page_title="OmniQuant · AutoML Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 25, 35, 0.95);
        border-right: 1px solid rgba(0, 230, 180, 0.15);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(0, 230, 180, 0.05);
        border: 1px solid rgba(0, 230, 180, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"]  { color: #00e6b4 !important; font-weight: 700; }
    [data-testid="stMetricLabel"]  { color: #8b9ab0 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00e6b4, #0099ff);
        color: #0d1117;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        letter-spacing: 0.5px;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, rgba(0,230,180,0.12), rgba(0,153,255,0.12));
        border: 1px solid rgba(0,230,180,0.35);
        border-radius: 14px;
        padding: 20px 28px;
        margin: 16px 0;
    }
    .winner-title { color: #00e6b4; font-size: 0.78rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }
    .winner-name  { color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0; }
    .winner-sub   { color: #8b9ab0; font-size: 0.88rem; margin-top: 4px; }

    /* Race table */
    .race-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #c9d1d9;
        font-size: 0.9rem;
    }
    .race-row:last-child { border-bottom: none; }
    .race-winner { color: #00e6b4; font-weight: 700; }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 OmniQuant")
    st.markdown("*AutoML Ensemble for Financial Forecasting*")
    st.markdown("---")

    ticker   = st.text_input("Stock Ticker", value="AAPL", max_chars=20).strip().upper()
    lookback = st.slider("Lookback Period (days)", min_value=365, max_value=1095,
                         value=730, step=30, help="Historical data used for feature engineering")

    st.markdown("---")
    st.markdown("**Models in the Race**")
    st.markdown("🌲 TreeForecaster *(XGBoost)*")
    st.markdown("🌳 ForestForecaster *(RandomForest)*")
    st.markdown("📈 StatisticalForecaster *(ARIMA)*")
    st.markdown("---")

    display_currency = st.selectbox(
        "Display Currency",
        options=["Native", "USD", "INR", "EUR", "GBP", "JPY"],
        index=0
    )

    n_future_days = st.slider("Future Forecast Days", min_value=3, max_value=14,
                               value=7, step=1)
    run_btn = st.button("🚀  Run Pipeline", use_container_width=True)
    st.markdown("---")
    st.caption("OmniQuant v0.5 · Layer 4 + Future Forecast")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 OmniQuant ")
st.markdown("### Automated Machine Learning Ensemble for Financial Forecasting")
st.markdown("---")

if not run_btn and "results" not in st.session_state:
    # ── Landing state ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Step 1 · Input**\n\nEnter a ticker in the sidebar and choose a lookback period.")
    col2.info("**Step 2 · AutoML Race**\n\nRuns XGBoost, RandomForest, and ARIMA models simultaneously.")
    col3.info("**Step 3 · Backtest**\n\nEvaluates models over 30 days; the one with lowest RMSE is crowned winner.")
    col4.info("**Step 4 · Future**\n\nThe winner is retrained on 100 % data to forecast the next N business days.")
    st.stop()

# ── Pipeline (only runs when the button is clicked) ───────────────────────────
if run_btn:
    from datetime import datetime, timedelta
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=lookback)
    end_str  = end_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%Y-%m-%d")

    # 1. Data Orchestration
    with st.spinner(f"⬇️  Fetching {ticker} data ({start_str} → {end_str})…"):
        try:
            df, currency_code, latest_headlines = load_data(ticker, start_str, end_str, display_currency)
        except Exception as exc:
            st.error(f"❌ Data fetch failed for **{ticker}**: {exc}")
            st.stop()

    if df.empty or len(df) < 60:
        st.error(f"❌ Not enough data for **{ticker}**. Try a different ticker or extend the lookback period.")
        st.stop()

    # 2. Feature / Target split
    OHLCV_COLS   = ["Open", "High", "Low", "Close", "Volume"]
    TARGET_COL   = "Close"
    feature_cols = [c for c in df.columns if c not in OHLCV_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 3. AutoML Race
    zoo = {
        "TreeForecaster (XGBoost)":        TreeForecaster(),
        "ForestForecaster (RandomForest)":  ForestForecaster(),
        "StatisticalForecaster (ARIMA)":    StatisticalForecaster(),
    }
    results = {}
    status_ph = st.empty()
    for name, model_inst in zoo.items():
        status_ph.info(f"⚙️  AutoML Race: Training **{name}**…")
        try:
            results[name] = run_backtest(model_inst, X, y)
        except Exception as exc:
            st.warning(f"⚠️ {name} failed: {exc}")

    if not results:
        st.error("❌ All models in the zoo failed. See logs for details.")
        st.stop()
    status_ph.empty()

    winner_name = min(results, key=lambda k: results[k]["rmse"])
    winner_res  = results[winner_name]
    model       = zoo[winner_name]

    # 4. Future Forecast
    with st.spinner(f"⏳  Running {n_future_days}-day forecast…"):
        try:
            if isinstance(model, TreeForecaster):
                model_clone = TreeForecaster()
            elif isinstance(model, ForestForecaster):
                model_clone = ForestForecaster()
            else:
                model_clone = StatisticalForecaster()
            future_df = generate_future_forecast(
                model=model_clone, model_name=winner_name,
                df=df, feature_cols=feature_cols,
                target_col=TARGET_COL, n_days=n_future_days,
            )
            forecast_ok = True
        except Exception as exc:
            st.error(f"❌ Future forecast failed: {exc}")
            forecast_ok = False

    # ── Persist everything to session_state ───────────────────────────────────
    st.session_state["df"]               = df
    st.session_state["currency_code"]    = currency_code
    st.session_state["latest_headlines"] = latest_headlines
    st.session_state["feature_cols"]     = feature_cols
    st.session_state["results"]          = results
    st.session_state["winner_name"]      = winner_name
    st.session_state["winner_res"]       = winner_res
    st.session_state["model"]            = model
    st.session_state["future_df"]        = future_df if forecast_ok else None
    st.session_state["ticker_ran"]       = ticker
    st.session_state["n_future_days"]    = n_future_days

    st.toast(f"Pipeline complete! {winner_name} won the race.", icon="🚀")

# ── UI (renders from session_state — survives widget interactions) ─────────────
if "results" not in st.session_state:
    st.stop()

# Pull everything out of state
df               = st.session_state["df"]
currency_code    = st.session_state["currency_code"]
latest_headlines = st.session_state["latest_headlines"]
feature_cols     = st.session_state["feature_cols"]
results          = st.session_state["results"]
winner_name      = st.session_state["winner_name"]
winner_res       = st.session_state["winner_res"]
model            = st.session_state["model"]
future_df        = st.session_state["future_df"]
ticker_ran       = st.session_state["ticker_ran"]
n_future_days    = st.session_state["n_future_days"]

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL = "Close"

currency_symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}
curr_sym = currency_symbols.get(currency_code, currency_code + " ")

st.success(f"✅ {ticker_ran} — {len(df)} trading-day rows loaded  |  {df.shape[1]} features engineered")

# ── Winner Banner ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="winner-banner">
        <p class="winner-title">🏆 AUTO-ML WINNER</p>
        <p class="winner-name">{winner_name}</p>
        <p class="winner-sub">Lowest Root Mean Squared Error (RMSE) across 3-model backtest.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Metrics Row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ticker",       ticker_ran)
m2.metric("Best Engine",  winner_name.split(" (")[0])
m3.metric("Winner RMSE",  f"{winner_res['rmse']:.4f}")
m4.metric("Winner MAPE",  f"{winner_res['mape']:.4f} %")

# Sentiment
latest_sentiment = df["Sentiment"].iloc[-1] if "Sentiment" in df.columns else 0.0
if latest_sentiment > 0.05:
    s_label = "Bullish 🟢"
elif latest_sentiment < -0.05:
    s_label = "Bearish 🔴"
else:
    s_label = "Neutral ⚪"
st.info(f"**📰 Current News Sentiment:** {s_label} (Score: {latest_sentiment:.2f})")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "🏎️ Leaderboard", "📊 Raw Data"])

# ── TAB 1 — Backtest Chart + Future Forecast ──────────────────────────────────
with tab1:
    st.markdown(f"#### 📉 30-Day Backtest · {winner_name}")

    col_chart, col_overlay = st.columns([1, 2])
    with col_chart:
        chart_type = st.selectbox(
            "Chart Style",
            ["Candlestick", "Line", "OHLC", "Area"],
            index=0,
        )
    with col_overlay:
        overlays = st.multiselect(
            "Chart Overlays",
            ["SMA 20", "SMA 50"],
            default=[],
            help="Overlay Simple Moving Averages on the backtest chart.",
        )

    traj      = winner_res["trajectory"]
    bt_dates  = traj.index
    actuals   = traj["Actual"].values
    preds     = traj["Predicted"].values
    residuals = preds - actuals
    sigma = np.std(residuals)
    upper = preds + 1.96 * sigma
    lower = preds - 1.96 * sigma

    ohlc_bt = df[["Open", "High", "Low", "Close"]].loc[df.index.isin(bt_dates)]

    # ── Charting SMAs (pandas rolling — no ta-lib name collision) ─────────────
    sma20 = df["Close"].rolling(window=20).mean()
    sma50 = df["Close"].rolling(window=50).mean()

    fig = go.Figure()

    # ── Dynamic historical trace ──────────────────────────────────────────────
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=ohlc_bt.index, open=ohlc_bt["Open"], high=ohlc_bt["High"],
            low=ohlc_bt["Low"], close=ohlc_bt["Close"],
            name="OHLC (Actual)",
            increasing=dict(line=dict(color="#00e6b4"), fillcolor="rgba(0,230,180,0.5)"),
            decreasing=dict(line=dict(color="#ff4c6a"), fillcolor="rgba(255,76,106,0.5)"),
            hoverinfo="x+y",
        ))
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=ohlc_bt.index, open=ohlc_bt["Open"], high=ohlc_bt["High"],
            low=ohlc_bt["Low"], close=ohlc_bt["Close"],
            name="OHLC (Actual)",
            increasing=dict(line=dict(color="#00e6b4")),
            decreasing=dict(line=dict(color="#ff4c6a")),
        ))
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=bt_dates, y=actuals, mode="lines",
            name="Historical Close",
            line=dict(color="#0099ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,153,255,0.10)",
            hovertemplate=f"<b>Close</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ))
    else:  # Line
        fig.add_trace(go.Scatter(
            x=bt_dates, y=actuals, mode="lines",
            name="Historical Close",
            line=dict(color="#ffffff", width=2),
            hovertemplate=f"<b>Close</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ))

    # ── SMA Overlays (user-selected) ──────────────────────────────────────────
    sma20_bt = sma20.loc[sma20.index.isin(bt_dates)]
    sma50_bt = sma50.loc[sma50.index.isin(bt_dates)]

    if "SMA 20" in overlays:
        fig.add_trace(go.Scatter(
            x=sma20_bt.index, y=sma20_bt.values,
            mode="lines", name="SMA 20",
            line=dict(color="orange", width=2),
            hovertemplate=f"<b>SMA 20</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ))
    if "SMA 50" in overlays:
        fig.add_trace(go.Scatter(
            x=sma50_bt.index, y=sma50_bt.values,
            mode="lines", name="SMA 50",
            line=dict(color="#6fa8ff", width=2),
            hovertemplate=f"<b>SMA 50</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ))

    # ── CI band + AI prediction (always present) ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([bt_dates, bt_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(0,153,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=bt_dates, y=upper, mode="lines",
        line=dict(color="rgba(0,153,255,0.40)", width=1, dash="dot"),
        name="Upper CI", hovertemplate=f"Upper CI: {curr_sym}%{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bt_dates, y=lower, mode="lines",
        line=dict(color="rgba(0,153,255,0.40)", width=1, dash="dot"),
        name="Lower CI", hovertemplate=f"Lower CI: {curr_sym}%{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=bt_dates, y=preds, mode="lines+markers", name="AI Prediction",
        line=dict(color="#00e6b4", width=2.5, dash="dash"),
        marker=dict(size=5, color="#00e6b4", symbol="diamond"),
        hovertemplate=f"<b>Predicted</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.06)",
                   showline=True, linecolor="rgba(255,255,255,0.15)",
                   rangeslider=dict(visible=True, bgcolor="rgba(13,17,23,0.8)", thickness=0.05)),
        yaxis=dict(title=f"{ticker_ran} Price ({currency_code})",
                   gridcolor="rgba(255,255,255,0.06)",
                   showline=True, linecolor="rgba(255,255,255,0.15)",
                   tickprefix=curr_sym),
        hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0), height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Backtest table
    with st.expander("📋  View 30-Day Backtest Data"):
        display_df = traj.copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df.columns = [f"Actual ({curr_sym})", f"Predicted ({curr_sym})", f"Error ({curr_sym})"]
        st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)

    # Feature Importance
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        with st.expander(f"🌲  Feature Importance — {winner_name.split(' (')[0]}"):
            import plotly.express as px
            fi = pd.Series(model.model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            fi_fig = px.bar(fi, orientation="h",
                            title=f"{winner_name.split(' (')[0]} Feature Importances",
                            labels={"value": "Importance", "index": "Feature"},
                            color=fi.values,
                            color_continuous_scale=[[0, "#0099ff"], [1, "#00e6b4"]])
            fi_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(13,17,23,0.8)",
                                 showlegend=False, coloraxis_showscale=False,
                                 height=360, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown("---")

    # ── Future Forecast Chart ─────────────────────────────────────────────────
    st.markdown(f"## 🔮 The Future: Next {n_future_days} Days Forecast")
    st.markdown(
        f"The **{winner_name}** engine is retrained on **100 % of all available data** "
        f"({len(df)} trading days) and projecting the next **{n_future_days} business days**."
    )

    if future_df is not None:
        last_actual_date  = df.index[-1]
        last_actual_price = float(df[TARGET_COL].iloc[-1])
        ohlc_hist = df[["Open", "High", "Low", "Close"]].iloc[-30:]
        fut_dates  = future_df.index
        fut_sigma  = np.std(winner_res["trajectory"]["Error"].values)
        fut_upper  = future_df["Predicted_Close"].values + 1.96 * fut_sigma
        fut_lower  = future_df["Predicted_Close"].values - 1.96 * fut_sigma

        fut_fig = go.Figure()

        # History context — mirrors the selected chart style
        if chart_type == "Candlestick":
            fut_fig.add_trace(go.Candlestick(
                x=ohlc_hist.index, open=ohlc_hist["Open"], high=ohlc_hist["High"],
                low=ohlc_hist["Low"], close=ohlc_hist["Close"],
                name="Recent History (30d)",
                increasing=dict(line=dict(color="rgba(0,230,180,0.6)"),  fillcolor="rgba(0,230,180,0.25)"),
                decreasing=dict(line=dict(color="rgba(255,76,106,0.6)"), fillcolor="rgba(255,76,106,0.25)"),
            ))
        elif chart_type == "OHLC":
            fut_fig.add_trace(go.Ohlc(
                x=ohlc_hist.index, open=ohlc_hist["Open"], high=ohlc_hist["High"],
                low=ohlc_hist["Low"], close=ohlc_hist["Close"],
                name="Recent History (30d)",
                increasing=dict(line=dict(color="rgba(0,230,180,0.6)")),
                decreasing=dict(line=dict(color="rgba(255,76,106,0.6)")),
            ))
        elif chart_type == "Area":
            fut_fig.add_trace(go.Scatter(
                x=ohlc_hist.index, y=ohlc_hist["Close"], mode="lines",
                name="Recent History (30d)",
                line=dict(color="#0099ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,153,255,0.10)",
            ))
        else:
            fut_fig.add_trace(go.Scatter(
                x=ohlc_hist.index, y=ohlc_hist["Close"], mode="lines",
                name="Recent History (30d)",
                line=dict(color="rgba(255,255,255,0.5)", width=1.8),
            ))

        # Bridge + CI + forecast line
        fut_fig.add_trace(go.Scatter(
            x=[last_actual_date, fut_dates[0]],
            y=[last_actual_price, future_df["Predicted_Close"].iloc[0]],
            mode="lines", name="_bridge", showlegend=False,
            line=dict(color="#00e6b4", width=2, dash="dot"), hoverinfo="skip",
        ))
        fut_fig.add_trace(go.Scatter(
            x=np.concatenate([fut_dates, fut_dates[::-1]]),
            y=np.concatenate([fut_upper, fut_lower[::-1]]),
            fill="toself", fillcolor="rgba(0,230,180,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip",
        ))
        fut_fig.add_trace(go.Scatter(x=fut_dates, y=fut_upper, mode="lines",
            line=dict(color="rgba(0,230,180,0.30)", width=1, dash="dot"), name="Upper CI",
            hovertemplate=f"Upper CI: {curr_sym}%{{y:.2f}}<extra></extra>"))
        fut_fig.add_trace(go.Scatter(x=fut_dates, y=fut_lower, mode="lines",
            line=dict(color="rgba(0,230,180,0.30)", width=1, dash="dot"), name="Lower CI",
            hovertemplate=f"Lower CI: {curr_sym}%{{y:.2f}}<extra></extra>"))
        fut_fig.add_trace(go.Scatter(
            x=fut_dates, y=future_df["Predicted_Close"].values,
            mode="lines+markers", name=f"Forecast ({n_future_days}d)",
            line=dict(color="#00e6b4", width=3),
            marker=dict(size=9, color="#00e6b4", symbol="diamond", line=dict(color="#0d1117", width=1.5)),
            hovertemplate=f"<b>Forecast</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ))

        fut_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
            font=dict(family="Inter, sans-serif", color="#c9d1d9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.06)",
                       showline=True, linecolor="rgba(255,255,255,0.15)",
                       rangeslider=dict(visible=True, bgcolor="rgba(13,17,23,0.8)", thickness=0.05)),
            yaxis=dict(title=f"{ticker_ran} Projected Close ({currency_code})",
                       gridcolor="rgba(255,255,255,0.06)",
                       showline=True, linecolor="rgba(255,255,255,0.15)",
                       tickprefix=curr_sym),
            hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0), height=500,
        )
        st.plotly_chart(fut_fig, use_container_width=True)

        # Future price table
        st.markdown("##### 📅 Projected Prices — Next Trading Days")
        table_df = future_df.copy()
        table_df.index = table_df.index.strftime("%A, %b %d %Y")
        table_df.columns = [f"Projected Close ({curr_sym})"]
        table_df[f"Δ vs Today ({curr_sym})"] = (future_df["Predicted_Close"].values - last_actual_price).round(4)
        table_df["Δ vs Today (%)"] = ((future_df["Predicted_Close"].values - last_actual_price) / last_actual_price * 100).round(4)
        st.dataframe(
            table_df.style.format({
                f"Projected Close ({curr_sym})": f"{curr_sym}{{:.4f}}",
                f"Δ vs Today ({curr_sym})"     : "{:+.4f}",
                "Δ vs Today (%)"               : "{:+.4f} %",
            }),
            use_container_width=True,
        )
        st.caption(
            "⚠️ **Disclaimer**: Forecasts are generated by an ML model trained on historical data and are "
            "for educational purposes only. They do not constitute financial advice."
        )
    else:
        st.warning("⚠️ Future forecast could not be generated for this run.")

# ── TAB 2 — Leaderboard + Sentiment ──────────────────────────────────────────
with tab2:
    st.markdown("#### 🏎️ AutoML Race Leaderboard")
    leaderboard_data = []
    for name, res in results.items():
        is_winner = (name == winner_name)
        leaderboard_data.append({
            "Model": f"{'✅ ' if is_winner else ''}{name}",
            "RMSE":  round(res["rmse"], 4),
            "MAPE":  f"{res['mape']:.4f} %",
        })
    st.table(pd.DataFrame(leaderboard_data))

    st.markdown("---")
    st.markdown("#### 📰 NLP News Sentiment")
    st.info(f"**Current Sentiment:** {s_label} (Score: {latest_sentiment:.2f})")
    with st.expander("🔍 View Latest Headlines"):
        if not latest_headlines:
            st.warning("No recent news found on Yahoo Finance for this ticker.")
        else:
            for hl in latest_headlines:
                st.write(f"• {hl}")

# ── TAB 3 — Raw Data ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📊 Raw Feature Data (Last 100 Rows)")
    st.caption("The exact data matrix used to train the AutoML models, including all TA indicators and Sentiment score.")
    st.dataframe(df.tail(100), use_container_width=True)

st.markdown("---")
st.caption("OmniQuant v0.7 · Built with Streamlit & Plotly · Data via Yahoo Finance")

