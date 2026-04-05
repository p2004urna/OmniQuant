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

import pytz
from datetime import datetime, timedelta
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas_ta as ta

from data_orchestrator import DataOrchestrator
from evaluator import run_backtest
from model_zoo import TreeForecaster, ForestForecaster, StatisticalForecaster, TimeExpert
from model_trainer import MetaForecaster

# ── Future Forecasting Helper ─────────────────────────────────────────────────

def _recompute_features(
    close_s: pd.Series,
    feature_cols: list,
    last_known: dict | None = None,
) -> pd.DataFrame:
    """Recalculate all TA features from a Close price series.

    Args:
        close_s:      Full simulated close-price history.
        feature_cols: The exact columns the model was trained on.
        last_known:   Dict of {col: value} from the last historical row —
                      used as a fallback when the series is too short for
                      an indicator to produce a value (e.g. SMA_200 needs
                      200 rows).
    """
    if last_known is None:
        last_known = {}

    tmp = pd.DataFrame({"Close": close_s})

    tmp["Open"]   = tmp["Close"]
    tmp["High"]   = tmp["Close"]
    tmp["Low"]    = tmp["Close"]
    tmp["Volume"] = 1_000_000

    # Core TA indicators (existing)
    tmp.ta.bbands(length=20, std=2, append=True)
    tmp.ta.atr(length=14, append=True)
    tmp.ta.rsi(length=14, append=True)
    tmp.ta.macd(fast=12, slow=26, signal=9, append=True)
    tmp.ta.obv(append=True)

    # Explicit SMA / RSI columns matching the training schema
    tmp["SMA_50"]  = ta.sma(close=tmp["Close"], length=50)
    tmp["SMA_200"] = ta.sma(close=tmp["Close"], length=200)
    tmp["RSI_14"]  = ta.rsi(close=tmp["Close"], length=14)

    tmp["Log_Returns"] = np.log(tmp["Close"] / tmp["Close"].shift(1))

    # Fill any column the model expects but is missing or NaN
    for col in feature_cols:
        if col not in tmp.columns:
            tmp[col] = last_known.get(col, 0.0)
        elif tmp[col].isna().any() and col in last_known:
            tmp[col] = tmp[col].fillna(last_known[col])

    return tmp[feature_cols]


def generate_future_forecast(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_inference: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Close",
    n_days: int = 7,
) -> pd.DataFrame:
    from pandas.tseries.offsets import BDay
    from model_zoo import TimeExpert
    from model_trainer import MetaForecaster

    # ── Snapshot of the last known indicator values (trend context) ────────
    last_known = {}
    for col in feature_cols:
        if col in X_train.columns:
            last_known[col] = float(X_train[col].iloc[-1])

    last_date = X_train.index[-1] + BDay(1)
    if not X_inference.empty and X_inference.index[0] > X_train.index[-1]:
        last_date = X_inference.index[0]

    future_dates = pd.bdate_range(start=last_date, periods=n_days)

    xgb_future_preds = []

    # ── Day 1: predict from X_inference, injecting missing indicator cols ─
    curr_inf_row = X_inference.copy()
    for col in feature_cols:
        if col not in curr_inf_row.columns:
            curr_inf_row[col] = last_known.get(col, 0.0)
    curr_inf_row = curr_inf_row[feature_cols]

    xgb_pred_day1 = float(model.predict(curr_inf_row)[0])
    xgb_future_preds.append(round(xgb_pred_day1, 4))

    # ── Days 2‥N: iterative forecast using recomputed features ────────────
    close_history = X_train[target_col].tolist()
    close_history.append(xgb_pred_day1)

    for _ in range(1, n_days):
        close_series = pd.Series(close_history, dtype=float)
        feature_df = _recompute_features(close_series, feature_cols, last_known)
        last_row = feature_df.dropna().iloc[[-1]]
        pred_close = float(model.predict(last_row)[0])
        xgb_future_preds.append(round(pred_close, 4))
        close_history.append(pred_close)

    # ── Prophet predictions ───────────────────────────────────────────────
    time_model = TimeExpert()
    prophet_preds = time_model.forecast(X_train, horizon=n_days)

    # ── Ensemble ──────────────────────────────────────────────────────────
    meta = MetaForecaster()
    weights_used = meta._get_volatility_weights(X_train)

    valid_keys = ['xgboost', 'prophet']
    final_weights = meta._redistribute_weights(weights_used, valid_keys)
    w_xgb = final_weights['xgboost']
    w_pro = final_weights['prophet']

    ensemble_preds = []
    for i in range(n_days):
        ensemble_price = xgb_future_preds[i] * w_xgb + prophet_preds[i] * w_pro
        ensemble_preds.append(round(float(ensemble_price), 4))

    return pd.DataFrame({"Predicted_Close": ensemble_preds}, index=future_dates[:n_days])



@st.cache_data(ttl=3600)
def get_bulletproof_data(
    ticker: str, days_back: int, display_currency: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str, list]:
    ist        = pytz.timezone("Asia/Kolkata")
    now_india  = datetime.now(ist)
    start_india = now_india - timedelta(days=days_back)
    end_str    = now_india.strftime("%Y-%m-%d")
    start_str  = start_india.strftime("%Y-%m-%d")

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
    st.markdown("**The AI Council (Ensemble)**")
    st.markdown("🌲 TreeExpert *(XGBoost)*")
    st.markdown("⏱️ TimeExpert *(Prophet)*")
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
    st.caption("OmniQuant v0.8 · Ensembles + AutoML")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 OmniQuant ")
st.markdown("### Automated Machine Learning Ensemble for Financial Forecasting")
st.markdown("---")

if not run_btn and "results" not in st.session_state:
    # ── Landing state ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Step 1 · Input**\n\nEnter a ticker in the sidebar and choose a lookback period.")
    col2.info("**Step 2 · AutoML Race**\n\nRuns XGBoost, RandomForest, and ARIMA models simultaneously.")
    col3.info("**Step 3 · Backtest & Meta-Ensemble**\n\nEvaluates RMSE for the race and generates Volatility-Weighted Ensembles.")
    col4.info("**Step 4 · Future**\n\nThe winner is retrained on 100 % data to forecast the next N business days.")
    st.stop()

# ── Pipeline (only runs when the button is clicked) ───────────────────────────
if run_btn:
    gc.collect()
    # 1. Data Orchestration
    with st.spinner(f"⬇️  Fetching {ticker} data via Bulletproof Engine …"):
        try:
            X_train, y_train, X_inference, currency_code, latest_headlines = get_bulletproof_data(
                ticker, lookback, display_currency
            )
            df = X_train # Chronological data for charting
            
            df['SMA_50'] = ta.sma(close=df['Close'], length=50)
            df['SMA_200'] = ta.sma(close=df['Close'], length=200)
            df['RSI_14'] = ta.rsi(close=df['Close'], length=14)
            df.dropna(inplace=True)
            y_train = y_train.loc[df.index]
            
            print(df[['Close', 'SMA_50', 'SMA_200', 'RSI_14']].tail())
            
            current_sentiment = float(X_inference.get('Sentiment_Score', pd.Series([0.0])).iloc[-1])
        except Exception as exc:
            st.error(f"❌ Data fetch failed for **{ticker}**: {exc}")
            st.stop()

    if df is None or df.empty:
        st.warning(
            f"⚠️ Data fetch failed for **{ticker}**. "
            "Please check the symbol or try again later."
        )
        st.stop()

    if len(df) < 60:
        st.warning(
            f"⚠️ Not enough trading data for **{ticker}** "
            "(fewer than 60 rows). Try extending the lookback period."
        )
        st.stop()

    # 2. Feature / Target split
    TARGET_COL   = "Close"
    
    ta_patterns = ("RSI_", "BBL_", "BBM_", "BBU_", "BBB_", "BBP_", "MACD", "ATR", "OBV", "SMA_")
    feature_cols = [
        c for c in df.columns 
        if c != TARGET_COL and c != "Target_Next_Close" and (c.startswith(ta_patterns) or c in ["Log_Returns", "Sentiment_Score", "Volume"])
    ]
    
    # Guarantee our TA indicators are always visible to the AI
    for indicator in ["SMA_50", "SMA_200", "RSI_14"]:
        if indicator in df.columns and indicator not in feature_cols:
            feature_cols.append(indicator)
    
    X = df[feature_cols]
    y = y_train

    # 3. AutoML Race
    zoo = {
        "TreeForecaster (XGBoost)":        TreeForecaster(),
        # "ForestForecaster (RandomForest)":  ForestForecaster(),
        # "StatisticalForecaster (ARIMA)":    StatisticalForecaster(),
    }
    results = {}
    status_ph = st.empty()
    for name, model_inst in zoo.items():
        status_ph.info(f"⚙️  AutoML Race: Training **{name}**…")
        try:
            if isinstance(model_inst, TreeForecaster):
                with st.spinner('AI is tuning hyperparameters (20 trials)...'):
                    results[name] = run_backtest(model_inst, X, y)
            else:
                results[name] = run_backtest(model_inst, X, y)
        except Exception as exc:
            st.warning(f"⚠️ {name} failed: {exc}")
        finally:
            gc.collect()

    if not results:
        st.error("❌ All models in the zoo failed. See logs for details.")
        st.stop()
    status_ph.empty()

    winner_name = min(results, key=lambda k: results[k]["rmse"])
    winner_res  = results[winner_name]
    model       = winner_res.get("model", zoo[winner_name])

    # 4. Future Forecast (Using fully fitted final model)
    status_msg = f"⏳  Running {n_future_days}-day ensemble forecast…"
         
    with st.spinner(status_msg):
        try:
            future_df = generate_future_forecast(
                model=model, model_name=winner_name,
                X_train=X_train, y_train=y_train, X_inference=X_inference,
                feature_cols=feature_cols, target_col=TARGET_COL, n_days=n_future_days,
            )
            forecast_ok = True
        except Exception as exc:
            st.error(f"❌ Future forecast failed: {exc}")
            forecast_ok = False

    # 5. AI Council Initialization & Meta-Ensemble Match
    with st.spinner("🤖 Consulting AI Council (Ensemble Generation)..."):
        council_preds = {}
        
        # 1. Tree (XGBoost)
        try:
            tree_model = TreeForecaster()
            tree_model.train(X, y)
            # Seed X_inference with indicator columns the model expects
            tree_inf = X_inference.copy()
            for col in feature_cols:
                if col not in tree_inf.columns:
                    tree_inf[col] = float(df[col].iloc[-1]) if col in df.columns else 0.0
            tree_inf = tree_inf[feature_cols]  # exact column order
            council_preds['xgboost'] = float(tree_model.predict(tree_inf)[0])
        except Exception as e:
            st.warning(f"TreeExpert Failed: {e}")
            council_preds['xgboost'] = None
        finally:
            gc.collect()
            
        # 2. Time (Prophet)
        try:
            time_model = TimeExpert()
            prophet_preds = time_model.forecast(X_train, horizon=1)
            council_preds['prophet'] = float(prophet_preds[-1])
        except Exception as e:
            st.warning(f"TimeExpert Failed: {e}")
            council_preds['prophet'] = None
        finally:
            gc.collect()
            
        # Meta-Ensemble Execution
        try:
            meta = MetaForecaster()
            ensemble_price = meta.ensemble(df, council_preds)
            weights_used = meta._get_volatility_weights(df)
            is_high_vol = (weights_used == meta.weights_high_vol)
            market_regime = "High Volatility (Stress)" if is_high_vol else "Low Volatility (Calm)"
        except Exception as e:
            st.error(f"❌ MetaForecaster Failed: {e}")
            ensemble_price = None
            market_regime = "Unknown"
            weights_used = {}
            meta = MetaForecaster()

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
    st.session_state["current_sentiment"] = current_sentiment
    st.session_state["n_future_days"]    = n_future_days
    
    # Store Council Details
    st.session_state["council_preds"]    = council_preds
    st.session_state["ensemble_price"]   = ensemble_price
    st.session_state["market_regime"]    = market_regime
    st.session_state["weights_used"]     = weights_used
    st.session_state["meta"]             = meta

    st.toast(f"Pipeline complete! {winner_name} won the race.", icon="🚀")
    gc.collect()

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
current_sentiment = st.session_state.get("current_sentiment", 0.0)

council_preds    = st.session_state.get("council_preds", {})
ensemble_price   = st.session_state.get("ensemble_price", None)
market_regime    = st.session_state.get("market_regime", "Unknown")
weights_used     = st.session_state.get("weights_used", {})
meta_model       = st.session_state.get("meta", None)

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
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker",       ticker_ran)
m2.metric("Best Engine",  winner_name.split(" (")[0])
m3.metric("Winner RMSE",  f"{winner_res['rmse']:.4f}")
m4.metric("Winner MAPE",  f"{winner_res['mape']:.4f} %")

if ensemble_price is not None and not np.isnan(ensemble_price):
    pred_diff = ensemble_price - df["Close"].iloc[-1]
    pred_pct = (pred_diff / df["Close"].iloc[-1]) * 100
    m5.metric("Ensemble AI Target", f"{curr_sym}{ensemble_price:.2f}", f"{pred_diff:+.2f} ({pred_pct:+.2f}%)")
else:
    m5.metric("Ensemble AI Target", "N/A")

# Sentiment
if current_sentiment > 0.05:
    st.info(f"📰 **Current News Sentiment:** Bullish 🟢 (Score: {current_sentiment:.2f}) | **Market Regime:** {market_regime}")
elif current_sentiment < -0.05:
    st.warning(f"📰 **Current News Sentiment:** Bearish 🔴 (Score: {current_sentiment:.2f}) | **Market Regime:** {market_regime}")
else:
    st.markdown(f"📰 **Current News Sentiment:** Neutral ⚪ (Score: {current_sentiment:.2f}) | **Market Regime:** {market_regime}")

if isinstance(model, TreeForecaster) and hasattr(model, "model") and hasattr(model.model, "get_params"):
    with st.expander("Model Hyperparameters"):
        params = model.model.get_params()
        tuned = {k: params[k] for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"] if k in params}
        st.json(tuned if tuned else params)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "🏎️ Leaderboard", "📊 Raw Data"])

# ── TAB 1 — Backtest Chart + Future Forecast ──────────────────────────────────
with tab1:
    st.markdown(f"#### 📉 30-Day Backtest · {winner_name}")

    col_chart, _ = st.columns([1, 2])
    with col_chart:
        chart_type = st.selectbox(
            "Chart Style",
            ["Candlestick", "Line"],
            index=0,
        )

    traj      = winner_res["trajectory"]
    bt_dates  = traj.index
    actuals   = traj["Actual"].values
    preds     = traj["Predicted"].values
    upper = preds * 1.05
    lower = preds * 0.95

    ohlc_bt = df[["Open", "High", "Low", "Close"]].loc[df.index.isin(bt_dates)]

    # ── SMA & RSI series aligned to backtest window ───────────────────────────
    sma50_bt  = df["SMA_50"].loc[df.index.isin(bt_dates)]
    sma200_bt = df["SMA_200"].loc[df.index.isin(bt_dates)]
    rsi_bt    = df["RSI_14"].loc[df.index.isin(bt_dates)]

    # ── Double-Decker Subplot Layout ──────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker_ran} Price", "RSI (14)"),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — Price Chart + SMA Overlays + CI Band + AI Prediction
    # ══════════════════════════════════════════════════════════════════════════
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=ohlc_bt.index, open=ohlc_bt["Open"], high=ohlc_bt["High"],
            low=ohlc_bt["Low"], close=ohlc_bt["Close"],
            name="OHLC (Actual)",
            increasing=dict(line=dict(color="#00e6b4"), fillcolor="rgba(0,230,180,0.5)"),
            decreasing=dict(line=dict(color="#ff4c6a"), fillcolor="rgba(255,76,106,0.5)"),
            hoverinfo="x+y",
        ), row=1, col=1)
    else:  # Line
        fig.add_trace(go.Scatter(
            x=bt_dates, y=actuals, mode="lines",
            name="Historical Close",
            line=dict(color="#ffffff", width=2),
            hovertemplate=f"<b>Close</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
        ), row=1, col=1)

    # SMA 50 (Gold)
    fig.add_trace(go.Scatter(
        x=sma50_bt.index, y=sma50_bt.values,
        mode="lines", name="SMA 50",
        line=dict(color="gold", width=1.5),
        hovertemplate=f"<b>SMA 50</b>: {curr_sym}%{{y:.2f}}<extra></extra>",
    ), row=1, col=1)

    # SMA 200 (Royal Blue)
    fig.add_trace(go.Scatter(
        x=sma200_bt.index, y=sma200_bt.values,
        mode="lines", name="SMA 200",
        line=dict(color="royalblue", width=1.5),
        hovertemplate=f"<b>SMA 200</b>: {curr_sym}%{{y:.2f}}<extra></extra>",
    ), row=1, col=1)

    # 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=np.concatenate([bt_dates, bt_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(0,153,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt_dates, y=upper, mode="lines",
        line=dict(color="rgba(0,153,255,0.40)", width=1, dash="dot"),
        name="Upper CI", hovertemplate=f"Upper CI: {curr_sym}%{{y:.2f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt_dates, y=lower, mode="lines",
        line=dict(color="rgba(0,153,255,0.40)", width=1, dash="dot"),
        name="Lower CI", hovertemplate=f"Lower CI: {curr_sym}%{{y:.2f}}<extra></extra>",
    ), row=1, col=1)

    # AI Prediction
    fig.add_trace(go.Scatter(
        x=bt_dates, y=preds, mode="lines+markers", name="AI Prediction",
        line=dict(color="#00e6b4", width=2.5, dash="dash"),
        marker=dict(size=5, color="#00e6b4", symbol="diamond"),
        hovertemplate=f"<b>Predicted</b>: {curr_sym}%{{y:.2f}}<br>%{{x|%b %d %Y}}<extra></extra>",
    ), row=1, col=1)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — RSI Panel
    # ══════════════════════════════════════════════════════════════════════════
    fig.add_trace(go.Scatter(
        x=rsi_bt.index, y=rsi_bt.values,
        mode="lines", name="RSI 14",
        line=dict(color="#9b59b6", width=2),
        hovertemplate="<b>RSI</b>: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    # Overbought / Oversold threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Overbought (70)",
                  annotation_position="top right",
                  annotation_font_color="gray",
                  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Oversold (30)",
                  annotation_position="bottom right",
                  annotation_font_color="gray",
                  row=2, col=1)

    # ══════════════════════════════════════════════════════════════════════════
    # Global Layout
    # ══════════════════════════════════════════════════════════════════════════
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=60, b=0),
        height=680,
    )

    # Row 1 axes
    fig.update_yaxes(
        title_text="Price ($)",
        gridcolor="rgba(255,255,255,0.06)",
        showline=True, linecolor="rgba(255,255,255,0.15)",
        tickprefix=curr_sym,
        row=1, col=1,
    )

    # Row 2 axes — RSI locked 0-100
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        gridcolor="rgba(255,255,255,0.06)",
        showline=True, linecolor="rgba(255,255,255,0.15)",
        row=2, col=1,
    )

    # Shared x-axis label only on the bottom row; remove rangeslider
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        showline=True, linecolor="rgba(255,255,255,0.15)",
        row=1, col=1,
        rangeslider_visible=False,
    )
    fig.update_xaxes(
        title_text="Date",
        gridcolor="rgba(255,255,255,0.06)",
        showline=True, linecolor="rgba(255,255,255,0.15)",
        row=2, col=1,
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
            
            dynamic_height = max(360, len(feature_cols) * 28)
            fi_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(13,17,23,0.8)",
                                 showlegend=False, coloraxis_showscale=False,
                                 height=dynamic_height, margin=dict(l=0, r=0, t=40, b=0))
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
        fut_upper  = future_df["Predicted_Close"].values * 1.05
        fut_lower  = future_df["Predicted_Close"].values * 0.95

        fut_fig = go.Figure()

        # History context
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
    st.markdown("#### 🧠 AI Council Ensemble (1-Day Target)")
    if council_preds and meta_model:
        council_data = []
        engine_details = {
            'xgboost': "TreeExpert (XGBoost)",
            'prophet': "TimeExpert (Prophet)",
        }
        
        final_weights = meta_model._redistribute_weights(weights_used, [k for k, v in council_preds.items() if v is not None])
        
        for engine_key, engine_name in engine_details.items():
            base_w = weights_used.get(engine_key, 0)
            final_w = final_weights.get(engine_key, 0)
            pred_val = council_preds.get(engine_key)
            
            council_data.append({
                "Model": engine_name,
                "Target Price": f"{curr_sym}{pred_val:.2f}" if pred_val is not None else "⚠️ FAILED",
                "Volatility Engine Weight": f"{final_w * 100:.1f}%",
            })
            
        st.table(pd.DataFrame(council_data))
        st.caption(f"Currently acting under **{market_regime}** parameters. Weights dynamically adjusted.")
    else:
        st.info("AI Council Ensemble calculations not available.")

    st.markdown("---")
    st.markdown("#### 📰 NLP News Sentiment")
    # st.info(f"**Current Sentiment:** {s_label} (Score: {latest_sentiment:.2f})")
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
st.caption("OmniQuant v0.8.1 · Ensembles + AutoML")
