import urllib.request
urllib.request.getproxies = lambda: {}

from datetime import datetime, timedelta
import pandas as pd

from data_orchestrator import DataOrchestrator
from model_zoo import TreeForecaster
from evaluator import run_backtest

DIVIDER = "=" * 57

# ── Layer 1: Data Orchestrator ────────────────────────────────────────────────
end_date   = datetime.now()
start_date = end_date - timedelta(days=730)

end_date_str   = end_date.strftime("%Y-%m-%d")
start_date_str = start_date.strftime("%Y-%m-%d")

ticker = "AAPL"
print(DIVIDER)
print(" OmniQuant — Layer 1 + 2 + 3 Verification")
print(DIVIDER)
print(f"Ticker     : {ticker}")
print(f"Date Range : {start_date_str}  →  {end_date_str}\n")

orchestrator = DataOrchestrator()
df = orchestrator.process(ticker, start_date_str, end_date_str)
print(f"\nLayer 1 OK — DataFrame shape: {df.shape}")

# ── Feature / Target split ────────────────────────────────────────────────────
TARGET_COL   = "Close"

# Dynamically include pandas_ta columns (matching specific prefixes) and exclude the target variable
ta_patterns = ("RSI_", "BBL_", "BBM_", "BBU_", "BBB_", "BBP_", "MACD", "ATR", "OBV", "SMA_")
feature_cols = [
    c for c in df.columns 
    if c != TARGET_COL and (c.startswith(ta_patterns) or c in ["Log_Returns", "Sentiment", "Volume"])
]

X = df[feature_cols]
y = df[TARGET_COL]

# ── Layer 3: Validation Engine ────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print(" Layer 3 — Walk-Forward Backtest (30-day hold-out)")
print(DIVIDER)

tree = TreeForecaster(n_estimators=300, learning_rate=0.05, max_depth=4)
results = run_backtest(tree, X, y)

# ── Print Scores ──────────────────────────────────────────────────────────────
print(f"\n{'─' * 40}")
print(f"  RMSE : {results['rmse']:.4f}")
print(f"  MAPE : {results['mape']:.4f} %")
print(f"{'─' * 40}")

# ── Print 30-day Trajectory ───────────────────────────────────────────────────
print("\n--- 30-Day Prediction Trajectory ---")
print(results["trajectory"].to_string())
print(f"\nLayer 3 OK — Validation Engine verified successfully!")
