"""
OmniQuant — Layer 3: The Validation Engine
===========================================
Implements strict Walk-Forward Validation over a fixed 30-day hold-out
window.  The backtest function is model-agnostic — any class that conforms
to the BaseForecaster interface (train / predict) can be evaluated.

Returns
-------
dict with keys:
    rmse        : float  — Root Mean Squared Error
    mape        : float  — Mean Absolute Percentage Error (%)
    trajectory  : pd.DataFrame — columns ['Actual', 'Predicted'] for the
                  30-day test window (used by the Layer 4 dashboard)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# Number of days in the hold-out test window
HOLDOUT_DAYS = 30


def run_backtest(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Walk-Forward Validation over the last HOLDOUT_DAYS rows.

    Parameters
    ----------
    model : BaseForecaster
        Any forecaster from model_zoo.py that implements train() and predict().
    X : pd.DataFrame
        Full feature matrix (chronologically ordered, NaNs already dropped).
    y : pd.Series
        Full target series aligned with X.

    Returns
    -------
    dict
        {
            "rmse"      : float,
            "mape"      : float,   # expressed as a percentage, e.g. 2.34
            "trajectory": pd.DataFrame with columns ["Actual", "Predicted"]
        }
    """
    if len(X) <= HOLDOUT_DAYS:
        raise ValueError(
            f"Dataset has only {len(X)} rows — need more than {HOLDOUT_DAYS} "
            "rows for a 30-day hold-out backtest."
        )

    # ── 1. Split ─────────────────────────────────────────────────────────────
    X_train = X.iloc[:-HOLDOUT_DAYS]
    y_train = y.iloc[:-HOLDOUT_DAYS]

    X_test  = X.iloc[-HOLDOUT_DAYS:]
    y_test  = y.iloc[-HOLDOUT_DAYS:]

    print(f"\n[Evaluator] Walk-Forward Validation")
    print(f"  Training window : {X_train.index[0].date()} → {X_train.index[-1].date()}  ({len(X_train)} rows)")
    print(f"  Test window     : {X_test.index[0].date()}  → {X_test.index[-1].date()}  ({len(X_test)} rows)")

    # ── 2. Train ─────────────────────────────────────────────────────────────
    model.train(X_train, y_train)

    # ── 3. Predict ───────────────────────────────────────────────────────────
    preds = model.predict(X_test)

    # ── 4. Score ─────────────────────────────────────────────────────────────
    actuals = y_test.values

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mape = float(mean_absolute_percentage_error(actuals, preds)) * 100  # → %

    # ── 5. Build trajectory DataFrame ────────────────────────────────────────
    trajectory = pd.DataFrame(
        {
            "Actual"   : actuals,
            "Predicted": np.round(preds, 4),
            "Error"    : np.round(preds - actuals, 4),
        },
        index=X_test.index,
    )
    trajectory.index.name = "Date"

    return {
        "rmse"      : rmse,
        "mape"      : mape,
        "trajectory": trajectory,
    }
