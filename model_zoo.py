"""
OmniQuant — Layer 2: The Model Zoo
===================================
Provides a unified interface (BaseForecaster) for the primary tree-based model:
  - TreeForecaster : XGBoost gradient-boosted trees
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseForecaster(ABC):
    """Abstract base class that every forecaster in the zoo must implement."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TreeForecaster — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class TreeForecaster(BaseForecaster):
    """Gradient-boosted tree forecaster backed by XGBoost."""

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.01,
                 max_depth: int = 5, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42, **kwargs):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
            verbosity=0,
            **kwargs,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        print("[TreeForecaster] Training XGBRegressor...")
        self.model.fit(X, y)
        print("[TreeForecaster] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# ForestForecaster — Scikit-Learn RandomForest
# ─────────────────────────────────────────────────────────────────────────────

class ForestForecaster(BaseForecaster):
    """Random Forest regressor from Scikit-Learn."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 10,
                 random_state: int = 42, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        print("[ForestForecaster] Training RandomForest...")
        self.model.fit(X, y)
        print("[ForestForecaster] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# StatisticalForecaster — ARIMA
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalForecaster(BaseForecaster):
    """Classical statistical model using ARIMA (p,d,q)."""

    def __init__(self, order: tuple = (5, 1, 0), **kwargs):
        self.order = order
        self.model_fit = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA
        print(f"[StatisticalForecaster] Fitting ARIMA{self.order}...")
        # ARIMA in statsmodels uses the endogenous series 'y'
        # We ignore 'X' as it's a univariate statistical model
        model = ARIMA(y, order=self.order)
        self.model_fit = model.fit()
        print("[StatisticalForecaster] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_fit is None:
            raise ValueError("Model must be trained before predicting.")
        
        # We predict 'n' steps ahead based on the length of X
        # For walk-forward validation, it might be slightly different
        # if the evaluator passes multiple rows at once.
        # But here run_backtest passes X_test (30 days).
        n_steps = len(X)
        forecast = self.model_fit.forecast(steps=n_steps)
        return forecast.values

