"""
OmniQuant — Layer 2: The Model Zoo
===================================
Provides a unified interface (BaseForecaster) for the primary tree-based model:
  - TreeForecaster : XGBoost gradient-boosted trees
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


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
# TreeForecaster — XGBoost + Bayesian Tuning Brain
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial, X_train_full, y_train_full):
    from xgboost import XGBRegressor
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    tscv = TimeSeriesSplit(n_splits=3)
    rmses = []
    
    for train_index, test_index in tscv.split(X_train_full):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        # Predict and score
        preds = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        from sklearn.metrics import mean_absolute_percentage_error
        mape = float(mean_absolute_percentage_error(y_test, preds)) * 100
        
        rmses.append(rmse)
        if 'mapes' not in locals(): mapes = []
        mapes.append(mape)
        
    trial.set_user_attr('mape', float(np.mean(mapes)))
    return float(np.mean(rmses))


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

    def train(self, X: pd.DataFrame, y: pd.Series):
        print("[TreeForecaster] Orchestrating Optuna study for hyperparameter tuning...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=20, show_progress_bar=True)
        
        best_params = study.best_params
        self.best_cv_rmse = study.best_value
        self.best_cv_mape = study.best_trial.user_attrs.get('mape', 0.0)
        print(f"[TreeForecaster] Optimized RMSE: {self.best_cv_rmse}")
        
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
            **best_params
        )
        
        print("[TreeForecaster] Fitting final model on full dataset with optimized parameters...")
        self.model.fit(X, y)
        print("[TreeForecaster] Training complete.")
        return self

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

# ─────────────────────────────────────────────────────────────────────────────
# TimeExpert — Prophet
# ─────────────────────────────────────────────────────────────────────────────

class TimeExpert:
    """Specialized forecaster wrapping Facebook/Prophet."""

    def __init__(self, **kwargs):
        from prophet import Prophet
        self.model = Prophet(**kwargs)

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the input DataFrame into the mandatory Prophet format.
        Renames the index/date column to 'ds' and the target price column to 'y'.
        """
        prophet_df = df.copy()
        
        if not isinstance(prophet_df.index, pd.RangeIndex):
            prophet_df = prophet_df.reset_index()
            
        cols = list(prophet_df.columns)
        rename_mapping = {}
        
        # Heuristically identify the date column for 'ds'
        date_col = cols[0]
        for col in cols:
            if 'date' in str(col).lower() or 'time' in str(col).lower() or col == 'index':
                date_col = col
                break
        rename_mapping[date_col] = 'ds'
        
        # Heuristically identify the price column for 'y'
        price_col = cols[-1]
        for col in reversed(cols):
            if str(col).lower() in ['close', 'price', 'y', 'target']:
                price_col = col
                break
                
        if price_col == date_col and len(cols) > 1:
            price_col = cols[1]
            
        rename_mapping[price_col] = 'y'
        
        prophet_df = prophet_df.rename(columns=rename_mapping)
        return prophet_df[['ds', 'y']]

    def forecast(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Fits the model on historical data and returns the yhat value for the requested future horizon.
        """
        prophet_df = self.preprocessing(df)
        self.model.fit(prophet_df)
        future = self.model.make_future_dataframe(periods=horizon)
        forecast_df = self.model.predict(future)
        return forecast_df['yhat'].tail(horizon).values



