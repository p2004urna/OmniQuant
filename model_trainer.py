"""
OmniQuant Meta-Logic Layer: Model Trainer & Ensembling
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import gc

class MetaForecaster:
    """
    Volatilty-Based Weighted Averaging to ensemble the outputs 
    of two independent models: XGBoost and Prophet.
    """

    def __init__(self):
        # Condition B (Low Volatility) -> Calm 
        self.weights_low_vol = {
            'prophet': 0.50,
            'xgboost': 0.50,
        }
        
        # Condition A (High Volatility) -> Stress
        self.weights_high_vol = {
            'xgboost': 0.75,
            'prophet': 0.25,
        }

    def _get_volatility_weights(self, df: pd.DataFrame) -> dict:
        """
        Calculates weights based on the current ATR relative to its 100-day distribution.
        """
        required_cols = ['High', 'Low', 'Close']
        has_required_cols = all(col in df.columns for col in required_cols)
        
        if has_required_cols:
            # Standard 14-period Average True Range
            atr = df.ta.atr(length=14)
        elif 'Close' in df.columns:
            # Fallback pseudo-ATR if only Close is available
            atr = df['Close'].diff().abs().rolling(window=14).mean()
        else:
            # Fallback to low volatility weights if we have no valid price data
            return self.weights_low_vol.copy()

        # Check if we have enough data (ATR calculation itself drops starting rows, 
        # so we need at least 100 valid ATR data points to gauge distribution)
        if atr is None or len(atr.dropna()) < 100:
            return self.weights_low_vol.copy()

        # Get the 100-day distribution
        atr_100_days = atr.dropna().tail(100)
        current_atr = atr_100_days.iloc[-1]
        
        # Calculate 75th percentile (top 25% boundary)
        percentile_75 = np.percentile(atr_100_days, 75)

        if current_atr >= percentile_75:
            # Condition A: Top 25% - High Stress
            return self.weights_high_vol.copy()
        else:
            # Condition B: Calm
            return self.weights_low_vol.copy()

    def _redistribute_weights(self, base_weights: dict, valid_models: list) -> dict:
        """
        Redistributes weights proportionally if a model fails.
        """
        active_weights = {k: v for k, v in base_weights.items() if k in valid_models}
        total_weight = sum(active_weights.values())
        
        if total_weight == 0:
            raise ValueError("All models failed. Cannot calculate an Ensemble_Price.")
            
        return {k: v / total_weight for k, v in active_weights.items()}

    def ensemble(self, df: pd.DataFrame, predictions: dict) -> float:
        """
        Takes independent predictions and returns a single Ensemble_Price.
        
        Parameters:
        - df: The historical DataFrame used to calculate Volatility/ATR.
        - predictions: A dictionary mapping model names to their predictions.
          e.g., {'xgboost': 150.5, 'lstm': None, 'prophet': 152.0}
          
        Returns:
        - Ensemble_Price as a weighted float.
        """
        base_weights = self._get_volatility_weights(df)
        
        valid_predictions = {}
        for model_name, pred in predictions.items():
            # Standardize model names against the weights dictionary
            key = str(model_name).lower()
            if key not in base_weights:
                continue
                
            try:
                if pred is not None:
                    # Extract scalar representation (fallback logic for sequences/iterables)
                    val = float(np.ravel(pred)[-1]) if isinstance(pred, (list, np.ndarray, pd.Series)) else float(pred)
                    if not np.isnan(val):
                        valid_predictions[key] = val
            except (ValueError, TypeError, IndexError):
                # If extraction fails, we treat it as a failed model and ignore it
                pass
                
        # Automatically redistributes weights to the successful models
        final_weights = self._redistribute_weights(base_weights, list(valid_predictions.keys()))
        
        # Execute: (P1*W1) + (P2*W2) + (P3*W3)
        ensemble_price = 0.0
        for model_key, pred_val in valid_predictions.items():
            ensemble_price += pred_val * final_weights[model_key]
            
        return ensemble_price
