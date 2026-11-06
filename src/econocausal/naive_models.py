"""
EconoCausal Naive Models Module
Implements standard non-causal approaches (Naive OLS, Standard Supervised ML)
to demonstrate failure modes and omitted-variable / simultaneity bias in dynamic pricing.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


class NaiveOLSPricing:
    """
    Standard Ordinary Least Squares (OLS) regression without orthogonalization.
    Demonstrates attenuation and positive endogeneity bias when prices are set
    in response to unmodeled or non-linearly interacting demand shocks.
    """

    def __init__(self, include_controls: bool = True):
        self.include_controls = include_controls
        self.model = LinearRegression()
        self.features: List[str] = []
        self.price_coef_: Optional[float] = None
        self.r2_: Optional[float] = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "price",
        outcome_col: str = "demand",
        control_cols: Optional[List[str]] = None,
    ) -> "NaiveOLSPricing":
        if self.include_controls:
            controls = control_cols or [
                "seasonality", "competitor_price", "inventory_level",
                "marketing_spend", "customer_tier", "urgency_score"
            ]
            self.features = [treatment_col] + controls
        else:
            self.features = [treatment_col]

        X = df[self.features].values
        y = df[outcome_col].values

        self.model.fit(X, y)
        self.price_coef_ = float(self.model.coef_[0])
        y_pred = self.model.predict(X)
        self.r2_ = float(r2_score(y, y_pred))
        return self

    def predict_demand(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].values)

    def summary(self) -> Dict[str, Any]:
        return {
            "model_type": f"Naive OLS ({'With Controls' if self.include_controls else 'Bivariate'})",
            "estimated_elasticity_slope": self.price_coef_,
            "r2": self.r2_,
        }


class NaiveSupervisedMLPricing:
    """
    Standard Supervised Machine Learning (Gradient Boosted Trees) trained directly
    on (Price + Confounders) to predict Demand.
    
    While achieving high predictive accuracy (low RMSE), standard ML trees confound
    the causal price effect with demand-boosting conditions (peak season, surge events),
    resulting in erratic and biased price sensitivity estimates via numerical differentiation.
    """

    def __init__(self, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, random_state=random_state
        )
        self.feature_cols: List[str] = []
        self.r2_: Optional[float] = None
        self.rmse_: Optional[float] = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "price",
        outcome_col: str = "demand",
        feature_cols: Optional[List[str]] = None,
    ) -> "NaiveSupervisedMLPricing":
        self.feature_cols = feature_cols or [
            "price", "seasonality", "competitor_price", "inventory_level",
            "marketing_spend", "day_of_week", "is_weekend", "store_rating",
            "customer_tier", "urgency_score", "device_type"
        ]
        X = df[self.feature_cols].values
        y = df[outcome_col].values

        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.r2_ = float(r2_score(y, y_pred))
        self.rmse_ = float(np.sqrt(mean_squared_error(y, y_pred)))
        return self

    def estimate_pseudo_elasticity(self, df: pd.DataFrame, delta: float = 1.0) -> np.ndarray:
        """
        Calculates numerical marginal change d(Demand)/d(Price) using finite differences.
        """
        df_base = df[self.feature_cols].copy()
        df_plus = df[self.feature_cols].copy()

        df_plus["price"] = df_plus["price"] + delta

        pred_base = self.model.predict(df_base.values)
        pred_plus = self.model.predict(df_plus.values)

        pseudo_elasticity = (pred_plus - pred_base) / delta
        return pseudo_elasticity
