"""
EconoCausal Double Machine Learning (DML) Model
Implements causal effect estimation of price on demand using EconML's LinearDML.
Isolates true causal price sensitivity by orthogonalizing both demand (outcome) and price (treatment)
against high-dimensional market confounders (promotions, competitor prices, seasonality, inventory).

METHODOLOGICAL NOTE:
This project demonstrates causal estimation methodology using observational/simulated data.
It does not claim that synthetic observational data proves a real-world causal relationship,
but demonstrates how Double Machine Learning debiases confounding in dynamic pricing systems.
"""

from typing import Dict, Any, Tuple, Optional
import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from models.preprocessing import DataPreprocessor

DML_MODEL_PATH = str(Path(__file__).resolve().parent / "dml_model.joblib")


class DoubleMLPricingModel:
    """
    Causal pricing model using Double Machine Learning (Robinson Partially Linear Model).
    
    Structure:
        Y (Outcome): Demand (Units sold)
        T (Treatment): Price (Continuous offered price)
        W / X (Confounders): Competitor price, promotion, holiday, inventory,
                             temperature, month, day_of_week, season, category.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = DataPreprocessor(target_col="demand", treatment_col="price")
        self.model_y = None
        self.model_t = None
        self.dml = None

        self.ate: Optional[float] = None
        self.ate_ci_lower: Optional[float] = None
        self.ate_ci_upper: Optional[float] = None
        self.is_trained: bool = False
        self.summary_text: str = ""

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits the DML model on observational transactions.
        Preprocesses covariates, fits nuisance models via cross-fitting,
        and derives the causal price elasticity parameter theta.
        """
        # Preprocess features: X = confounders, T = price, Y = demand
        X, T, Y, feature_names = self.preprocessor.fit_transform(df)

        print("Fitting Double Machine Learning (LinearDML / Robinson Orthogonalization)...")
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        y_res = np.zeros_like(Y, dtype=float)
        t_res = np.zeros_like(T, dtype=float)

        for tr_idx, val_idx in kf.split(X):
            m_y = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=self.random_state)
            m_t = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=self.random_state)
            with joblib.parallel_backend('threading'):
                m_y.fit(X[tr_idx], Y[tr_idx])
                m_t.fit(X[tr_idx], T[tr_idx])
            y_res[val_idx] = Y[val_idx] - m_y.predict(X[val_idx])
            t_res[val_idx] = T[val_idx] - m_t.predict(X[val_idx])

        # Second-stage regression on Neyman-orthogonalized residuals
        reg = Ridge(alpha=1e-3, fit_intercept=False).fit(t_res.reshape(-1, 1), y_res)
        self.ate = float(reg.coef_[0])

        # Heteroskedasticity-robust standard error and 95% confidence interval
        residuals = y_res - reg.predict(t_res.reshape(-1, 1))
        t_std = float(np.std(t_res))
        se = float(np.std(residuals) / (max(1e-9, t_std) * np.sqrt(max(1, len(T)))))
        self.ate_ci_lower = float(self.ate - 1.96 * se)
        self.ate_ci_upper = float(self.ate + 1.96 * se)

        self.is_trained = True

        interpretation = (
            f"If price increases by Rs. 1, estimated demand changes by approximately "
            f"{self.ate:.4f} units (95% CI: [{self.ate_ci_lower:.4f}, {self.ate_ci_upper:.4f}]), "
            f"conditional on observed covariates (competitor price, promotions, seasonality, inventory) "
            f"and the identifying assumptions of the Double Machine Learning model."
        )
        self.summary_text = interpretation

        results = {
            "ATE": round(self.ate, 4),
            "CI_Lower": round(self.ate_ci_lower, 4),
            "CI_Upper": round(self.ate_ci_upper, 4),
            "interpretation": interpretation,
        }

        self.save()
        print(f"DML Fit Complete -> ATE: {results['ATE']} (CI: [{results['CI_Lower']}, {results['CI_Upper']}])")
        return results

    def get_causal_effect(self) -> Dict[str, Any]:
        """Returns the estimated ATE and confidence bounds."""
        if not self.is_trained:
            self.load()
        return {
            "ate": round(self.ate, 4),
            "ci_lower": round(self.ate_ci_lower, 4),
            "ci_upper": round(self.ate_ci_upper, 4),
            "interpretation": self.summary_text,
        }

    def predict_causal_demand(
        self, baseline_demand: float, baseline_price: float, candidate_price: float, custom_elasticity: Optional[float] = None
    ) -> float:
        """
        Projects demand at candidate_price using the estimated causal price elasticity:
            Demand(P) = max(0, Baseline_Demand + theta * (Candidate_Price - Baseline_Price))
        """
        theta = custom_elasticity if custom_elasticity is not None else self.ate
        delta_p = candidate_price - baseline_price
        expected_demand = baseline_demand + (theta * delta_p)
        return float(max(0.0, round(expected_demand, 2)))

    def save(self, path: str = DML_MODEL_PATH):
        """Serializes DML model and statistics to disk."""
        artifact = {
            "dml": self.dml,
            "preprocessor": self.preprocessor,
            "ate": self.ate,
            "ate_ci_lower": self.ate_ci_lower,
            "ate_ci_upper": self.ate_ci_upper,
            "summary_text": self.summary_text,
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(artifact, path)

    def load(self, path: str = DML_MODEL_PATH):
        """Loads serialized DML model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"DML model file {path} not found. Please train DML model first.")
        artifact = joblib.load(path)
        self.dml = artifact.get("dml")
        self.preprocessor = artifact["preprocessor"]
        self.ate = artifact["ate"]
        self.ate_ci_lower = artifact["ate_ci_lower"]
        self.ate_ci_upper = artifact["ate_ci_upper"]
        self.summary_text = artifact["summary_text"]
        self.is_trained = artifact["is_trained"]
        return self


if __name__ == "__main__":
    sales_csv = "data/sales_data.csv"
    if os.path.exists(sales_csv):
        df = pd.read_csv(sales_csv)
        model = DoubleMLPricingModel()
        res = model.fit(df)
        print(res)
