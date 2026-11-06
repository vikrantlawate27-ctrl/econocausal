"""
EconoCausal Double Machine Learning (DML) Engine
Implements Robinson's transformation with Neyman Orthogonal Scores and K-Fold Cross-Fitting
for unbiased estimation of Price Elasticity and Conditional Average Treatment Effects (CATE).

Theoretical Foundation:
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats


class DoubleMachineLearningPricing:
    """
    Double/Debiased Machine Learning (DML) estimator for dynamic pricing elasticity.
    
    Model formulation:
        Y = theta(X) * T + g(W, X) + U,   E[U | T, W, X] = 0
        T = m(W, X) + V,                 E[V | W, X] = 0
        
    Where:
        Y: Observed Demand (Outcome)
        T: Price (Treatment)
        W: Confounders (competitor price, seasonality, marketing, inventory, etc.)
        X: Heterogeneity modifiers (customer tier, urgency score, device type)
        theta(X): Heterogeneous Causal Price Elasticity (CATE)
    """

    def __init__(
        self,
        n_splits: int = 5,
        outcome_model=None,
        treatment_model=None,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state

        # First stage models: Non-parametric ML regressors
        self.outcome_model = outcome_model or HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.08, random_state=random_state
        )
        self.treatment_model = treatment_model or HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, learning_rate=0.08, random_state=random_state
        )

        # Second stage model: Orthogonalized CATE estimator
        self.second_stage = Ridge(alpha=1e-3, fit_intercept=False)

        # Storage for results and diagnostics
        self.confounders: List[str] = []
        self.moderators: List[str] = []
        self.ate_: Optional[float] = None
        self.ate_se_: Optional[float] = None
        self.ate_ci_: Optional[Tuple[float, float]] = None
        self.coef_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[np.ndarray] = None
        self.feature_names_second_stage_: List[str] = []
        self.first_stage_diagnostics_: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "price",
        outcome_col: str = "demand",
        confounder_cols: Optional[List[str]] = None,
        moderator_cols: Optional[List[str]] = None,
    ) -> "DoubleMachineLearningPricing":
        """
        Fits the DML model via K-Fold Cross-Fitting.
        """
        self.confounders = confounder_cols or [
            "seasonality", "competitor_price", "inventory_level",
            "marketing_spend", "day_of_week", "is_weekend", "store_rating"
        ]
        self.moderators = moderator_cols or ["customer_tier", "urgency_score", "device_type"]

        # Confounders + moderators form the full background conditioning set Z = [W, X]
        z_cols = list(dict.fromkeys(self.confounders + self.moderators))

        Z = df[z_cols].values
        T = df[treatment_col].values
        Y = df[outcome_col].values
        X = df[self.moderators].values if self.moderators else None

        n = len(df)
        t_res = np.zeros(n)
        y_res = np.zeros(n)
        t_pred = np.zeros(n)
        y_pred = np.zeros(n)

        # K-Fold Cross-Fitting to eliminate regularization and overfitting bias
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(Z):
            Z_train, Z_val = Z[train_idx], Z[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Clone and fit Stage 1 Outcome Model: E[Y | Z]
            m_y = HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=31, learning_rate=0.08, random_state=self.random_state
            )
            m_y.fit(Z_train, Y_train)
            y_hat_val = m_y.predict(Z_val)
            y_pred[val_idx] = y_hat_val
            y_res[val_idx] = Y_val - y_hat_val

            # Clone and fit Stage 1 Treatment Model: E[T | Z]
            m_t = HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=31, learning_rate=0.08, random_state=self.random_state
            )
            m_t.fit(Z_train, T_train)
            t_hat_val = m_t.predict(Z_val)
            t_pred[val_idx] = t_hat_val
            t_res[val_idx] = T_val - t_hat_val

        # First stage diagnostics
        self.first_stage_diagnostics_ = {
            "y_rmse": float(np.sqrt(mean_squared_error(Y, y_pred))),
            "y_r2": float(r2_score(Y, y_pred)),
            "t_rmse": float(np.sqrt(mean_squared_error(T, t_pred))),
            "t_r2": float(r2_score(T, t_pred)),
        }

        # ------------------------------------------------------------------
        # Second Stage: Robinson Residual-on-Residual Regression with CATE
        # ------------------------------------------------------------------
        # Design matrix for second stage: [t_res, t_res * X_1, ..., t_res * X_p]
        features = [t_res.reshape(-1, 1)]
        feat_names = ["treatment_base"]

        if X is not None and len(self.moderators) > 0:
            for i, col in enumerate(self.moderators):
                inter = (t_res * X[:, i]).reshape(-1, 1)
                features.append(inter)
                feat_names.append(f"treatment_x_{col}")

        M_res = np.hstack(features)
        self.feature_names_second_stage_ = feat_names

        # Fit second stage
        self.second_stage.fit(M_res, y_res)
        self.coef_ = self.second_stage.coef_

        # Robust HC1 Covariance Matrix estimation: (M^T M)^(-1) (M^T diag(e^2) M) (M^T M)^(-1)
        residuals = y_res - self.second_stage.predict(M_res)
        deg_freedom = max(1, n - M_res.shape[1])
        hc1_scale = n / deg_freedom

        inv_mtm = np.linalg.pinv(M_res.T @ M_res)
        meat = (M_res.T * (residuals ** 2)) @ M_res
        self.cov_matrix_ = hc1_scale * (inv_mtm @ meat @ inv_mtm)
        self.is_fitted = True

        # Baseline ATE (Average Treatment Effect)
        # Point estimate of ATE = mean of theta(X_i) across sample
        cate_all = self.predict_cate(df[self.moderators])
        self.ate_ = float(np.mean(cate_all))
        
        # Standard error of ATE
        base_se = np.sqrt(np.diag(self.cov_matrix_)[0])
        self.ate_se_ = float(base_se)
        self.ate_ci_ = (
            float(self.ate_ - 1.96 * base_se),
            float(self.ate_ + 1.96 * base_se),
        )

        return self

    def predict_cate(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts conditional average treatment effect (elasticity theta(X)) for given customer features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        n_samples = len(X_df)
        theta = np.full(n_samples, self.coef_[0])

        if len(self.moderators) > 0:
            X_vals = X_df[self.moderators].values
            for i in range(len(self.moderators)):
                theta += self.coef_[i + 1] * X_vals[:, i]

        return theta

    def predict_cate_interval(self, X_df: pd.DataFrame, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates pointwise point estimates and (1 - alpha)% confidence intervals for CATE.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        z_crit = stats.norm.ppf(1 - alpha / 2)
        n = len(X_df)

        # Build feature multiplier vector: [1, x_1, x_2, ..., x_p]
        design = [np.ones((n, 1))]
        if len(self.moderators) > 0:
            design.append(X_df[self.moderators].values)
        V = np.hstack(design)

        # Point estimate: theta = V @ beta
        theta = V @ self.coef_

        # Pointwise standard error: sqrt(diag(V @ Cov @ V^T))
        var_theta = np.sum((V @ self.cov_matrix_) * V, axis=1)
        se_theta = np.sqrt(np.maximum(1e-12, var_theta))

        lower = theta - z_crit * se_theta
        upper = theta + z_crit * se_theta

        return theta, lower, upper

    def summary(self) -> Dict[str, Any]:
        """Returns model coefficients, standard errors, p-values, and confidence intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        std_errs = np.sqrt(np.maximum(1e-12, np.diag(self.cov_matrix_)))
        t_stats = self.coef_ / std_errs
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

        coef_summary = []
        for name, coef, se, t_val, p_val in zip(
            self.feature_names_second_stage_, self.coef_, std_errs, t_stats, p_values
        ):
            coef_summary.append({
                "parameter": name,
                "estimate": float(coef),
                "std_error": float(se),
                "t_stat": float(t_val),
                "p_value": float(p_val),
                "ci_lower": float(coef - 1.96 * se),
                "ci_upper": float(coef + 1.96 * se),
            })

        return {
            "ate": self.ate_,
            "ate_se": self.ate_se_,
            "ate_ci": self.ate_ci_,
            "coefficients": coef_summary,
            "first_stage_diagnostics": self.first_stage_diagnostics_,
        }
