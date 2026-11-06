"""
EconoCausal Baseline Demand Model
Implements a supervised Machine Learning demand prediction model using RandomForestRegressor.
Evaluates accuracy metrics (MAE, RMSE, R2) and serves point predictions.
"""

from typing import Dict, Any, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from models.preprocessing import DataPreprocessor

MODEL_PATH = str(Path(__file__).resolve().parent / "demand_rf.joblib")


class BaselineDemandModel:
    """
    Supervised predictive model for sales demand using RandomForestRegressor.
    Used as an operational predictive baseline.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.preprocessor = DataPreprocessor(target_col="demand", treatment_col="price")
        self.metrics: Dict[str, float] = {}
        self.is_trained: bool = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Trains the Random Forest model and computes out-of-sample evaluation metrics.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        train_df, test_df = self.preprocessor.split_data(df, test_size=0.2, random_state=self.random_state)

        # Preprocess features (including price as a predictor feature)
        X_train_cov, T_train, Y_train, feat_names = self.preprocessor.fit_transform(train_df)
        X_test_cov, T_test, Y_test = self.preprocessor.transform(test_df)

        # Include Price in feature matrix for supervised demand prediction
        X_train = np.hstack([T_train.reshape(-1, 1), X_train_cov])
        X_test = np.hstack([T_test.reshape(-1, 1), X_test_cov])

        with joblib.parallel_backend('threading'):
            self.model.fit(X_train, Y_train)
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = float(mean_absolute_error(Y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(Y_test, y_pred)))
        r2 = float(r2_score(Y_test, y_pred))

        self.metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        }

        # Save model and preprocessor artifact
        self.save()
        print(f"Baseline Demand Model Trained -> MAE: {self.metrics['MAE']}, RMSE: {self.metrics['RMSE']}, R2: {self.metrics['R2']}")
        return self.metrics

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts demand given a DataFrame of transaction features.
        """
        if not self.is_trained:
            self.load()

        X_cov, T, _ = self.preprocessor.transform(feature_df)
        if T is None:
            raise ValueError("Price column must be provided in feature_df for demand prediction.")
        
        X_full = np.hstack([T.reshape(-1, 1), X_cov])
        preds = self.model.predict(X_full)
        return np.maximum(0.0, np.round(preds, 1))

    def save(self, path: str = MODEL_PATH):
        """Saves model and preprocessor to disk."""
        artifact = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(artifact, path)

    def load(self, path: str = MODEL_PATH):
        """Loads model and preprocessor from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found. Please train model first.")
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.metrics = artifact["metrics"]
        self.is_trained = artifact["is_trained"]
        return self


if __name__ == "__main__":
    sales_csv = "data/sales_data.csv"
    if os.path.exists(sales_csv):
        df = pd.read_csv(sales_csv)
        dm = BaselineDemandModel()
        dm.train(df)
