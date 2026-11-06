"""
EconoCausal Data Preprocessing Module
Handles missing values, duplicate removal, date parsing,
categorical encoding, numerical normalization, and train/test splitting.
Designed to be clean, modular, and reusable for ML and DML pipelines.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """
    Standardizes and prepares sales and pricing data for demand modeling and causal estimation.
    """

    def __init__(self, target_col: str = "demand", treatment_col: str = "price"):
        self.target_col = target_col
        self.treatment_col = treatment_col

        # Feature sets
        self.numeric_features = [
            "competitor_price",
            "promotion",
            "holiday",
            "inventory",
            "temperature",
            "month",
        ]
        self.categorical_features = [
            "category",
            "day_of_week",
            "season",
        ]
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.encoded_feature_names: List[str] = []
        self.is_fitted = False

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates, handles missing values, and validates schema constraints.
        """
        df_clean = df.copy()

        # 1. Deduplication
        df_clean = df_clean.drop_duplicates()

        # 2. Date conversion (if month/day_of_week missing)
        if "month" not in df_clean.columns and "date" in df_clean.columns:
            d = pd.to_datetime(df_clean["date"])
            df_clean["month"] = d.dt.month
            df_clean["day_of_week"] = d.dt.day_name()

        # 3. Supply sensible defaults for missing numeric features
        defaults_numeric = {
            "competitor_price": df_clean.get("price", pd.Series([2500.0])).mean() if "price" in df_clean.columns else 2500.0,
            "promotion": 0,
            "holiday": 0,
            "inventory": 150,
            "temperature": 24.0,
            "month": 6,
        }
        for col, default_val in defaults_numeric.items():
            if col not in df_clean.columns:
                df_clean[col] = default_val

        # 4. Supply sensible defaults for missing categorical features
        defaults_cat = {
            "category": "Electronics",
            "day_of_week": "Wednesday",
            "season": "Summer",
        }
        for col, default_val in defaults_cat.items():
            if col not in df_clean.columns:
                df_clean[col] = default_val

        # 5. Missing value filling
        if df_clean.isnull().values.any():
            df_clean = df_clean.ffill().bfill()
            # If any NaNs remain (e.g. single-row input), fill with column defaults
            for col in self.numeric_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(defaults_numeric.get(col, 0.0))
            for col in self.categorical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(defaults_cat.get(col, "Unknown"))

        # 6. Range Validation & Integrity Checks
        if self.treatment_col in df_clean.columns:
            df_clean = df_clean[df_clean[self.treatment_col] > 0]
        if self.target_col in df_clean.columns:
            df_clean = df_clean[df_clean[self.target_col] >= 0]

        return df_clean

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Fits encoders on training data and returns:
        - X (Confounders / covariates matrix)
        - T (Treatment vector / price)
        - Y (Outcome vector / demand)
        - feature_names (List of processed column names)
        """
        df_clean = self.clean_raw_data(df)

        # Extract Treatment and Outcome
        T = df_clean[self.treatment_col].values
        Y = df_clean[self.target_col].values

        # Encode categorical features
        cat_data = df_clean[self.categorical_features]
        encoded_cats = self.encoder.fit_transform(cat_data)
        cat_feature_names = self.encoder.get_feature_names_out(self.categorical_features).tolist()

        # Numeric features
        num_data = df_clean[self.numeric_features].values

        # Combine covariates X
        X = np.hstack([num_data, encoded_cats])
        self.encoded_feature_names = self.numeric_features + cat_feature_names
        self.is_fitted = True

        return X, T, Y, self.encoded_feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transforms new input data using already fitted encoders.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted with fit_transform() first.")

        df_clean = self.clean_raw_data(df)

        T = df_clean[self.treatment_col].values if self.treatment_col in df_clean.columns else None
        Y = df_clean[self.target_col].values if self.target_col in df_clean.columns else None

        cat_data = df_clean[self.categorical_features]
        encoded_cats = self.encoder.transform(cat_data)
        num_data = df_clean[self.numeric_features].values

        X = np.hstack([num_data, encoded_cats])
        return X, T, Y

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into train and test sets.
        """
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
