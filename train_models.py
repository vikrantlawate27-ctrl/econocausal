"""
EconoCausal Model Training Script
Trains both the Baseline Random Forest Demand Model and the EconML Double Machine Learning Model.
Saves artifacts for the Flask application.
"""

import os
import sys
import time
import pandas as pd

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.demand_model import BaselineDemandModel
from models.dml_model import DoubleMLPricingModel
from database.database import init_db

def train_all():
    print("=" * 65)
    print("ECONOCAUSAL: TRAINING MACHINE LEARNING & CAUSAL DML PIPELINES")
    print("=" * 65)

    data_path = "data/sales_data.csv"
    if not os.path.exists(data_path):
        print("Data file not found. Generating dataset first...")
        from data.build_dataset import build_sales_data
        build_sales_data()

    # Ensure database is initialized
    print("\n[1/3] Checking SQLite database...")
    init_db(data_path)

    # Load data
    print(f"\n[2/3] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records across {df['product_id'].nunique()} products.")

    # Train Baseline Model
    print("\n[3/3] Training Baseline Random Forest Demand Model...")
    t0 = time.time()
    demand_model = BaselineDemandModel(n_estimators=80, max_depth=10, random_state=42)
    demand_metrics = demand_model.train(df)
    print(f"-> Completed in {time.time() - t0:.1f}s | Metrics: {demand_metrics}")

    # Train Double Machine Learning Model
    print("\n[4/4] Training EconML Double Machine Learning (LinearDML) Model...")
    t1 = time.time()
    dml_model = DoubleMLPricingModel(random_state=42)
    # We fit DML on a representative sample of 2500 records for fast, accurate convergence
    sample_df = df.sample(min(2500, len(df)), random_state=42)
    dml_results = dml_model.fit(sample_df)
    print(f"-> Completed in {time.time() - t1:.1f}s")
    print(f"-> ATE (Causal Price Effect): {dml_results['ATE']}")
    print(f"-> 95% Confidence Interval: [{dml_results['CI_Lower']}, {dml_results['CI_Upper']}]")

    print("\n" + "=" * 65)
    print("ALL MODELS SUCCESSFULLY TRAINED AND ARTIFACTS SERIALIZED")
    print("=" * 65)

if __name__ == "__main__":
    train_all()
