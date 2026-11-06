"""
EconoCausal End-to-End Pipeline
Executes data generation, EDA, model training (Naive vs Double ML),
counterfactual pricing policy evaluation, and artifacts serialization.
"""

import os
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src is in sys.path
SRC_DIR = str(Path(__file__).resolve().parent.parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from econocausal.data_generator import DynamicPricingDataGenerator, SimulationConfig
from econocausal.eda import PricingEDA
from econocausal.double_ml import DoubleMachineLearningPricing
from econocausal.naive_models import NaiveOLSPricing, NaiveSupervisedMLPricing
from econocausal.pricing_optimizer import PricingOptimizer
from econocausal.counterfactual_eval import CounterfactualEvaluator


def run_pipeline():
    print("=" * 70)
    print("ECONOCAUSAL: DYNAMIC PRICING VIA DOUBLE MACHINE LEARNING")
    print("Infotact Solutions - Data Science & ML Internship Project")
    print("Intern: Vikrant Bharat Lawate | Ref. No.: ITS/DSML/1090")
    print("=" * 70)

    # 1. Output Directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # 2. Data Generation
    print("\n[Step 1/6] Generating synthetic dynamic pricing dataset with price endogeneity...")
    config = SimulationConfig(n_samples=30000, random_state=42)
    generator = DynamicPricingDataGenerator(config)
    df, meta = generator.generate()
    
    data_path = "data/dynamic_pricing_transactions.csv"
    df.to_csv(data_path, index=False)
    print(f"-> Generated {len(df):,} records saved to {data_path}")
    print(f"-> True Population Average Treatment Effect (ATE): {meta['true_ate']:.4f}")

    # 3. Exploratory Data Analysis & Endogeneity Demonstration
    print("\n[Step 2/6] Running Exploratory Data Analysis & Confounder Visualizations...")
    eda = PricingEDA(df, output_dir="outputs/figures")
    eda_summary = eda.run_summary()
    eda_charts = eda.generate_all_visualizations()
    print(f"-> Raw Price-Demand Correlation: {eda_summary['raw_price_demand_corr']:+.4f} (Endogeneity attenuated!)")
    print(f"-> Saved EDA figures to outputs/figures/")

    # 4. Train / Test Split
    print("\n[Step 3/6] Partitioning data into train (80%) and holdout test (20%)...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    print(f"-> Train set: {len(train_df):,} | Test set: {len(test_df):,}")

    # 5. Fit Benchmarks (Naive OLS and Naive Supervised ML)
    print("\n[Step 4/6] Fitting benchmark models (Naive OLS and Naive Supervised ML)...")
    naive_bivariate = NaiveOLSPricing(include_controls=False).fit(train_df)
    naive_multivariate = NaiveOLSPricing(include_controls=True).fit(train_df)
    naive_ml = NaiveSupervisedMLPricing().fit(train_df)

    print(f"-> Naive Bivariate OLS Elasticity: {naive_bivariate.price_coef_:.4f}")
    print(f"-> Naive Multivariate OLS Elasticity: {naive_multivariate.price_coef_:.4f}")

    # 6. Fit EconoCausal Double Machine Learning (DML)
    print("\n[Step 5/6] Fitting EconoCausal Double Machine Learning (Chernozhukov et al.)...")
    dml = DoubleMachineLearningPricing(n_splits=5, random_state=42)
    dml.fit(train_df)

    dml_summary = dml.summary()
    print(f"-> DML Estimated ATE: {dml_summary['ate']:.4f} (95% CI: [{dml_summary['ate_ci'][0]:.4f}, {dml_summary['ate_ci'][1]:.4f}])")
    print(f"-> Ground Truth ATE:   {meta['true_ate']:.4f}")
    
    # Save DML Model
    model_artifact = {
        "model": dml,
        "metadata": meta,
        "summary": dml_summary,
    }
    joblib.dump(model_artifact, "outputs/models/dml_pricing_model.joblib")
    print("-> Saved DML model artifact to outputs/models/dml_pricing_model.joblib")

    # 7. Counterfactual Pricing Optimization & Policy Evaluation
    print("\n[Step 6/6] Executing counterfactual policy optimization on test set...")
    optimizer = PricingOptimizer()

    # Predict test causal elasticity theta(X)
    dml_test_elasticities = dml.predict_cate(test_df)
    
    # DML baseline demand from first stage
    test_z_cols = list(dict.fromkeys(dml.confounders + dml.moderators))
    test_baseline_demand = test_df["demand"].values # base reference

    # Compute optimal prices under DML
    dml_opt_df = optimizer.batch_optimize(
        test_df,
        causal_elasticities=dml_test_elasticities,
        baseline_demands=test_baseline_demand,
        objective="profit",
    )

    # Compute naive optimal prices using naive multivariate OLS elasticity
    naive_opt_df = optimizer.batch_optimize(
        test_df,
        causal_elasticities=np.full(len(test_df), naive_multivariate.price_coef_),
        baseline_demands=test_baseline_demand,
        objective="profit",
    )

    # Evaluate counterfactual outcomes
    evaluator = CounterfactualEvaluator(test_df, output_dir="outputs/figures")
    comparison_df, sim_df = evaluator.evaluate_policies(
        dml_optimal_prices=dml_opt_df["opt_price"].values,
        naive_optimal_prices=naive_opt_df["opt_price"].values,
        fixed_price=75.0,
    )
    chart_path = evaluator.plot_counterfactual_comparison(comparison_df)

    # Export metrics
    results_summary = {
        "internship_details": {
            "company": "Infotact Solutions",
            "intern_name": "Vikrant Bharat Lawate",
            "role": "Data Associate L1. Intern",
            "ref_no": "ITS/DSML/1090",
            "project_name": "EconoCausal: Dynamic Pricing via Double Machine Learning",
        },
        "ground_truth_ate": meta["true_ate"],
        "naive_bivariate_elasticity": naive_bivariate.price_coef_,
        "naive_multivariate_elasticity": naive_multivariate.price_coef_,
        "dml_estimated_ate": dml_summary["ate"],
        "dml_confidence_interval": dml_summary["ate_ci"],
        "policy_comparison": comparison_df.to_dict(orient="index"),
    }

    with open("outputs/reports/evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    comparison_df.to_csv("outputs/reports/policy_comparison_metrics.csv")
    print("\n" + "=" * 70)
    print("POLICY EVALUATION PERFORMANCE SUMMARY:")
    print("=" * 70)
    print(comparison_df[["avg_price", "total_revenue", "total_profit", "profit_margin_pct", "profit_uplift_pct"]])
    print("=" * 70)
    print(f"\nAll outputs successfully saved to /outputs/")
    return results_summary


if __name__ == "__main__":
    run_pipeline()
