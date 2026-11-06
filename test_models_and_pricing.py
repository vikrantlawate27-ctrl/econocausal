"""
Unit & Integration Tests for ML Models, Preprocessing, and Pricing Engine
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.preprocessing import DataPreprocessor
from models.demand_model import BaselineDemandModel
from models.dml_model import DoubleMLPricingModel
from pricing.pricing_engine import DynamicPricingEngine
from src.econocausal.pricing_optimizer import PricingOptimizer
from src.econocausal.data_generator import DynamicPricingDataGenerator, SimulationConfig


class TestPreprocessingAndModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample realistic data for testing
        cls.sample_df = pd.DataFrame({
            "product_id": ["PROD_001", "PROD_002", "PROD_001", "PROD_003"],
            "product_name": ["Headphones", "TV", "Headphones", "Chair"],
            "category": ["Electronics", "Electronics", "Electronics", "Home & Office"],
            "price": [2499.0, 24999.0, 2550.0, 6499.0],
            "cost": [1350.0, 16000.0, 1350.0, 3600.0],
            "demand": [45, 20, 42, 35],
            "competitor_price": [2520.0, 25500.0, 2580.0, 6700.0],
            "promotion": [0, 1, 0, 0],
            "holiday": [0, 0, 1, 0],
            "inventory": [150, 40, 120, 80],
            "temperature": [25.0, 22.0, 26.5, 24.0],
            "day_of_week": ["Monday", "Friday", "Wednesday", "Sunday"],
            "month": [6, 11, 7, 3],
            "season": ["Summer", "Autumn", "Summer", "Spring"],
        })

    def test_01_preprocessor_fit_transform(self):
        """Test DataPreprocessor fit_transform and output shapes."""
        preprocessor = DataPreprocessor()
        X, T, Y, feature_names = preprocessor.fit_transform(self.sample_df)

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(len(X), len(self.sample_df))
        self.assertEqual(len(T), len(self.sample_df))
        self.assertEqual(len(Y), len(self.sample_df))
        self.assertTrue(all(np.isfinite(X.flatten())))
        self.assertTrue(all(T > 0))
        self.assertTrue(all(Y >= 0))

    def test_02_preprocessor_robust_transform(self):
        """Test DataPreprocessor transform handles missing columns gracefully."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(self.sample_df)

        # Incomplete inference payload (missing several features)
        sparse_df = pd.DataFrame([{
            "price": 2499.0,
            "category": "Electronics",
        }])

        X, T, Y = preprocessor.transform(sparse_df)
        self.assertEqual(X.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertEqual(float(T[0]), 2499.0)

    def test_03_demand_model_prediction(self):
        """Test BaselineDemandModel predictions are finite, non-negative numbers."""
        dm = BaselineDemandModel()
        dm.load()

        infer_df = pd.DataFrame([{
            "price": 2499.0,
            "competitor_price": 2550.0,
            "promotion": 0,
            "holiday": 0,
            "inventory": 150,
            "temperature": 25.0,
            "day_of_week": "Wednesday",
            "month": 6,
            "season": "Summer",
            "category": "Electronics",
        }])

        preds = dm.predict(infer_df)
        self.assertEqual(len(preds), 1)
        self.assertGreaterEqual(float(preds[0]), 0.0)
        self.assertTrue(np.isfinite(float(preds[0])))

    def test_04_dml_model_causal_effect(self):
        """Test DoubleMLPricingModel returns valid causal parameters and confidence interval."""
        dml = DoubleMLPricingModel()
        dml.load()

        effect = dml.get_causal_effect()
        self.assertIn("ate", effect)
        self.assertIn("ci_lower", effect)
        self.assertIn("ci_upper", effect)
        self.assertLessEqual(effect["ci_lower"], effect["ci_upper"])

        # Test projected causal demand at candidate price
        projected = dml.predict_causal_demand(
            baseline_demand=50.0,
            baseline_price=2499.0,
            candidate_price=2700.0,
            custom_elasticity=-0.015,
        )
        self.assertGreaterEqual(projected, 0.0)
        self.assertLess(projected, 50.0)  # Demand should decrease with higher price under negative elasticity

    def test_05_dynamic_pricing_engine_optimization(self):
        """Test DynamicPricingEngine generates feasible prices and valid uplift metrics."""
        engine = DynamicPricingEngine()

        res = engine.optimize_price(
            current_price=2499.0,
            cost=1350.0,
            baseline_demand=48.0,
            causal_elasticity=-0.016,
            product_id="PROD_001",
            product_name="Pro Wireless Headphones",
        )

        self.assertIn("recommended_price", res)
        self.assertIn("expected_demand", res)
        self.assertIn("expected_profit", res)
        self.assertIn("candidate_grid", res)

        rec_price = res["recommended_price"]
        # Recommended price must respect marginal cost + minimum margin
        self.assertGreater(rec_price, 1350.0 * 1.10)
        # Expected demand must be non-negative
        self.assertGreaterEqual(res["expected_demand"], 1.0)
        # Expected profit must be finite
        self.assertTrue(np.isfinite(res["expected_profit"]))

    def test_06_pricing_engine_boundary_conditions(self):
        """Test pricing engine handles edge cases (e.g. low margins, near-zero cost)."""
        engine = DynamicPricingEngine(min_margin_pct=0.15)
        candidates = engine.generate_candidate_prices(current_price=100.0, cost=95.0)

        self.assertGreater(len(candidates), 0)
        for p in candidates:
            # Candidate prices must never violate min margin
            self.assertGreaterEqual(p, 95.0 * 1.15 - 1e-4)

    def test_07_econocausal_research_pipeline_components(self):
        """Test data generator and pricing optimizer from research package."""
        config = SimulationConfig(n_samples=500, random_state=42)
        gen = DynamicPricingDataGenerator(config)
        df, meta = gen.generate()

        self.assertEqual(len(df), 500)
        self.assertIn("price", df.columns)
        self.assertIn("demand", df.columns)
        self.assertIn("true_theta", df.columns)
        self.assertLess(meta["true_ate"], 0)

        opt = PricingOptimizer()
        opt_res = opt.optimize_price_analytical(
            baseline_price=75.0,
            baseline_demand=50.0,
            causal_elasticity=-1.85,
            unit_cost=45.0,
            objective="profit",
        )
        self.assertGreater(opt_res["optimal_price"], 45.0)
        self.assertGreater(opt_res["expected_demand"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
