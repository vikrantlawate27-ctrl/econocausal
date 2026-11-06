"""
Automated Route and REST API Testing for EconoCausal
Supports in-memory testing via Flask test_client() as well as optional live server testing.
"""

import sys
import os
import json
import unittest

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database.database import init_db


class TestEconoCausalEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.test_client()
        # Create an authenticated admin client
        cls.admin_client = app.test_client()
        cls.admin_client.post("/admin/login", data={
            "email": "admin@econocausal.com",
            "password": "Admin@123"
        })

    def test_01_public_pages(self):
        """Test public landing and catalog pages."""
        for path in ["/", "/products"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"GET {path} failed with {res.status_code}")

    def test_02_admin_pages_with_auth(self):
        """Test admin pages when logged in."""
        for path in ["/dashboard", "/analysis", "/simulator"]:
            res = self.admin_client.get(path, follow_redirects=True)
            self.assertEqual(res.status_code, 200, f"GET {path} failed with {res.status_code}")

    def test_03_api_products(self):
        """Test /api/products returns product list."""
        res = self.client.get("/api/products")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("products", data)
        self.assertGreater(len(data["products"]), 0)

    def test_04_api_summary(self):
        """Test /api/summary returns KPI metrics for authenticated admin."""
        # Unauthenticated access must redirect
        unauth_res = self.client.get("/api/summary")
        self.assertEqual(unauth_res.status_code, 302)

        res = self.admin_client.get("/api/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("data", data)
        summary = data["data"]
        self.assertIn("total_products", summary)
        self.assertIn("total_sales_records", summary)

    def test_05_api_causal_effect(self):
        """Test /api/causal-effect returns ATE and 95% CI."""
        res = self.admin_client.get("/api/causal-effect")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("data", data)
        self.assertIn("ate", data["data"])

    def test_06_api_analytics(self):
        """Test /api/analytics returns chart data."""
        res = self.admin_client.get("/api/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("analytics", data)

    def test_07_api_predict(self):
        """Test /api/predict demand prediction POST."""
        payload = {
            "price": 2499.0,
            "competitor_price": 2550.0,
            "promotion": 1,
            "holiday": 0,
            "inventory": 180,
            "temperature": 25.0,
            "day_of_week": "Wednesday",
            "month": 6,
            "season": "Summer",
            "category": "Electronics"
        }
        res = self.admin_client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("predicted_demand", data)

    def test_08_api_recommend_price(self):
        """Test /api/recommend-price dynamic pricing POST."""
        payload = {
            "product_id": "PROD_001",
            "current_price": 2499.0,
            "cost": 1350.0,
            "competitor_price": 2580.0,
            "promotion": 0,
            "holiday": 0,
            "inventory": 160
        }
        res = self.admin_client.post("/api/recommend-price", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("recommendation", data)
        rec = data["recommendation"]
        self.assertIn("recommended_price", rec)
        self.assertIn("expected_demand", rec)
        self.assertIn("expected_profit", rec)


def run_standalone():
    print("=" * 60)
    print("TESTING FLASK APPLICATION ROUTES & APIS")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEconoCausalEndpoints)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print("ALL FLASK ROUTES TESTED SUCCESSFULLY")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_standalone()
    if not success:
        sys.exit(1)
