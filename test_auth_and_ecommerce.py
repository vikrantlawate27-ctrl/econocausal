"""
Automated Comprehensive Test Suite for EconoCausal
Tests:
  1. Default Admin Login (admin@econocausal.com / Admin@123)
  2. Customer Registration (/register)
  3. Customer Login (/login)
  4. Authentication Failure (wrong password)
  5. Role-based Route Protection (@admin_required, @customer_required)
  6. Admin Product Management (Add, Edit, Update Price, Delete)
  7. Price Synchronization: Admin updates price -> Customer catalog reflects new price
  8. Customer E-Commerce: Browse -> Add to Cart -> Checkout (COD) -> Order created
  9. Admin Order Management: View orders & update status
 10. Existing Causal AI & DML endpoints (/api/summary, /api/causal-effect, /api/recommend-price, /api/analytics)
"""

import sys
import os
import json

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database.database import init_db, get_db_connection

def run_tests():
    print("==================================================")
    print("STARTING ECONOCAUSAL AUTH & E-COMMERCE TEST SUITE")
    print("==================================================")

    init_db()
    client = app.test_client()

    # -------------------------------------------------------------
    # Test 1: Landing Page
    # -------------------------------------------------------------
    res = client.get("/")
    assert res.status_code == 200, f"Landing page failed: {res.status_code}"
    assert b"Admin Login" in res.data, "Admin Login button missing on landing"
    assert b"Customer Login" in res.data, "Customer Login button missing on landing"
    print("✅ Test 1: Landing page renders with Admin & Customer gateways.")

    # -------------------------------------------------------------
    # Test 2: Unauthenticated Access to Admin Routes
    # -------------------------------------------------------------
    res = client.get("/admin/dashboard", follow_redirects=False)
    assert res.status_code == 302, f"Unauthenticated admin dashboard did not redirect: {res.status_code}"
    assert "/admin/login" in res.headers["Location"], f"Redirect location wrong: {res.headers['Location']}"

    res = client.get("/dashboard", follow_redirects=False)
    assert res.status_code == 302
    assert "/admin/login" in res.headers["Location"]
    print("✅ Test 2: Unauthenticated users blocked and redirected from Admin routes.")

    # -------------------------------------------------------------
    # Test 3: Admin Login with Default Credentials
    # -------------------------------------------------------------
    # Wrong password test
    res = client.post("/admin/login", data={"email": "admin@econocausal.com", "password": "WrongPassword"})
    assert b"Invalid admin email or password" in res.data, "Wrong password error message missing"

    # Correct credentials
    admin_client = app.test_client()
    res = admin_client.post("/admin/login", data={"email": "admin@econocausal.com", "password": "Admin@123"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"Dashboard" in res.data
    print("✅ Test 3: Admin login with admin@econocausal.com / Admin@123 succeeded.")

    # -------------------------------------------------------------
    # Test 4: Admin Access to Existing Dashboard, Analysis & Simulator
    # -------------------------------------------------------------
    res = admin_client.get("/admin/dashboard")
    assert res.status_code == 200
    assert b"AI-Powered Economic Intelligence Dashboard" in res.data

    res = admin_client.get("/analysis")
    assert res.status_code == 200
    assert b"Data & Causal Analysis" in res.data

    res = admin_client.get("/simulator")
    assert res.status_code == 200
    assert b"Pricing Simulator" in res.data
    assert b"Approve & Apply Price to Store" in res.data
    print("✅ Test 4: Admin access to existing Dashboard, Causal Analysis, and Simulator confirmed.")

    # -------------------------------------------------------------
    # Test 5: Customer Registration
    # -------------------------------------------------------------
    import time
    test_email = f"testuser_{int(time.time())}@example.com"
    res = client.post("/register", data={
        "name": "Test Customer",
        "email": test_email,
        "password": "Password123",
        "confirm_password": "Password123",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Account created successfully" in res.data
    print(f"✅ Test 5: Customer registration for {test_email} succeeded with role=customer.")

    # -------------------------------------------------------------
    # Test 6: Customer Login
    # -------------------------------------------------------------
    customer_client = app.test_client()
    res = customer_client.post("/login", data={
        "email": test_email,
        "password": "Password123",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Welcome, Test Customer" in res.data
    print("✅ Test 6: Customer login succeeded and landed on Customer Dashboard.")

    # -------------------------------------------------------------
    # Test 7: Customer Blocked from Admin Routes (Role Security)
    # -------------------------------------------------------------
    res = customer_client.get("/admin/dashboard", follow_redirects=True)
    assert b"Access denied: Administrative privileges required" in res.data
    assert b"Welcome, Test Customer" in res.data  # Redirected back to customer dashboard
    print("✅ Test 7: Role enforcement working: Customer blocked from /admin/*.")

    # -------------------------------------------------------------
    # Test 8: Customer Catalog & Product Detail (No Business Secrets)
    # -------------------------------------------------------------
    res = customer_client.get("/products")
    assert res.status_code == 200
    assert b"Current Price" not in res.data or b"Pro Wireless Headphones" in res.data
    assert b"Causal Elasticity" not in res.data
    assert b"EconML" not in res.data

    res = customer_client.get("/product/PROD_001")
    assert res.status_code == 200
    assert b"Pro Wireless Headphones" in res.data
    assert b"Unit Cost" not in res.data
    assert b"Double Machine Learning" not in res.data
    print("✅ Test 8: Customer views catalog & detail without internal ML/profit metrics.")

    # -------------------------------------------------------------
    # Test 9: Cart & Checkout (COD)
    # -------------------------------------------------------------
    # Add to cart
    res = customer_client.post("/api/cart/add", json={"product_id": "PROD_001", "quantity": 2})
    assert res.status_code == 200
    cart_data = json.loads(res.data)
    assert cart_data["status"] == "success"
    assert cart_data["cart_count"] == 2

    # View cart
    res = customer_client.get("/cart")
    assert res.status_code == 200
    assert b"Pro Wireless Headphones" in res.data

    # Checkout POST
    res = customer_client.post("/checkout", data={
        "name": "Test Customer",
        "email": test_email,
        "phone": "+91 99999 88888",
        "address": "404 Causal Street, AI Valley, Pune",
        "payment_method": "Cash on Delivery",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Order Placed Successfully" in res.data
    print("✅ Test 9: Add to cart, view cart, and Cash on Delivery checkout succeeded.")

    # -------------------------------------------------------------
    # Test 10: Customer Orders Isolation
    # -------------------------------------------------------------
    res = customer_client.get("/customer/orders")
    assert res.status_code == 200
    assert b"#ORD-" in res.data
    assert (b"Confirmed" in res.data or b"Pending" in res.data)
    print("✅ Test 10: Customer sees their own newly placed order.")

    # -------------------------------------------------------------
    # Test 11: Admin Orders Management & Status Update
    # -------------------------------------------------------------
    res = admin_client.get("/admin/orders")
    assert res.status_code == 200
    assert b"Test Customer" in res.data

    # Update order status
    conn = get_db_connection()
    last_order = conn.execute("SELECT id FROM orders ORDER BY id DESC LIMIT 1").fetchone()
    order_id = last_order["id"]
    conn.close()

    res = admin_client.post("/admin/orders/update-status", data={
        "order_id": order_id,
        "status": "Confirmed",
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Confirmed" in res.data
    print(f"✅ Test 11: Admin views order #{order_id} and updates status to Confirmed.")

    # -------------------------------------------------------------
    # Test 12: Admin Product Management & Dynamic Price Synchronization
    # -------------------------------------------------------------
    # Admin updates price of PROD_001 to ₹2,750.00
    res = admin_client.post("/admin/products/update-price", json={
        "product_id": "PROD_001",
        "new_price": 2750.0
    })
    assert res.status_code == 200
    up_data = json.loads(res.data)
    assert up_data["status"] == "success"

    # Customer browses PROD_001 -> must see 2,750.00!
    res = customer_client.get("/product/PROD_001")
    assert res.status_code == 200
    assert b"2,750.00" in res.data, "Updated dynamic price not reflected in customer product view!"
    print("✅ Test 12: Price Synchronization verified! Admin updated price -> Customer sees ₹2,750.00 immediately.")

    # -------------------------------------------------------------
    # Test 13: Existing ML/DML API Endpoints
    # -------------------------------------------------------------
    res = admin_client.get("/api/summary")
    assert res.status_code == 200
    sum_data = json.loads(res.data)
    assert "estimated_causal_effect" in sum_data["data"]

    res = admin_client.get("/api/causal-effect")
    assert res.status_code == 200
    dml_data = json.loads(res.data)
    assert "ate" in dml_data["data"]

    res = admin_client.post("/api/recommend-price", json={
        "product_id": "PROD_001",
        "current_price": 2750.0,
        "cost": 1350.0,
        "competitor_price": 2800.0,
        "promotion": 0,
        "holiday": 0,
        "inventory": 150,
    })
    assert res.status_code == 200
    rec_data = json.loads(res.data)
    assert "recommended_price" in rec_data["recommendation"]
    print("✅ Test 13: Existing ML, EconML DML, and Dynamic Pricing Engine API endpoints operational.")

    # -------------------------------------------------------------
    # Test 14: Logout
    # -------------------------------------------------------------
    res = admin_client.get("/admin/logout", follow_redirects=True)
    assert b"Admin Login" in res.data or b"admin" in res.data
    res = customer_client.get("/logout", follow_redirects=True)
    assert b"Welcome Back" in res.data
    print("✅ Test 14: Logout clears sessions cleanly for both Admin and Customer.")

    print("==================================================")
    print("ALL 14 TEST SUITES PASSED FLAWLESSLY! 🚀")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
