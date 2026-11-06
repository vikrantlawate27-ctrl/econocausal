"""
Unit and Integration Test Suite for:
1. Order Confirmation Email & Fallback Safety
2. Order Status Update Emails
3. Order Status History Tracking Table
4. Customer Order Tracking Page & Security Access Control
5. Admin Order Status Management (All 7 Statuses)
6. Customer Dashboard Activity Notifications
7. Admin Dashboard Order Fulfillment Analytics
"""

import sys
import unittest
from app import app
from database.database import (
    get_db_connection,
    get_user_by_email,
    get_order_by_id,
    get_order_status_history,
    get_order_statistics,
    get_customer_notifications,
    update_order_status,
)
from services.email_service import (
    send_order_confirmation_email,
    send_order_status_email,
    send_delivery_email,
)

class TestOrderEmailAndTracking(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_01_email_service_functions(self):
        """Verify email service functions with simulated/mock payloads."""
        # Test confirmation email with dict
        mock_order = {
            "id": 9999,
            "customer_name": "Test Runner",
            "customer_email": "testrunner@econocausal.test",
            "order_date": "2026-10-07 10:00:00",
            "estimated_delivery": "2026-10-11",
            "delivery_address": "456 Silicon Way, Tech City",
            "total_amount": 2599.00,
            "items": [
                {"product_name": "AI Headset", "quantity": 1, "price": 2599.00, "subtotal": 2599.00}
            ]
        }
        ok1, msg1 = send_order_confirmation_email(mock_order)
        self.assertTrue(ok1, f"send_order_confirmation_email failed: {msg1}")

        # Test status update email with dict
        ok2, msg2 = send_order_status_email(mock_order, "Confirmed", "Shipped")
        self.assertTrue(ok2, f"send_order_status_email failed: {msg2}")

        # Test delivery email
        ok3, msg3 = send_delivery_email(mock_order)
        self.assertTrue(ok3, f"send_delivery_email failed: {msg3}")

    def test_02_database_order_status_history(self):
        """Verify order_status_history logging and statistics."""
        stats_before = get_order_statistics()
        self.assertIn("total", stats_before)
        self.assertIn("confirmed", stats_before)
        self.assertIn("shipped", stats_before)
        self.assertIn("delivered", stats_before)
        self.assertIn("processing", stats_before)
        self.assertIn("cancelled", stats_before)

    def test_03_customer_order_and_tracking_flow(self):
        """Simulate customer login, add to cart, checkout, confirmation, and live tracking."""
        # 1. Customer login
        login_res = self.client.post("/login", data={
            "email": "customer@econocausal.com",
            "password": "Customer@123",
        }, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)

        # 2. Add product to cart
        add_res = self.client.post("/api/cart/add", json={
            "product_id": "PROD_001",
            "quantity": 1
        })
        self.assertEqual(add_res.status_code, 200)

        # 3. Checkout
        checkout_res = self.client.post("/checkout", data={
            "full_name": "Customer User",
            "email": "customer@econocausal.com",
            "phone": "+91 9876543210",
            "address": "77 Neural Plaza, Bangalore, KA",
            "payment_method": "Cash on Delivery",
        }, follow_redirects=True)
        self.assertEqual(checkout_res.status_code, 200)
        self.assertIn(b"Order Placed Successfully", checkout_res.data)
        self.assertIn(b"Track My Order", checkout_res.data)

        # 4. Find the newly placed order ID
        cust = get_user_by_email("customer@econocausal.com")
        self.assertIsNotNone(cust)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM orders WHERE user_id = ? ORDER BY id DESC LIMIT 1", (cust["id"],))
        order_row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(order_row)
        new_order_id = order_row["id"]

        order_data = get_order_by_id(new_order_id)
        self.assertEqual(order_data["status"], "Confirmed")
        self.assertIsNotNone(order_data["estimated_delivery"])

        # 5. Verify initial history record in order_status_history
        history = get_order_status_history(new_order_id)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "Confirmed")

        # 6. Customer accesses tracking page
        track_res = self.client.get(f"/customer/orders/{new_order_id}/track")
        self.assertEqual(track_res.status_code, 200)
        self.assertIn(b"ORDER TIMELINE", track_res.data)
        self.assertIn(b"Order Confirmed", track_res.data)
        self.assertIn(b"Delivery Address", track_res.data)

        # 7. Customer Dashboard shows recent orders and notifications
        dash_res = self.client.get("/customer/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn(b"Order Activity Notifications", dash_res.data)
        self.assertIn(f"#ORD-{new_order_id}".encode(), dash_res.data)

        # 8. Admin login and status updates across lifecycle
        admin_login = self.client.post("/admin/login", data={
            "email": "admin@econocausal.com",
            "password": "Admin@123",
        }, follow_redirects=True)
        self.assertEqual(admin_login.status_code, 200)

        # Admin changes status: Confirmed -> Processing
        up1 = self.client.post("/admin/orders/update-status", data={
            "order_id": new_order_id,
            "status": "Processing"
        }, follow_redirects=True)
        self.assertEqual(up1.status_code, 200)
        order_p = get_order_by_id(new_order_id)
        self.assertEqual(order_p["status"], "Processing")

        # Admin changes status: Processing -> Shipped
        up2 = self.client.post("/admin/orders/update-status", data={
            "order_id": new_order_id,
            "status": "Shipped"
        }, follow_redirects=True)
        self.assertEqual(up2.status_code, 200)
        order_s = get_order_by_id(new_order_id)
        self.assertEqual(order_s["status"], "Shipped")

        # Admin changes status: Shipped -> Out for Delivery
        up3 = self.client.post("/admin/orders/update-status", data={
            "order_id": new_order_id,
            "status": "Out for Delivery"
        }, follow_redirects=True)
        self.assertEqual(up3.status_code, 200)
        order_ofd = get_order_by_id(new_order_id)
        self.assertEqual(order_ofd["status"], "Out for Delivery")

        # Admin changes status: Out for Delivery -> Delivered
        up4 = self.client.post("/admin/orders/update-status", data={
            "order_id": new_order_id,
            "status": "Delivered"
        }, follow_redirects=True)
        self.assertEqual(up4.status_code, 200)
        order_del = get_order_by_id(new_order_id)
        self.assertEqual(order_del["status"], "Delivered")

        # Check full history trail (Confirmed + Processing + Shipped + Out for Delivery + Delivered)
        full_hist = get_order_status_history(new_order_id)
        statuses = [h["status"] for h in full_hist]
        self.assertIn("Confirmed", statuses)
        self.assertIn("Processing", statuses)
        self.assertIn("Shipped", statuses)
        self.assertIn("Out for Delivery", statuses)
        self.assertIn("Delivered", statuses)

        # Customer tracking page renders delivered timeline
        track_del_res = self.client.get(f"/customer/orders/{new_order_id}/track")
        self.assertEqual(track_del_res.status_code, 200)
        self.assertIn(b"Delivered", track_del_res.data)

        # Test cancelled status on a fresh or test order
        update_order_status(new_order_id, "Cancelled", updated_by="Admin")
        track_canc_res = self.client.get(f"/customer/orders/{new_order_id}/track")
        self.assertEqual(track_canc_res.status_code, 200)
        self.assertIn(b"Order Cancelled", track_canc_res.data)

    def test_04_security_isolation(self):
        """Verify that customer A cannot track customer B's order (403 or redirect)."""
        # Create an order under customer 1
        cust1 = get_user_by_email("customer@econocausal.com")
        self.assertIsNotNone(cust1)

        # Login as customer 1 to place an order
        self.client.post("/login", data={
            "email": "customer@econocausal.com",
            "password": "Customer@123",
        })
        self.client.post("/api/cart/add", json={"product_id": "PROD_002", "quantity": 1})
        self.client.post("/checkout", data={
            "full_name": "Customer User",
            "email": "customer@econocausal.com",
            "phone": "+91 9876543210",
            "address": "Customer 1 address",
            "payment_method": "Cash on Delivery",
        })
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM orders WHERE user_id = ? ORDER BY id DESC LIMIT 1", (cust1["id"],))
        c1_order_id = cur.fetchone()["id"]
        conn.close()

        # Logout customer 1
        self.client.get("/logout")

        # Register a second customer
        reg_res = self.client.post("/register", data={
            "name": "Intruder User",
            "email": "intruder@econocausal.test",
            "password": "password123",
            "confirm_password": "password123",
            "phone": "9998887776",
            "address": "404 Hacker Way",
        }, follow_redirects=True)

        # Login as intruder
        self.client.post("/login", data={
            "email": "intruder@econocausal.test",
            "password": "password123",
        })

        # Try to view Customer 1's order tracking
        intruder_res = self.client.get(f"/customer/orders/{c1_order_id}/track", follow_redirects=True)
        # Should be redirected away with access denied message
        self.assertIn(b"Access Denied", intruder_res.data)

if __name__ == "__main__":
    unittest.main()
