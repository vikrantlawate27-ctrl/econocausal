# EconoCausal: Quality Assurance & Test Verification Report

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Evaluation Standard:** Unit, Integration, and System Level End-to-End Verification  
**OS Environment:** Windows (PowerShell) | Python 3.12.10 | Flask 3.1.3  

---

## 1. Quality Assurance Objectives & Scope

The testing architecture for EconoCausal was designed to validate:
1. **Security & Role-Based Access Control:** Strict route guarding via `@login_required`, `@admin_required`, and `@customer_required`.
2. **Business Privacy Isolation:** Ensuring internal metrics (marginal unit costs, target profit margins, Double Machine Learning coefficients) are strictly sequestered to the administrative portal and never leaked into customer storefront views or client-side assets.
3. **Price Synchronization Integrity:** Real-time reflection of dynamic pricing approvals—when an admin approves an optimized price, customer catalog and product detail pages must immediately reflect the updated dynamic price.
4. **Order Lifecycle & Audit Trails:** End-to-end order processing (Cart $\to$ Checkout with Cash on Delivery $\to$ Order Creation $\to$ Multi-stage status updates from `Pending` through `Delivered` $\to$ Immutable audit history in `order_status_history`).
5. **Notification Fault Tolerance:** Verification that transactional email services safely fall back to console logging when SMTP credentials are unconfigured, preventing checkout crashes.
6. **Causal & Machine Learning Endpoints:** Verification that baseline Random Forest demand predictions and EconML DML causal parameters (ATE and 95% confidence intervals) return valid JSON schemas.

---

## 2. Initial Audit Failures & Root Cause Remediation

| Test / Component | Initial Symptom | Root Cause | Engineering Remediation | Status |
| :--- | :--- | :--- | :--- | :---: |
| `requirements.txt` | `unsupported mime type text/plain; charset=utf-16le` | PowerShell redirection (`>`) wrote UTF-16LE with BOM | Re-encoded to UTF-8 without BOM | **Fixed & Verified** |
| `test_endpoints.py` | `[WinError 10061] No connection could be made` | Script used `urllib.request` requiring external live web server | Refactored to in-memory WSGI `app.test_client()` with optional live fallback | **Fixed & Verified** |
| Checkout Email Dispatch | Possible network hang or unhandled exception on missing SMTP | Direct call to SMTP server with placeholder credentials | Wrapped with placeholder detector that logs simulated email safely | **Fixed & Verified** |
| Product Price Update Sync | UI stale price cache | Static product prices in customer view | Connected customer catalog queries directly to SQLite `current_price` column | **Fixed & Verified** |

---

## 3. Test Suites & Execution Details

### Suite 1: Authentication & E-Commerce Integration (`test_auth_and_ecommerce.py`)
- **Execution Command:** `.venv\Scripts\python test_auth_and_ecommerce.py`
- **Test Cases:**
  1. `Test 1`: Landing page renders public discovery gateways (Admin & Customer portals).
  2. `Test 2`: Unauthenticated access to `/admin/*` and `/dashboard` redirected to `/admin/login`.
  3. `Test 3`: Admin login with default credentials (`admin@econocausal.com` / `Admin@123`) and rejection of invalid passwords.
  4. `Test 4`: Admin access to AI Dashboard, Causal Analysis, and Real-Time Pricing Simulator.
  5. `Test 5`: Customer account self-registration (`/register`) with automatic `customer` role assignment.
  6. `Test 6`: Customer login (`/login`) landing on Customer Dashboard.
  7. `Test 7`: Role enforcement: customer blocked from `/admin/*` routes with 403 / redirect.
  8. `Test 8`: Customer catalog (`/products`) and product detail (`/product/<id>`) privacy isolation (no unit costs or DML coefficients exposed).
  9. `Test 9`: Persistent cart addition (`/api/cart/add`), cart view (`/cart`), and Cash on Delivery checkout (`/checkout`).
  10. `Test 10`: Customer order isolation (customer views only their own orders).
  11. `Test 11`: Admin order fulfillment and status update to `Confirmed`.
  12. `Test 12`: Dynamic Price Synchronization: Admin updates price of `PROD_001` to ₹2,750.00 $\to$ Customer immediately sees ₹2,750.00 on storefront.
  13. `Test 13`: ML and EconML DML API endpoints (`/api/summary`, `/api/causal-effect`, `/api/recommend-price`).
  14. `Test 14`: Session clearance and logout for both Admin and Customer accounts.

### Suite 2: Email Service & Order Tracking Lifecycle (`test_email_and_tracking.py`)
- **Execution Command:** `.venv\Scripts\python test_email_and_tracking.py`
- **Test Cases:**
  1. `test_01_email_service_functions`: Verifies order confirmation, status update, and delivery email generators with safe fallback logging.
  2. `test_02_database_order_status_history`: Validates database tracking schema, aggregations, and fulfillment status counts.
  3. `test_03_customer_order_and_tracking_flow`: Full customer order lifecycle through all statuses: `Confirmed` $\to$ `Processing` $\to$ `Shipped` $\to$ `Out for Delivery` $\to$ `Delivered` $\to$ `Cancelled`.
  4. `test_04_security_isolation`: Confirms that Customer A cannot view or tamper with Customer B's live tracking timeline (403 Forbidden / Access Denied).

### Suite 3: Flask Route & REST API Endpoint Suite (`test_endpoints.py`)
- **Execution Command:** `.venv\Scripts\python test_endpoints.py`
- **Test Cases:**
  1. `test_01_public_pages`: GET `/` and GET `/products` return 200 OK.
  2. `test_02_admin_pages_with_auth`: GET `/dashboard`, `/analysis`, `/simulator` return 200 OK with admin session.
  3. `test_03_api_products`: GET `/api/products` returns complete catalog with valid JSON.
  4. `test_04_api_summary`: GET `/api/summary` returns KPI totals, average prices, and ATE.
  5. `test_05_api_causal_effect`: GET `/api/causal-effect` returns DML ATE and 95% confidence interval bounds.
  6. `test_06_api_analytics`: GET `/api/analytics` returns structured data arrays for all 5 Chart.js dashboard charts.
  7. `test_07_api_predict`: POST `/api/predict` returns valid supervised point demand forecast.
  8. `test_08_api_recommend_price`: POST `/api/recommend-price` evaluates price candidate grid and outputs profit-maximizing recommendation and uplift percentages.

---

## 4. Test Verification Matrix

| Suite File | Tests Defined | Tests Passed | Tests Failed | Quality Assessment |
| :--- | :---: | :---: | :---: | :---: |
| `test_auth_and_ecommerce.py` | 14 | 14 | 0 | **100% Passed (Full Stack Verified)** |
| `test_email_and_tracking.py` | 4 | 4 | 0 | **100% Passed (Fulfillment Verified)** |
| `test_endpoints.py` | 8 | 8 | 0 | **100% Passed (API Surface Verified)** |
| **Total** | **26** | **26** | **0** | **Zero Regressions / Production Ready** |
