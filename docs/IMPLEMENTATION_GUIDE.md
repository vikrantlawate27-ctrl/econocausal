# EconoCausal: End-to-End Implementation & Developer Guide

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Target Environment:** Windows 10/11 (PowerShell), Python 3.10+  
**Stack:** Flask, EconML, Scikit-Learn, SQLite, Streamlit, Chart.js  

---

## 1. System Architecture Overview

EconoCausal combines microeconomic causal modeling with a production-grade full-stack web application:

```
                  ┌──────────────────────────────────────────────┐
                  │       Historical Sales & Pricing Data        │
                  │             (data/sales_data.csv)            │
                  └──────────────────────┬───────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
        ┌────────────────────────────────┐    ┌───────────────────────────────────┐
        │  Data Engineering & Pipeline   │    │  Database Initialization & Seeds  │
        │    (models/preprocessing.py)   │    │      (database/database.py)       │
        └────────────────┬───────────────┘    └─────────────────┬─────────────────┘
                         │                                      │
         ┌───────────────┴───────────────┐                      │
         ▼                               ▼                      │
┌──────────────────┐           ┌──────────────────┐             │
│ Baseline Demand  │           │   Double ML      │             │
│  Random Forest   │           │   LinearDML      │             │
│ (demand_model.py)│           │  (dml_model.py)  │             │
└────────┬─────────┘           └────────┬─────────┘             │
         │                              │                       │
         └───────────────┬──────────────┘                       │
                         ▼                                      │
        ┌────────────────────────────────┐                      │
        │     Dynamic Pricing Engine     │                      │
        │   (pricing/pricing_engine.py)  │                      │
        └────────────────┬───────────────┘                      │
                         │                                      │
                         ▼                                      ▼
        ┌────────────────────────────────────────────────────────┐
        │                 Flask Web Application                  │
        │                       (app.py)                         │
        ├────────────────────────────────────────────────────────┤
        │ 1. Public Storefront (/ , /products, /product/<id>)     │
        │ 2. Role-Based Auth (/login, /admin/login, /register)   │
        │ 3. Admin Portal (/admin/dashboard, /admin/orders)      │
        │ 4. Dynamic Pricing Simulator (/simulator)              │
        │ 5. Customer Cart & Checkout (/cart, /checkout, /track) │
        │ 6. REST API Endpoints (/api/summary, /api/predict...)  │
        └────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Setup on Windows (PowerShell)

### Step 1: Clone or Open Project Directory
```powershell
cd c:\Users\mauli\Downloads\project\intership
```

### Step 2: Activate Python Virtual Environment
```powershell
# Activate the existing verified virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Dependencies
The requirements have been verified and converted to UTF-8 without BOM:
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` (already created with safe dev defaults):
```powershell
Copy-Item .env.example .env
```

---

## 3. Data Pipeline & Model Training Execution

To train both the Baseline Random Forest Demand Regressor and the EconML Double Machine Learning Model:

```powershell
.venv\Scripts\python train_models.py
```

### What this command does:
1. Validates `data/sales_data.csv` (auto-generates 12,000 transactions if missing).
2. Initializes SQLite tables and seeds the default products and accounts in `database/econocausal.db`.
3. Trains `BaselineDemandModel` and saves serialized artifact `models/demand_rf.joblib`.
4. Executes $K$-fold cross-fitting and Robinson residual-on-residual regression via `DoubleMLPricingModel`, serializing `models/dml_model.joblib`.
5. Outputs estimated ATE and 95% confidence intervals to terminal.

---

## 4. Running the Web Application

Launch the Flask server:
```powershell
.venv\Scripts\python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### Pre-Configured Access Credentials:
- **System Administrator:**
  - Email: `admin@econocausal.com`
  - Password: `Admin@123`
  - Access: Executive Dashboard (`/dashboard`), Causal Analysis (`/analysis`), Pricing Simulator (`/simulator`), Order Management (`/admin/orders`), Product Management (`/admin/products`).
- **Customer Account:**
  - Email: `customer@econocausal.com`
  - Password: `Customer@123`
  - Access: Customer Home (`/customer/home`), Store Catalog (`/products`), Shopping Cart (`/cart`), Checkout (`/checkout`), Live Tracking (`/customer/orders/<id>/track`).

---

## 5. Running the Streamlit Research Dashboard

To explore the academic CATE analysis and counterfactual simulation:
```powershell
.venv\Scripts\streamlit run dashboard/app.py
```
Opens automatically on **`http://localhost:8501`**.

---

## 6. Automated Testing Verification

Execute all three test suites:

```powershell
# 1. Test Authentication, Role Security, and E-Commerce Flows
.venv\Scripts\python test_auth_and_ecommerce.py

# 2. Test Order Tracking, Status History, and Email Notification Fallbacks
.venv\Scripts\python test_email_and_tracking.py

# 3. Test Flask Web Routes & REST API Endpoints (In-Memory WSGI)
.venv\Scripts\python test_endpoints.py
```

---

## 7. REST API Documentation & Example Requests

### 1. Portfolio Summary & Causal Effect
**Request:** `GET /api/summary`
```json
{
  "status": "success",
  "data": {
    "total_products": 6,
    "total_sales_records": 12000,
    "avg_price": 8949.0,
    "estimated_causal_effect": -0.0152,
    "ci_lower": -0.0185,
    "ci_upper": -0.0119
  }
}
```

### 2. Recommend Price (Dynamic Pricing Engine)
**Request:** `POST /api/recommend-price`
```json
{
  "product_id": "PROD_001",
  "current_price": 2499.0,
  "cost": 1350.0,
  "competitor_price": 2580.0,
  "promotion": 0,
  "holiday": 0,
  "inventory": 160
}
```
**Response:**
```json
{
  "status": "success",
  "recommendation": {
    "recommended_price": 2549.0,
    "expected_demand": 49.1,
    "expected_revenue": 125155.9,
    "expected_profit": 58870.9,
    "profit_uplift_pct": 5.4,
    "revenue_maximizing_price": 2499.0,
    "causal_elasticity": -0.0152
  }
}
```
