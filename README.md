# EconoCausal: Dynamic Pricing via Double Machine Learning
### An AI-Powered Economic Intelligence & Dynamic Pricing Framework

![Project Banner](https://img.shields.io/badge/Project-EconoCausal-blue?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-Python_%7C_Flask_%7C_Streamlit_%7C_Scikit--Learn_%7C_SQLite_%7C_Chart.js-emerald?style=for-the-badge)
![Testing](https://img.shields.io/badge/Tests-All_Passed_(100%25)-brightgreen?style=for-the-badge)

---

## 📌 1. Project Overview
**EconoCausal** is an end-to-end Machine Learning, Econometric Causal Inference, and Dynamic Pricing platform. It integrates:
1. **Flask Full-Stack Web Application (`app.py`)**: A production-style role-based e-commerce platform and executive pricing intelligence suite with:
   - Public product discovery and catalog browsing (`/`, `/products`, `/product/<id>`).
   - Role-based authentication (`/login`, `/admin/login`, `/register`).
   - Admin management portal for pricing optimization, simulator, inventory, sales, and order fulfillment (`/dashboard`, `/analysis`, `/simulator`, `/admin/*`).
   - Customer e-commerce shopping experience with cart, cash-on-delivery checkout, and live order tracking (`/cart`, `/checkout`, `/customer/orders/<id>/track`).
   - Automated transactional email notification service with live or mock/dev logging.
   - REST API endpoints for demand forecasting, causal elasticity estimation, and price recommendations (`/api/*`).
2. **Streamlit Interactive Research Dashboard (`dashboard/app.py`)**: An analytical demonstration dashboard evaluating counterfactual pricing policies, heterogeneous treatment effects (CATE), and microeconomic profit optimization.
3. **Double Machine Learning Core (`src/econocausal/` & `models/`)**: Robinson's partially linear orthogonalization algorithm using cross-fitting to eliminate omitted-variable and simultaneity biases from observational retail sales data.

---

## 📂 2. Actual Project Folder Structure

```
intership/
│
├── app.py                             # Main Flask Web Application & REST API Server
├── train_models.py                    # Model training script for Flask ML models
├── requirements.txt                   # Tested Python package dependencies
├── README.md                          # Complete project documentation & run guide
├── .env.example                       # Environment variables template
├── .env                               # Local configuration (email, base URL, secrets)
│
├── dashboard/
│   └── app.py                         # Streamlit interactive dynamic pricing demonstration app
│
├── data/
│   ├── build_dataset.py               # Deterministic fast sales dataset generator
│   ├── generate_data.py               # Synthetic data generator with endogenous pricing
│   ├── sales_data.csv                 # 12,000 retail sales transactions across 6 products
│   └── dynamic_pricing_transactions.csv # 30,000 transactions for research pipeline
│
├── database/
│   ├── database.py                    # SQLite schema migration, seeds, cart & order helpers
│   └── econocausal.db                 # Persistent SQLite relational database
│
├── models/
│   ├── preprocessing.py               # Data cleaner, imputer & One-Hot encoder
│   ├── demand_model.py                # Supervised RandomForest demand predictor
│   ├── dml_model.py                   # Robinson LinearDML causal elasticity estimator
│   ├── demand_rf.joblib               # Serialized baseline demand model artifact
│   └── dml_model.joblib               # Serialized causal DML model artifact
│
├── pricing/
│   └── pricing_engine.py              # Candidate price generator & profit/revenue optimizer
│
├── services/
│   └── email_service.py               # SMTP transactional email service (Confirmation, Tracking, Delivery)
│
├── src/
│   └── econocausal/                   # Modular research & simulation package
│       ├── __init__.py                # Package metadata
│       ├── data_generator.py          # Microeconomic data generator with true causal ATE
│       ├── double_ml.py               # Chernozhukov Double Machine Learning implementation
│       ├── naive_models.py            # Naive OLS & Supervised ML benchmark models
│       ├── pricing_optimizer.py       # Analytical microeconomic pricing optimizer
│       ├── counterfactual_eval.py     # Counterfactual policy simulator & visualizer
│       ├── eda.py                     # Exploratory Data Analysis & confounding visualizer
│       └── pipeline.py                # End-to-end benchmark pipeline execution script
│
├── outputs/                           # Generated research artifacts
│   ├── figures/                       # Publication-quality evaluation PNG charts
│   ├── models/                        # Serialized research DML model artifacts
│   └── reports/                       # Counterfactual metrics JSON & CSV exports
│
├── templates/                         # 21 Flask Jinja2 HTML Templates
│   ├── index.html                     # Landing & gateway page
│   ├── login.html, register.html      # Customer authentication
│   ├── admin_login.html               # Administrative login
│   ├── dashboard.html                 # Executive AI pricing dashboard
│   ├── analysis.html                  # Causal and exploratory data analysis
│   ├── simulator.html                 # Interactive price simulation & approval
│   ├── admin_products.html            # Product inventory & price management
│   ├── admin_orders.html              # Order fulfillment & status updates
│   ├── admin_sales.html               # Historical sales transactions
│   ├── customer_dashboard.html        # Customer account dashboard
│   ├── customer_products.html         # Product catalog with filters
│   ├── customer_product_detail.html   # Customer product view (no internal margins)
│   ├── customer_cart.html             # Shopping cart
│   ├── customer_checkout.html         # Checkout flow
│   ├── customer_order_success.html    # Order receipt
│   └── customer_order_track.html      # Visual chronological order tracking
│
├── static/                            # Static CSS, JS, and media assets
│   ├── css/style.css                  # Modern UI design system
│   └── js/dashboard.js                # Frontend controllers & Chart.js rendering
│
├── test_endpoints.py                  # Flask route & REST API integration tests
├── test_auth_and_ecommerce.py         # Role security & e-commerce workflow tests
├── test_email_and_tracking.py         # Email notification & order tracking tests
└── test_models_and_pricing.py         # ML models, preprocessing, and pricing engine unit tests
```

---

## 💻 3. Prerequisites & Environment Setup

### System Prerequisites
- **Operating System:** Windows 10/11 (or macOS / Linux)
- **Python Version:** Python 3.10 to 3.12 (Python 3.12 recommended)
- **PowerShell / Terminal**

### Virtual Environment Setup on Windows

Open PowerShell or VS Code terminal in the project directory:

```powershell
# 1. Verify Python installation
python --version

# 2. If .venv does not exist, create it:
python -m venv .venv

# 3. Activate the virtual environment:
.\.venv\Scripts\Activate.ps1
```

> **Note on Windows PowerShell Execution Policy:**
> If PowerShell displays `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system`, bypass it safely for the current terminal session without weakening system-wide settings:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
> Alternatively, you can run commands directly using `.\.venv\Scripts\python.exe`.

---

## 📦 4. Dependency Installation

Verify and install the required dependencies:

```powershell
# Verify pip in virtual environment
python -m pip install --upgrade pip

# Install project dependencies
python -m pip install -r requirements.txt

# Verify dependency integrity
python -m pip check
```

---

## ⚙️ 5. Configuration & Environment Variables

The project includes `.env.example` and a pre-configured `.env` file:

```ini
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_USE_TLS=True
MAIL_DEFAULT_SENDER="EconoCausal Orders <your-email@gmail.com>"
APP_BASE_URL=http://127.0.0.1:5000
```

- When using placeholder credentials, the email service operates in safe **Mock/Dev Mode**, logging nicely formatted email notifications to the console without failing or blocking order placement.
- To send live emails, configure real SMTP credentials (such as a Gmail 16-character App Password).

---

## 🚀 6. Model Training & Pipeline Execution

### Train Flask Application Models
To train the baseline demand prediction model and Double ML model for the Flask application:

```powershell
python train_models.py
```
This initializes the SQLite database, checks `data/sales_data.csv`, and serializes `models/demand_rf.joblib` and `models/dml_model.joblib`.

### Run the Research & Counterfactual Benchmark Pipeline
To generate the research dataset and model artifacts required for the Streamlit dashboard:

```powershell
python src\econocausal\pipeline.py
```
This runs the 6-step workflow:
1. Generates 30,000 synthetic transactions with price endogeneity (`data/dynamic_pricing_transactions.csv`).
2. Generates exploratory confounder charts in `outputs/figures/`.
3. Partitions data into 80/20 train/test splits.
4. Fits Naive Bivariate OLS, Naive Multivariate OLS, and Supervised ML benchmarks.
5. Fits EconoCausal Double Machine Learning (DML) with 5-fold cross-fitting and saves `outputs/models/dml_pricing_model.joblib`.
6. Simulates counterfactual pricing policies and exports performance metrics to `outputs/reports/`.

---

## 🌐 7. How to Run the Applications

### Option A: Run the Primary Full-Stack Flask Web Application (Recommended)

Run the Flask server:
```powershell
python app.py
```

- **Local URL:** 👉 **`http://127.0.0.1:5000`**
- **Admin Gateway:** Navigate to `http://127.0.0.1:5000/admin/login`
  - **Email:** `admin@econocausal.com`
  - **Password:** `Admin@123`
- **Customer Gateway:** Navigate to `http://127.0.0.1:5000/login`
  - **Email:** `customer@econocausal.com`
  - **Password:** `Customer@123`
  - *(Or click "Register" to create a new customer account)*

### Option B: Run the Interactive Streamlit Research Dashboard

Open a separate terminal window and run:
```powershell
streamlit run dashboard/app.py
```

- **Local URL:** 👉 **`http://localhost:8501`**
- Provides interactive sliders for customer tiers, urgency scores, competitor pricing, seasonality, and visualizes counterfactual profit curves.

---

## 🧪 8. How to Run Automated Unit Tests

The project includes four automated test suites covering all routes, authentication, e-commerce flows, ML models, and pricing engine logic.

### Run All Unit Tests
```powershell
python -m unittest test_endpoints.py test_models_and_pricing.py test_email_and_tracking.py test_auth_and_ecommerce.py
```

### Run Individual Test Suites
```powershell
# 1. Test Flask REST APIs and Routes
python test_endpoints.py

# 2. Test ML Models, Preprocessing, and Pricing Optimizer
python test_models_and_pricing.py

# 3. Test Email Notifications and Chronological Order Tracking
python test_email_and_tracking.py

# 4. Test Role-Based Authentication and Full E-Commerce Flow
python test_auth_and_ecommerce.py
```

---

## 🛠️ 9. Common Errors & Solutions

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `File Activate.ps1 cannot be loaded because running scripts is disabled` | Windows PowerShell execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, or invoke `.\.venv\Scripts\python.exe`. |
| `ModuleNotFoundError: No module named 'econocausal'` | Python path does not include the `src` directory when running scripts directly | Use `python src\econocausal\pipeline.py` (which now has automatic `sys.path` resolution). |
| `NameError: name 'RandomForestRegressor' is not defined` in `dml_model.py` | Missing import in `models/dml_model.py` | Resolved: Added `from sklearn.ensemble import RandomForestRegressor` to `dml_model.py`. |
| `RuntimeError: Model is not fitted yet` inside `double_ml.py` | `self.is_fitted = True` was placed after `self.predict_cate()` in `fit()` | Resolved: Set `self.is_fitted = True` before evaluating `self.predict_cate()`. |
| Headless Matplotlib blocking on script execution | Matplotlib default GUI backend attempting to create a window | Resolved: Added `matplotlib.use("Agg")` in `eda.py` and `counterfactual_eval.py`. |
| `AssertionError` in price boundary candidate generation | `generate_candidate_prices` added current price even if below minimum margin floor | Resolved: Candidate prices now strictly filter and enforce `min_margin_floor`. |

---

## 📜 10. Methodology & Academic Grounding

1. **Robinson's Transformation (1988):** Decomposes demand into structural causal effect $\theta \cdot T$ and non-parametric confounding nuisance function $g(W)$.
2. **Neyman Orthogonality (Chernozhukov et al., 2018):** Guarantees that first-stage estimation errors in predicting demand $\hat{Y}(W)$ and price $\hat{T}(W)$ have zero first-order derivative effect on the causal parameter $\hat{\theta}$.
3. **Cross-Fitting:** Partitions data into $K$ disjoint folds, training nuisance models on out-of-fold data to eliminate overfitting bias of order $1/\sqrt{N}$.
4. **Microeconomic Optimization:** Determines optimal prices $\frac{d\Pi}{dP} = 0 \implies P^* = \frac{P_0 + c}{2} - \frac{Q_0}{2\theta}$ bounded by business constraints.

---

**Developed for Internship & Academic Evaluation | ISO 9001:2015 & MSME Certified Deliverable**
