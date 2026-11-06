# EconoCausal: Project Audit & Architectural Analysis

**Project Name:** EconoCausal: Dynamic Pricing via Double Machine Learning  
**Internship Role:** Data Associate L1 Intern (Data Science & Machine Learning)  
**Organization:** Infotact Solutions  
**Audit Date:** October 2026  
**Auditor / Reviewer:** Senior AI Engineer & Project Mentor  

---

## 1. Executive Overview

This audit comprehensively evaluates the existing state of **EconoCausal**, an end-to-end Machine Learning, Causal Inference, and Dynamic Pricing system. The system addresses the microeconomic challenge of **price endogeneity** in retail/e-commerce transaction logs using **Double Machine Learning (DML)** (Chernozhukov et al., 2018; Robinson, 1988) and translates causal demand elasticity into profit- and revenue-maximizing pricing strategies with operational business guardrails.

The workspace consists of two integrated tracks:
1. **The Production Enterprise Web Application (Flask + SQLite + Chart.js):**
   - Full-featured e-commerce customer storefront (catalog, search, product details, persistent cart, checkout with Cash on Delivery).
   - Order tracking with timeline lifecycle states (`Pending` -> `Confirmed` -> `Processing` -> `Shipped` -> `Out for Delivery` -> `Delivered` / `Cancelled`).
   - Transactional notification service (with SMTP delivery and console simulation fallback).
   - Role-based authentication (`admin` and `customer` security decorators).
   - Executive AI economic intelligence dashboard, causal analytics visualizations, and real-time interactive pricing simulator.
   - RESTful APIs (`/api/products`, `/api/summary`, `/api/predict`, `/api/recommend-price`, `/api/causal-effect`, `/api/analytics`).
2. **The Rigorous Data Science & Causal Research Module (`src/econocausal/` & `dashboard/app.py`):**
   - Synthetic ground-truth data generator with parametric price endogeneity and heterogeneous treatment effects (`DynamicPricingDataGenerator`).
   - Exploratory Data Analysis & confounding diagnostic suite (`PricingEDA`).
   - Naive benchmarks: Bivariate OLS, Multivariate OLS, and Supervised ML Random Forest (`NaiveOLSPricing`, `NaiveSupervisedMLPricing`).
   - Chernozhukov Double Machine Learning implementation with $K$-fold cross-fitting and Neyman orthogonalization (`DoubleMachineLearningPricing`).
   - Analytical profit optimizer with business guardrails (`PricingOptimizer`).
   - Counterfactual policy evaluation comparing gross profit uplift across pricing policies (`CounterfactualEvaluator`).
   - Streamlit interactive research dashboard (`dashboard/app.py`).

---

## 2. Inventory of Existing Features

| Feature Area | Component | Implementation Status | Description |
| :--- | :--- | :---: | :--- |
| **Authentication & RBAC** | `app.py`, `database.py` | **Complete** | Password hashing (PBKDF2/Werkzeug), session management, `@login_required`, `@admin_required`, `@customer_required`. |
| **Relational Database** | `database/database.py` | **Complete** | SQLite relational schema: `products`, `sales`, `predictions`, `users`, `orders`, `order_items`, `cart`, `order_status_history`. Schema auto-migration and initial seed data. |
| **Product & Order Management** | `app.py`, `templates/admin_*.html` | **Complete** | Admin product CRUD, dynamic price synchronization, order status transitions across 7 lifecycle stages with status history audit trail. |
| **Customer E-Commerce** | `templates/customer_*.html` | **Complete** | Customer catalog, detailed product view with privacy isolation (proprietary margins/costs hidden), persistent shopping cart, checkout, order receipt, and live order tracking. |
| **Email Service** | `services/email_service.py` | **Complete & Safe** | Order confirmation, status updates, and delivery alerts using SMTP with TLS. Includes fallback mock simulation logging when SMTP is unconfigured. |
| **Supervised Baseline ML** | `models/demand_model.py` | **Functional** | Random Forest Regressor predicting demand $Y$ from price $P$ and covariates $W$. Provides baseline out-of-sample MAE, RMSE, and $R^2$. |
| **Double Machine Learning** | `models/dml_model.py`, `src/econocausal/double_ml.py` | **Functional** | Robinson residual-on-residual regression with cross-fitting. Nuisance models predict outcome $\hat{Y} = \mathbb{E}[Y \mid W]$ and treatment $\hat{T} = \mathbb{E}[T \mid W]$; orthogonalized residuals $\tilde{Y}$ and $\tilde{T}$ estimate unbiased causal price elasticity $\theta$. |
| **Dynamic Pricing Engine** | `pricing/pricing_engine.py`, `src/econocausal/pricing_optimizer.py` | **Complete** | Generates discrete candidate prices within search radius, ensures margin floor ($P \ge c \times (1 + \text{margin})$), optimizes revenue $\max(P \cdot Q)$ and gross profit $\max((P - c) \cdot Q)$, outputs expected demand and uplift percentages. |
| **REST APIs** | `app.py` | **Functional** | JSON endpoints for products, summary KPIs, demand prediction, price recommendation, causal effect (ATE & 95% CI), and analytics charts. |
| **Web Dashboard & UI** | `templates/` (21 HTML templates) | **Rich & Operational** | Executive KPI cards, Chart.js visualizations (price vs demand scatter, category breakdown, monthly seasonality, promotional lift), pricing simulator with price approve & push to catalog. |
| **Interactive Research Dashboard** | `dashboard/app.py` | **Functional** | Streamlit application displaying endogeneity demonstrations, DML architecture, CATE distributions, and policy comparison. |

---

## 3. Discovered Issues, Incomplete Features, and Root Causes

### 1. `requirements.txt` UTF-16LE Encoding Conflict
- **Symptom:** Viewing or parsing `requirements.txt` failed with `unsupported mime type text/plain; charset=utf-16le`.
- **Root Cause:** Generated on Windows PowerShell using redirection (`pip freeze > requirements.txt`), which defaults to UTF-16LE with BOM.
- **Resolution:** Re-encoded file to UTF-8 without BOM. Compatible across all platforms and standard pip tooling.

### 2. Standalone Test Endpoint Connection Refusal (`test_endpoints.py`)
- **Symptom:** Running `python test_endpoints.py` failed with `[WinError 10061] No connection could be made because the target machine actively refused it`.
- **Root Cause:** The test script used `urllib.request.urlopen("http://127.0.0.1:5000/...")` assuming an external web server was already manually running in another terminal window.
- **Resolution:** Refactor `test_endpoints.py` to use Flask's in-memory WSGI `app.test_client()`, while providing optional live-server testing when a server is detected.

### 3. Model Training & Pipeline Disconnect
- **Symptom:** `train_models.py` only trains `models/demand_model.py` and `models/dml_model.py` on `data/sales_data.csv`, but does not trigger or link to `src/econocausal/pipeline.py` which produces `outputs/models/dml_pricing_model.joblib` and `outputs/reports/evaluation_results.json` needed by the Streamlit dashboard (`dashboard/app.py`).
- **Root Cause:** Two parallel implementations were developed independently without a unified runner.
- **Resolution:** Provide a unified training orchestrator that trains both the Flask application models (`models/*.joblib`) and executes the scientific benchmark evaluation pipeline (`outputs/*`), ensuring both the Flask app and the Streamlit dashboard run seamlessly.

### 4. DML Estimator Heterogeneity & Scientific Validation
- **Symptom:** `models/dml_model.py` performs constant ATE estimation via Robinson's partial linear model with Ridge regression, but does not output full diagnostic metrics (e.g. first-stage nuisance model $R^2$, residual balance diagnostics, or Heterogeneous CATE).
- **Root Cause:** The model was optimized for speed and point-elasticity retrieval for the interactive web simulator rather than deep scientific reporting.
- **Resolution:** Enhance `models/dml_model.py` and integrate diagnostics from `src/econocausal/double_ml.py` so both average treatment effect (ATE) and conditional average treatment effects (CATE) are recorded with first-stage nuisance scores and standard errors.

### 5. Windows Process & ThreadPool Overhead
- **Symptom:** Python subprocess invocations take significant initialization time on Windows when importing heavy scientific stacks (`scikit-learn`, `scipy`, `numba`, `shap`, `econml`).
- **Root Cause:** On Windows, OpenMP runtime initialization, DLL loading across scientific libraries, and unbuffered stdout redirection can cause high latency or buffer blocking.
- **Resolution:** Use unbuffered execution (`python -u`), configure `PYTHONUNBUFFERED=1`, and ensure parallel backends in joblib avoid thread contention.

---

## 4. Key File Responsibilities

| File Path | Role and Responsibilities |
| :--- | :--- |
| `app.py` | Central Flask controller: routes, RBAC decorators, view handlers, REST APIs, model inference dispatching. |
| `train_models.py` | Training script for application models (`demand_rf.joblib`, `dml_model.joblib`) and SQLite database seeding. |
| `database/database.py` | SQLite connection pooling, table schemas, migration hooks, authentication helpers, order lifecycle transitions, and catalog queries. |
| `models/preprocessing.py` | Feature extraction, date decomposition, categorical one-hot encoding, feature matrix assembly. |
| `models/demand_model.py` | Baseline predictive demand regressor (Random Forest). Evaluates out-of-sample MAE, RMSE, and $R^2$. |
| `models/dml_model.py` | Double Machine Learning causal pricing model using cross-fitting and Robinson orthogonalization. Computes ATE and 95% confidence intervals. |
| `pricing/pricing_engine.py` | Dynamic pricing optimization engine. Evaluates candidate price grid, respects margin floors, computes profit- and revenue-maximizing recommendations. |
| `services/email_service.py` | Transactional email delivery service with HTML/text templates and safe fallback to console simulation. |
| `src/econocausal/pipeline.py` | Scientific research pipeline: data generation, EDA generation, Naive OLS, Naive ML, Double ML, counterfactual evaluation. |
| `dashboard/app.py` | Streamlit interactive research and evaluation dashboard. |
| `test_auth_and_ecommerce.py` | End-to-end integration test suite for authentication, role security, product catalog, cart, checkout, and admin order lifecycle. |
| `test_email_and_tracking.py` | Unit and integration test suite for order tracking, status history, customer isolation, and email notifications. |
| `test_endpoints.py` | Flask route and REST API endpoint validation suite. |

---

## 5. Recommended Implementation & Remediation Plan

1. **Step 1: Fix and Standardize Test Suites**
   - Update `test_endpoints.py` to use `app.test_client()` so automated testing runs reliably without needing a pre-launched HTTP server.
   - Run and verify all three test suites: `test_auth_and_ecommerce.py`, `test_email_and_tracking.py`, and `test_endpoints.py`.
2. **Step 2: Unify Model Training & Scientific Pipeline**
   - Verify `train_models.py` executes cleanly and generates all required model artifacts (`models/demand_rf.joblib`, `models/dml_model.joblib`).
   - Ensure the scientific pipeline (`src/econocausal/pipeline.py`) can generate `outputs/models/dml_pricing_model.joblib`, `outputs/reports/evaluation_results.json`, and EDA charts.
3. **Step 3: Verify Flask Web Application & REST APIs**
   - Validate application startup, route registration, database queries, and template rendering.
   - Test pricing simulator and dynamic price synchronization from admin portal to customer storefront.
4. **Step 4: Verify Streamlit Dashboard**
   - Verify that `dashboard/app.py` loads artifacts from `outputs/` without errors.
5. **Step 5: Complete Comprehensive Documentation**
   - Produce all deliverables required for internship submission: `IMPLEMENTATION_GUIDE.md`, `METHODOLOGY.md`, `DATA_DICTIONARY.md`, `MODEL_EVALUATION.md`, `CAUSAL_ASSUMPTIONS_AND_LIMITATIONS.md`, `TEST_REPORT.md`, `TROUBLESHOOTING.md`, `FINAL_PROJECT_REPORT.md`, and `PRESENTATION_OUTLINE.md`.

---

## 6. Risks, Safeguards, and Unresolved Dependencies

1. **Price Endogeneity Risk:** In observational data, prices correlate with demand shocks. Relying on naive regression causes positive or attenuated elasticity. *Mitigation: DML orthogonalization guarantees unbiased causal recovery.*
2. **Unobserved Confounding:** DML assumes conditional unconfoundedness (all major confounders observed). If unobserved promotions or viral events exist, causal estimates can have residual bias. *Mitigation: Clearly document positivity, overlap, and unconfoundedness assumptions in `CAUSAL_ASSUMPTIONS_AND_LIMITATIONS.md`.*
3. **SMTP Dependency:** Real email delivery requires valid credentials in `.env`. *Mitigation: Safe fallback logs simulated emails to console without interrupting customer checkout.*
