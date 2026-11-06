# EconoCausal: Data Dictionary & Schema Specification

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Dataset Path:** `data/sales_data.csv` (12,000 observational transactions)  
**Database Path:** `database/econocausal.db` (SQLite relational store)  

---

## 1. Observational Transactions Dataset (`data/sales_data.csv`)

The central dataset consists of 12,000 records structured across 6 products over a 2-year transactional window. It captures customer purchasing behavior, macroeconomic context, seasonal shocks, and managerial price interventions with explicit price endogeneity.

### 1.1 Column Specifications

| Column Name | Data Type | Role in DML | Value Range / Categories | Description |
| :--- | :---: | :---: | :---: | :--- |
| `date` | String (Date) | Temporal Index | `2025-01-01` to `2026-12-31` | Date of recorded transaction. |
| `product_id` | String | Stratifier | `PROD_001` - `PROD_006` | Unique SKU product identifier. |
| `product_name` | String | Metadata | e.g. "Pro Wireless Headphones" | Human-readable product title. |
| `category` | String | Confounder ($W$) | `Electronics`, `Home & Office`, `Wearables` | Retail product department. |
| `price` | Float | **Treatment ($T$)** | ₹1,000.00 to ₹35,000.00 | Realized price offered to customer. |
| `cost` | Float | Constraint ($c$) | ₹950.00 to ₹16,000.00 | Marginal unit cost of goods sold (COGS). |
| `demand` | Integer | **Outcome ($Y$)** | 1 to 150 units | Units purchased by customer in transaction. |
| `competitor_price`| Float | Confounder ($W$) | ₹1,050.00 to ₹36,500.00 | Competitor price observed in the market. |
| `promotion` | Integer (Binary)| Confounder ($W$) | `0` (No), `1` (Active Promo) | Marketing discount banner or voucher. |
| `holiday` | Integer (Binary)| Confounder ($W$) | `0` (Standard), `1` (Holiday/Festive) | National holiday or high-demand festive surge. |
| `inventory` | Integer | Confounder ($W$) | 15 to 480 units | Remaining stock in regional fulfillment center. |
| `temperature` | Float | Confounder ($W$) | 10.0°C to 45.0°C | Daily ambient temperature in Celsius. |
| `day_of_week` | String | Confounder ($W$) | `Monday` - `Sunday` | Day of week for weekly seasonality. |
| `month` | Integer | Confounder ($W$) | `1` to `12` | Calendar month index. |
| `season` | String | Confounder ($W$) | `Winter`, `Spring`, `Summer`, `Autumn` | Macro meteorological season. |

---

## 2. Product Catalog Baseline Specifications

The catalog models realistic price points, margins, and microeconomic price sensitivities:

| Product ID | Product Name | Department | Base Price ($P_0$) | Unit Cost ($c$) | Base Margin | Causal Elasticity ($\theta$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `PROD_001` | Pro Wireless Headphones | Electronics | ₹2,499.00 | ₹1,350.00 | 46.0% | -0.0180 |
| `PROD_002` | Ultra HD Smart TV 43-inch | Electronics | ₹24,999.00 | ₹16,000.00 | 36.0% | -0.0011 |
| `PROD_003` | Ergonomic Mesh Office Chair | Home & Office | ₹6,499.00 | ₹3,600.00 | 44.6% | -0.0045 |
| `PROD_004` | Smart Fitness Tracker Watch | Wearables | ₹3,199.00 | ₹1,700.00 | 46.9% | -0.0140 |
| `PROD_005` | Noise-Cancelling TWS Earbuds | Electronics | ₹1,899.00 | ₹950.00 | 50.0% | -0.0240 |
| `PROD_006` | Standing Electric Desk 120cm | Home & Office | ₹14,999.00 | ₹9,200.00 | 38.7% | -0.0013 |

---

## 3. SQLite Relational Database Architecture (`database/econocausal.db`)

### 3.1 Entity Relationship Summary

```
   [users] 1 ──── ∞ [orders] 1 ──── ∞ [order_items] ∞ ──── 1 [products]
      │                                    │
      └─── 1 ──── ∞ [cart]                 └── 1 ──── ∞ [order_status_history]
```

### 3.2 Table Schema Definitions

#### Table 1: `products`
```sql
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    base_price REAL NOT NULL,
    cost REAL NOT NULL,
    stock INTEGER DEFAULT 100,
    current_price REAL DEFAULT 0.0,
    description TEXT DEFAULT '',
    image_url TEXT DEFAULT '',
    rating REAL DEFAULT 4.5,
    competitor_price REAL DEFAULT 0.0,
    promotion INTEGER DEFAULT 0
);
```

#### Table 2: `sales`
```sql
CREATE TABLE sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,
    demand INTEGER NOT NULL,
    competitor_price REAL NOT NULL,
    promotion INTEGER NOT NULL,
    holiday INTEGER NOT NULL,
    inventory INTEGER NOT NULL,
    temperature REAL NOT NULL,
    day_of_week TEXT NOT NULL,
    month INTEGER NOT NULL,
    season TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
```

#### Table 3: `predictions` (Dynamic Pricing Decision Audit Trail)
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    product_id TEXT NOT NULL,
    current_price REAL NOT NULL,
    recommended_price REAL NOT NULL,
    optimal_revenue_price REAL NOT NULL,
    expected_demand REAL NOT NULL,
    expected_revenue REAL NOT NULL,
    expected_profit REAL NOT NULL,
    causal_elasticity REAL NOT NULL
);
```

#### Table 4: `users` (Role-Based Authentication)
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'customer')),
    created_at TEXT NOT NULL
);
```

#### Table 5: `orders` (Customer E-Commerce Transactions)
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    delivery_address TEXT NOT NULL,
    phone TEXT NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'Confirmed',
    payment_method TEXT DEFAULT 'Cash on Delivery',
    estimated_delivery TEXT DEFAULT '',
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

#### Table 6: `order_status_history` (Audit Log for Order Tracking)
```sql
CREATE TABLE order_status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (id)
);
```
