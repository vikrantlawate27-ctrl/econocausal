"""
EconoCausal Database Module
Manages SQLite connection, schema migrations, data ingestion from sales_data.csv,
and query helpers for products, sales history, and pricing prediction logs.
"""

import os
from pathlib import Path
import sqlite3
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DB_PATH = str(Path(__file__).resolve().parent / "econocausal.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(csv_path: str = "data/sales_data.csv"):
    """
    Initializes SQLite tables and seeds products and initial sales records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Products Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            base_price REAL NOT NULL,
            cost REAL NOT NULL,
            stock INTEGER DEFAULT 100
        )
    """)

    # 2. Historical Sales Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
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
        )
    """)

    # 3. Dynamic Pricing Predictions / Simulation Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
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
        )
    """)

    # 4. Users Table (Authentication & Role Management)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'customer')),
            created_at TEXT NOT NULL
        )
    """)

    # 5. Customer Orders Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            customer_email TEXT NOT NULL,
            delivery_address TEXT NOT NULL,
            phone TEXT NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            payment_method TEXT DEFAULT 'Cash on Delivery',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # 6. Order Items Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    """)

    # 7. Customer Cart Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            added_at TEXT NOT NULL,
            UNIQUE(user_id, product_id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    # 8. Order Status History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            updated_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    """)

    conn.commit()

    # Safely migrate orders table schema
    cursor.execute("PRAGMA table_info(orders)")
    existing_order_cols = [row["name"] for row in cursor.fetchall()]
    order_migrations = [
        ("estimated_delivery", "TEXT DEFAULT ''"),
        ("created_at", "TEXT DEFAULT ''"),
        ("updated_at", "TEXT DEFAULT ''"),
    ]
    for col_name, col_def in order_migrations:
        if col_name not in existing_order_cols:
            cursor.execute(f"ALTER TABLE orders ADD COLUMN {col_name} {col_def}")

    # Backfill estimated_delivery and timestamps for existing orders if empty
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    est_date = (datetime.now() + timedelta(days=4)).strftime("%d %B %Y")
    cursor.execute("""
        UPDATE orders 
        SET estimated_delivery = ?, created_at = COALESCE(order_date, ?), updated_at = ?
        WHERE estimated_delivery IS NULL OR estimated_delivery = ''
    """, (est_date, now_ts, now_ts))

    # Seed initial order_status_history for any existing orders missing history
    cursor.execute("SELECT id, status, order_date FROM orders")
    existing_orders = cursor.fetchall()
    for o in existing_orders:
        cursor.execute("SELECT COUNT(*) as cnt FROM order_status_history WHERE order_id = ?", (o["id"],))
        if cursor.fetchone()["cnt"] == 0:
            cursor.execute("""
                INSERT INTO order_status_history (order_id, status, message, updated_by, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (o["id"], o["status"] or "Confirmed", f"Order {o['status'] or 'Confirmed'} and recorded in system", "System", o["order_date"] or now_ts))

    # Safely migrate products table schema to add any new fields needed for e-commerce
    cursor.execute("PRAGMA table_info(products)")
    existing_cols = [row["name"] for row in cursor.fetchall()]
    
    product_migrations = [
        ("description", "TEXT DEFAULT ''"),
        ("image_url", "TEXT DEFAULT ''"),
        ("rating", "REAL DEFAULT 4.5"),
        ("current_price", "REAL DEFAULT 0.0"),
        ("competitor_price", "REAL DEFAULT 0.0"),
        ("promotion", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_def in product_migrations:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE products ADD COLUMN {col_name} {col_def}")
    
    # Ensure current_price is populated from base_price if 0 or NULL
    cursor.execute("UPDATE products SET current_price = base_price WHERE current_price IS NULL OR current_price <= 0")
    
    # Update default descriptions and image URLs for standard products
    product_images_and_desc = {
        "PROD_001": (
            "High-fidelity active noise cancelling wireless headphones with 40-hour battery life and memory foam comfort.",
            "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80",
            4.8, 2580.0, 0
        ),
        "PROD_002": (
            "Crystal-clear 4K Ultra HD Smart TV featuring Dolby Vision, HDR10+, voice remote, and dual stereo speakers.",
            "https://images.unsplash.com/photo-1593359677879-a4bb92f829d1?w=600&auto=format&fit=crop&q=80",
            4.6, 25999.0, 0
        ),
        "PROD_003": (
            "Breathable high-back ergonomic mesh office chair with lumbar adjustment, 3D armrests, and synchro-tilt mechanism.",
            "https://images.unsplash.com/photo-1580481077195-c9a910ec3f80?w=600&auto=format&fit=crop&q=80",
            4.7, 6800.0, 0
        ),
        "PROD_004": (
            "Advanced smart fitness watch with AMOLED touch display, 24/7 heart-rate and SpO2 tracking, and 5ATM water resistance.",
            "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80",
            4.5, 3350.0, 1
        ),
        "PROD_005": (
            "Compact in-ear true wireless earbuds with immersive bass, environmental noise cancellation, and IPX5 sweat resistance.",
            "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=600&auto=format&fit=crop&q=80",
            4.4, 1999.0, 0
        ),
        "PROD_006": (
            "Dual-motor motorized standing desk with programmable memory presets, anti-collision sensor, and heavy steel frame.",
            "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?w=600&auto=format&fit=crop&q=80",
            4.9, 15800.0, 0
        ),
    }
    for pid, (desc, img, rating, comp_p, promo) in product_images_and_desc.items():
        cursor.execute("""
            UPDATE products 
            SET description = ?, image_url = ?, rating = ?, competitor_price = ?, promotion = ?
            WHERE product_id = ? AND (description = '' OR description IS NULL)
        """, (desc, img, rating, comp_p, promo, pid))

    conn.commit()

    # Seed products if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM products")
    if cursor.fetchone()["cnt"] == 0:
        products_seed = [
            ("PROD_001", "Pro Wireless Headphones", "Electronics", 2499.0, 1350.0, 180, 2499.0, "High-fidelity active noise cancelling wireless headphones.", "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80", 4.8, 2580.0, 0),
            ("PROD_002", "Ultra HD Smart TV 43-inch", "Electronics", 24999.0, 16000.0, 45, 24999.0, "Crystal-clear 4K Ultra HD Smart TV featuring Dolby Vision.", "https://images.unsplash.com/photo-1593359677879-a4bb92f829d1?w=600&auto=format&fit=crop&q=80", 4.6, 25999.0, 0),
            ("PROD_003", "Ergonomic Mesh Office Chair", "Home & Office", 6499.0, 3600.0, 95, 6499.0, "Breathable high-back ergonomic mesh office chair.", "https://images.unsplash.com/photo-1580481077195-c9a910ec3f80?w=600&auto=format&fit=crop&q=80", 4.7, 6800.0, 0),
            ("PROD_004", "Smart Fitness Tracker Watch", "Wearables", 3199.0, 1700.0, 220, 3199.0, "Advanced smart fitness watch with AMOLED touch display.", "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80", 4.5, 3350.0, 1),
            ("PROD_005", "Noise-Cancelling True Wireless Earbuds", "Electronics", 1899.0, 950.0, 310, 1899.0, "Compact in-ear true wireless earbuds with immersive bass.", "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=600&auto=format&fit=crop&q=80", 4.4, 1999.0, 0),
            ("PROD_006", "Standing Electric Desk 120cm", "Home & Office", 14999.0, 9200.0, 40, 14999.0, "Dual-motor motorized standing desk with memory presets.", "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?w=600&auto=format&fit=crop&q=80", 4.9, 15800.0, 0),
        ]
        cursor.executemany("""
            INSERT INTO products (product_id, product_name, category, base_price, cost, stock, current_price, description, image_url, rating, competitor_price, promotion) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, products_seed)
        conn.commit()
        print("Seeded products table.")

    # Seed Default Admin Account: admin@econocausal.com / Admin@123
    cursor.execute("SELECT id FROM users WHERE email = ?", ("admin@econocausal.com",))
    if cursor.fetchone() is None:
        from werkzeug.security import generate_password_hash
        admin_pass_hash = generate_password_hash("Admin@123")
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("System Administrator", "admin@econocausal.com", admin_pass_hash, "admin", now_ts))
        conn.commit()
        print("Seeded default admin account: admin@econocausal.com / Admin@123")

    # Seed a demo customer account if none exists
    cursor.execute("SELECT id FROM users WHERE role = 'customer'")
    if cursor.fetchone() is None:
        from werkzeug.security import generate_password_hash
        cust_pass_hash = generate_password_hash("Customer@123")
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("Vikrant Lawate", "customer@econocausal.com", cust_pass_hash, "customer", now_ts))
        conn.commit()
        print("Seeded demo customer account: customer@econocausal.com / Customer@123")

    # Seed sales if empty and csv exists
    cursor.execute("SELECT COUNT(*) as cnt FROM sales")
    if cursor.fetchone()["cnt"] == 0 and os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sales_rows = [
                (
                    row["date"], row["product_id"], float(row["price"]), float(row["cost"]),
                    int(row["demand"]), float(row["competitor_price"]), int(row["promotion"]),
                    int(row["holiday"]), int(row["inventory"]), float(row["temperature"]),
                    row["day_of_week"], int(row["month"]), row["season"]
                )
                for row in reader
            ]
            cursor.executemany("""
                INSERT INTO sales (
                    date, product_id, price, cost, demand, competitor_price,
                    promotion, holiday, inventory, temperature, day_of_week, month, season
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sales_rows)
            conn.commit()
            print(f"Seeded sales table with {len(sales_rows):,} records.")

    conn.close()


# -----------------------------------------------------------------------------
# User & Authentication Queries
# -----------------------------------------------------------------------------
def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches user record by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user record by primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(name: str, email: str, password_hash: str, role: str = "customer") -> int:
    """Creates a new registered user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO users (name, email, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (name.strip(), email.strip().lower(), password_hash, role, now_ts))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def update_user_profile(user_id: int, name: str, password_hash: Optional[str] = None):
    """Updates user profile information."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if password_hash:
        cursor.execute("UPDATE users SET name = ?, password_hash = ? WHERE id = ?", (name.strip(), password_hash, user_id))
    else:
        cursor.execute("UPDATE users SET name = ? WHERE id = ?", (name.strip(), user_id))
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Products Management Queries
# -----------------------------------------------------------------------------
def get_all_products(category: Optional[str] = None, search: Optional[str] = None, sort_by: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns list of products with optional filtering and sorting."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM products WHERE 1=1"
    params = []
    
    if category and category.lower() != "all":
        query += " AND LOWER(category) = LOWER(?)"
        params.append(category)
        
    if search:
        query += " AND (LOWER(product_name) LIKE ? OR LOWER(description) LIKE ?)"
        term = f"%{search.strip().lower()}%"
        params.extend([term, term])
        
    if sort_by == "price_asc":
        query += " ORDER BY current_price ASC"
    elif sort_by == "price_desc":
        query += " ORDER BY current_price DESC"
    elif sort_by == "rating_desc":
        query += " ORDER BY rating DESC"
    else:
        query += " ORDER BY product_name ASC"
        
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a specific product by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def add_product(
    product_id: str,
    product_name: str,
    category: str,
    cost: float,
    current_price: float,
    stock: int = 100,
    description: str = "",
    image_url: str = "",
    competitor_price: float = 0.0,
    promotion: int = 0,
) -> bool:
    """Inserts a new product into catalog."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO products (
            product_id, product_name, category, cost, base_price, current_price,
            stock, description, image_url, competitor_price, promotion, rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 4.5)
    """, (
        product_id.strip(), product_name.strip(), category.strip(), cost,
        current_price, current_price, stock, description.strip(),
        image_url.strip(), competitor_price, promotion
    ))
    conn.commit()
    conn.close()
    return True


def update_product(
    product_id: str,
    product_name: str,
    category: str,
    cost: float,
    current_price: float,
    stock: int,
    description: str = "",
    image_url: str = "",
    competitor_price: float = 0.0,
    promotion: int = 0,
) -> bool:
    """Updates product attributes and selling price."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE products SET
            product_name = ?,
            category = ?,
            cost = ?,
            current_price = ?,
            base_price = ?,
            stock = ?,
            description = ?,
            image_url = ?,
            competitor_price = ?,
            promotion = ?
        WHERE product_id = ?
    """, (
        product_name.strip(), category.strip(), cost, current_price, current_price,
        stock, description.strip(), image_url.strip(), competitor_price, promotion, product_id
    ))
    conn.commit()
    conn.close()
    return True


def update_product_price_only(product_id: str, new_price: float) -> bool:
    """Updates only selling price (e.g. from DML recommendation approval)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE products 
        SET current_price = ?, base_price = ?
        WHERE product_id = ?
    """, (round(new_price, 2), round(new_price, 2), product_id))
    conn.commit()
    conn.close()
    return True


def delete_product(product_id: str) -> bool:
    """Deletes a product from the database and cleans up cart items."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cart WHERE product_id = ?", (product_id,))
    cursor.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
    conn.commit()
    conn.close()
    return True


# -----------------------------------------------------------------------------
# Cart Queries
# -----------------------------------------------------------------------------
def get_user_cart(user_id: int) -> List[Dict[str, Any]]:
    """Returns cart items with product details and subtotal for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id as cart_id, c.quantity, p.product_id, p.product_name, p.category, 
               p.current_price, p.stock, p.image_url, (c.quantity * p.current_price) as subtotal
        FROM cart c
        JOIN products p ON c.product_id = p.product_id
        WHERE c.user_id = ?
        ORDER BY c.id DESC
    """, (user_id,))
    items = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return items


def add_to_cart(user_id: int, product_id: str, quantity: int = 1) -> bool:
    """Adds a product to cart or increments existing quantity."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO cart (user_id, product_id, quantity, added_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, product_id) DO UPDATE SET
            quantity = quantity + excluded.quantity
    """, (user_id, product_id, max(1, quantity), now_ts))
    conn.commit()
    conn.close()
    return True


def update_cart_item_quantity(user_id: int, product_id: str, quantity: int) -> bool:
    """Updates quantity or removes item if quantity <= 0."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if quantity <= 0:
        cursor.execute("DELETE FROM cart WHERE user_id = ? AND product_id = ?", (user_id, product_id))
    else:
        cursor.execute("UPDATE cart SET quantity = ? WHERE user_id = ? AND product_id = ?", (quantity, user_id, product_id))
    conn.commit()
    conn.close()
    return True


def remove_from_cart(user_id: int, product_id: str) -> bool:
    """Removes a product from user's cart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cart WHERE user_id = ? AND product_id = ?", (user_id, product_id))
    conn.commit()
    conn.close()
    return True


def clear_user_cart(user_id: int) -> bool:
    """Empties user's cart (e.g. after order placement)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cart WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True


def get_cart_count(user_id: int) -> int:
    """Returns total number of items in user's cart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(quantity), 0) as cnt FROM cart WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()["cnt"]
    conn.close()
    return int(count)


# -----------------------------------------------------------------------------
# Orders Queries
# -----------------------------------------------------------------------------
def create_order(
    user_id: int,
    customer_name: str,
    customer_email: str,
    delivery_address: str,
    phone: str,
    items: List[Dict[str, Any]],
    payment_method: str = "Cash on Delivery",
) -> int:
    """Creates a new customer order, sets default status to Confirmed, and records initial history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    total_amount = sum(float(item["current_price"]) * int(item["quantity"]) for item in items)
    now = datetime.now()
    now_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    est_delivery = (now + timedelta(days=4)).strftime("%d %B %Y")
    
    cursor.execute("""
        INSERT INTO orders (
            user_id, order_date, customer_name, customer_email,
            delivery_address, phone, total_amount, status, payment_method,
            estimated_delivery, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Confirmed', ?, ?, ?, ?)
    """, (
        user_id, now_ts, customer_name.strip(), customer_email.strip(),
        delivery_address.strip(), phone.strip(), round(total_amount, 2), payment_method,
        est_delivery, now_ts, now_ts
    ))
    order_id = cursor.lastrowid
    
    # Insert order items
    order_items_rows = [
        (
            order_id,
            item["product_id"],
            item["product_name"],
            float(item["current_price"]),
            int(item["quantity"]),
            round(float(item["current_price"]) * int(item["quantity"]), 2)
        )
        for item in items
    ]
    cursor.executemany("""
        INSERT INTO order_items (order_id, product_id, product_name, price, quantity, subtotal)
        VALUES (?, ?, ?, ?, ?, ?)
    """, order_items_rows)
    
    # Decrement inventory stock
    for item in items:
        cursor.execute("UPDATE products SET stock = MAX(0, stock - ?) WHERE product_id = ?", (int(item["quantity"]), item["product_id"]))
        
    # Log initial status history
    cursor.execute("""
        INSERT INTO order_status_history (order_id, status, message, updated_by, created_at)
        VALUES (?, 'Confirmed', 'Your order has been confirmed.', 'Customer', ?)
    """, (order_id, now_ts))

    conn.commit()
    conn.close()
    
    # Clear user cart
    clear_user_cart(user_id)
    return order_id


def get_user_orders(user_id: int) -> List[Dict[str, Any]]:
    """Fetches orders for a specific customer with item details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM orders WHERE user_id = ? ORDER BY id DESC
    """, (user_id,))
    orders = [dict(row) for row in cursor.fetchall()]
    
    for order in orders:
        cursor.execute("SELECT * FROM order_items WHERE order_id = ?", (order["id"],))
        order["items"] = [dict(item) for item in cursor.fetchall()]
        order["item_count"] = sum(i["quantity"] for i in order["items"])
        
    conn.close()
    return orders


def get_all_orders() -> List[Dict[str, Any]]:
    """Fetches all orders across customers for admin management."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders ORDER BY id DESC")
    orders = [dict(row) for row in cursor.fetchall()]
    
    for order in orders:
        cursor.execute("SELECT * FROM order_items WHERE order_id = ?", (order["id"],))
        order["items"] = [dict(item) for item in cursor.fetchall()]
        order["item_count"] = sum(i["quantity"] for i in order["items"])
        
    conn.close()
    return orders


def update_order_status(
    order_id: int,
    new_status: str,
    updated_by: str = "Admin",
    custom_message: Optional[str] = None,
) -> bool:
    """Updates order fulfillment status and logs chronological entry into order_status_history."""
    valid_statuses = [
        "Pending", "Confirmed", "Processing", "Shipped", "Out for Delivery", "Delivered", "Cancelled"
    ]
    if new_status not in valid_statuses:
        return False

    default_messages = {
        "Pending": "Your order is pending verification.",
        "Confirmed": "Your order has been confirmed.",
        "Processing": "Your order is now being processed.",
        "Shipped": "Your order has been shipped.",
        "Out for Delivery": "Your order is out for delivery.",
        "Delivered": "Your order has been delivered successfully.",
        "Cancelled": "Your order has been cancelled.",
    }
    message = custom_message or default_messages.get(new_status, f"Order status changed to {new_status}")
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE orders 
        SET status = ?, updated_at = ? 
        WHERE id = ?
    """, (new_status, now_ts, order_id))

    cursor.execute("""
        INSERT INTO order_status_history (order_id, status, message, updated_by, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (order_id, new_status, message, updated_by, now_ts))

    conn.commit()
    conn.close()
    return True


def get_order_status_history(order_id: int) -> List[Dict[str, Any]]:
    """Returns chronological order status tracking events."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM order_status_history 
        WHERE order_id = ? 
        ORDER BY id ASC
    """, (order_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_order_statistics() -> Dict[str, int]:
    """Returns order counts partitioned by fulfillment status for executive dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) as cnt FROM orders GROUP BY status")
    counts = {row["status"]: row["cnt"] for row in cursor.fetchall()}
    cursor.execute("SELECT COUNT(*) as total FROM orders")
    total = cursor.fetchone()["total"]
    conn.close()
    
    return {
        "total": total,
        "pending": counts.get("Pending", 0),
        "confirmed": counts.get("Confirmed", 0),
        "processing": counts.get("Processing", 0),
        "shipped": counts.get("Shipped", 0),
        "out_for_delivery": counts.get("Out for Delivery", 0),
        "delivered": counts.get("Delivered", 0),
        "cancelled": counts.get("Cancelled", 0),
    }


def get_customer_notifications(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Returns recent order status notification alerts for a customer."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT h.id, h.order_id, h.status, h.message, h.created_at, o.total_amount
        FROM order_status_history h
        JOIN orders o ON h.order_id = o.id
        WHERE o.user_id = ?
        ORDER BY h.id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_order_by_id(order_id: int) -> Optional[Dict[str, Any]]:
    """Fetches single order details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    order = dict(row)
    cursor.execute("SELECT * FROM order_items WHERE order_id = ?", (order_id,))
    order["items"] = [dict(item) for item in cursor.fetchall()]
    conn.close()
    return order


# -----------------------------------------------------------------------------
# Historical Sales Data Query (Admin)
# -----------------------------------------------------------------------------
def get_recent_sales(limit: int = 150, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns recent historical sales transactions for Admin Sales table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if product_id:
        cursor.execute("""
            SELECT s.*, p.product_name, (s.price * s.demand) as revenue
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            WHERE s.product_id = ?
            ORDER BY s.date DESC
            LIMIT ?
        """, (product_id, limit))
    else:
        cursor.execute("""
            SELECT s.*, p.product_name, (s.price * s.demand) as revenue
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            ORDER BY s.date DESC
            LIMIT ?
        """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# -----------------------------------------------------------------------------
# Legacy Prediction & Analytics Helper Queries
# -----------------------------------------------------------------------------
def log_prediction(
    product_id: str,
    current_price: float,
    recommended_price: float,
    optimal_revenue_price: float,
    expected_demand: float,
    expected_revenue: float,
    expected_profit: float,
    causal_elasticity: float,
) -> int:
    """Inserts a dynamic pricing prediction into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, product_id, current_price, recommended_price,
            optimal_revenue_price, expected_demand, expected_revenue,
            expected_profit, causal_elasticity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, product_id, current_price, recommended_price,
        optimal_revenue_price, expected_demand, expected_revenue,
        expected_profit, causal_elasticity
    ))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def get_recent_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches recent dynamic pricing recommendations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, pr.product_name, pr.category
        FROM predictions p
        LEFT JOIN products pr ON p.product_id = pr.product_id
        ORDER BY p.id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_sales_summary() -> Dict[str, Any]:
    """Computes overall KPIs from sales table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total_sales_records,
            COUNT(DISTINCT product_id) as total_products,
            AVG(price) as avg_price,
            AVG(demand) as avg_demand,
            SUM(price * demand) as total_revenue,
            SUM((price - cost) * demand) as total_profit
        FROM sales
    """)
    summary = dict(cursor.fetchone())
    conn.close()
    return summary


if __name__ == "__main__":
    init_db()
    print("Database initialization check complete.")

