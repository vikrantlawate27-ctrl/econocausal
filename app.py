"""
EconoCausal: Dynamic Pricing via Double Machine Learning
Main Flask Application Server

Integrated Architecture:
  1. Public Landing & Discovery (/ , /products, /product/<id>)
  2. Role-Based Authentication (/login, /admin/login, /register, /logout)
  3. Admin Portal (/admin/dashboard, /admin/products, /admin/sales, /admin/orders, /admin/profile)
  4. AI & Causal Engines (/dashboard, /analysis, /simulator, /api/*)
  5. Customer E-Commerce Experience (/customer/dashboard, /cart, /checkout, /customer/orders, /customer/profile)
"""

import os
import sys
import json
import sqlite3
import re
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
from datetime import datetime

import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
    abort,
)
from werkzeug.security import generate_password_hash, check_password_hash

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.database import (
    init_db,
    get_db_connection,
    get_all_products,
    get_product_by_id,
    add_product,
    update_product,
    update_product_price_only,
    delete_product,
    get_user_by_email,
    get_user_by_id,
    create_user,
    update_user_profile,
    get_user_cart,
    add_to_cart,
    update_cart_item_quantity,
    remove_from_cart,
    clear_user_cart,
    get_cart_count,
    create_order,
    get_user_orders,
    get_all_orders,
    update_order_status,
    get_order_by_id,
    get_recent_sales,
    log_prediction,
    get_recent_predictions,
    get_sales_summary,
    get_order_status_history,
    get_order_statistics,
    get_customer_notifications,
)
from services.email_service import (
    send_order_confirmation_email,
    send_order_status_email,
    send_delivery_email,
)
from models.demand_model import BaselineDemandModel
from models.dml_model import DoubleMLPricingModel
from pricing.pricing_engine import DynamicPricingEngine

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = os.environ.get("SECRET_KEY", "econocausal-secure-session-key-2026-v2")

# Global singletons
demand_model: Optional[BaselineDemandModel] = None
dml_model: Optional[DoubleMLPricingModel] = None
pricing_engine = DynamicPricingEngine()


def get_models():
    """Lazily loads or initialises models."""
    global demand_model, dml_model
    if demand_model is None:
        try:
            dm = BaselineDemandModel()
            dm.load()
            demand_model = dm
        except Exception as e:
            print(f"Notice: Baseline model loading message ({e}).")

    if dml_model is None:
        try:
            dml = DoubleMLPricingModel()
            dml.load()
            dml_model = dml
        except Exception as e:
            print(f"Notice: DML model loading message ({e}).")

    return demand_model, dml_model


# -----------------------------------------------------------------------------
# Security & Role Decorators
# -----------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Admin login required.", "warning")
            return redirect(url_for("admin_login", next=request.path))
        if session.get("role") != "admin":
            flash("Access denied: Administrative privileges required.", "danger")
            return redirect(url_for("customer_dashboard"))
        return f(*args, **kwargs)
    return decorated_function


def customer_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to proceed with your shopping.", "info")
            return redirect(url_for("login", next=request.path))
        if session.get("role") == "admin":
            # If an admin wants to access customer store, allow or redirect
            return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_global_context():
    """Injects user authentication state and dynamic cart count into all templates."""
    user_id = session.get("user_id")
    cart_count = 0
    if user_id:
        try:
            cart_count = get_cart_count(user_id)
        except Exception:
            cart_count = 0
    return {
        "current_user": {
            "id": user_id,
            "name": session.get("name"),
            "email": session.get("email"),
            "role": session.get("role"),
        } if user_id else None,
        "cart_count": cart_count,
    }


# -----------------------------------------------------------------------------
# Public & Landing Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Landing and project overview page with Admin & Customer access pathways."""
    products = get_all_products()[:4]
    return render_template("index.html", featured_products=products)


# -----------------------------------------------------------------------------
# Authentication Routes
# -----------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin Login Portal."""
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if user and user.get("role") == "admin" and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            session["role"] = "admin"
            flash(f"Welcome back, {user['name']}! Admin session initialized.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("admin_dashboard"))
        else:
            error = "Invalid admin email or password."

    return render_template("admin_login.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Customer & General User Login."""
    if session.get("role") == "customer":
        return redirect(url_for("customer_dashboard"))
    elif session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                flash("Admin authenticated successfully.", "success")
                return redirect(url_for("admin_dashboard"))
            else:
                flash(f"Welcome back, {user['name']}!", "success")
                next_url = request.args.get("next")
                return redirect(next_url or url_for("customer_dashboard"))
        else:
            error = "Invalid email or password."

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Customer Account Registration."""
    if "user_id" in session:
        return redirect(url_for("customer_dashboard"))

    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not name:
            error = "Full Name is required."
        elif not email or not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            error = "A valid email address is required."
        elif not password or len(password) < 6:
            error = "Password must be at least 6 characters long."
        elif password != confirm_password:
            error = "Password confirmation does not match."
        elif get_user_by_email(email):
            error = "An account with this email address already exists."
        else:
            # Create Customer User (Admin cannot register publicly)
            password_hash = generate_password_hash(password)
            create_user(name=name, email=email, password_hash=password_hash, role="customer")
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    """Clears session and redirects customer to login."""
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/admin/logout")
def admin_logout():
    """Clears session and redirects admin to admin login."""
    session.clear()
    flash("Admin session terminated.", "info")
    return redirect(url_for("admin_login"))


# -----------------------------------------------------------------------------
# Admin Pages & Management Routes
# -----------------------------------------------------------------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Alias for existing executive dashboard, protected with admin auth."""
    order_stats = get_order_statistics()
    recent_orders = get_all_orders()[:5]
    return render_template("dashboard.html", order_stats=order_stats, recent_orders=recent_orders)


@app.route("/dashboard")
@admin_required
def dashboard():
    """Executive AI Pricing Dashboard."""
    order_stats = get_order_statistics()
    recent_orders = get_all_orders()[:5]
    return render_template("dashboard.html", order_stats=order_stats, recent_orders=recent_orders)


@app.route("/analysis")
@admin_required
def analysis():
    """Exploratory and Causal Data Analysis page."""
    return render_template("analysis.html")


@app.route("/simulator")
@admin_required
def simulator():
    """Interactive Pricing Simulator page."""
    return render_template("simulator.html")


@app.route("/admin/products", methods=["GET", "POST"])
@admin_required
def admin_products():
    """Admin Product Inventory & Pricing Management."""
    if request.method == "POST":
        # Handle Quick Add Product
        try:
            pid = request.form.get("product_id", "").strip()
            pname = request.form.get("product_name", "").strip()
            category = request.form.get("category", "Electronics").strip()
            cost = float(request.form.get("cost", 0.0))
            current_price = float(request.form.get("current_price", 0.0))
            stock = int(request.form.get("stock", 50))
            desc = request.form.get("description", "").strip()
            img = request.form.get("image_url", "").strip()
            comp_p = float(request.form.get("competitor_price", current_price * 1.05))
            promo = int(request.form.get("promotion", 0))

            if not pid or not pname or cost <= 0 or current_price <= 0:
                flash("Product ID, Name, Cost, and Current Price must be valid.", "danger")
            elif get_product_by_id(pid):
                flash(f"Product with ID '{pid}' already exists.", "danger")
            else:
                add_product(
                    product_id=pid,
                    product_name=pname,
                    category=category,
                    cost=cost,
                    current_price=current_price,
                    stock=stock,
                    description=desc,
                    image_url=img or "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80",
                    competitor_price=comp_p,
                    promotion=promo,
                )
                flash(f"Product '{pname}' added successfully.", "success")
        except Exception as e:
            flash(f"Failed to add product: {str(e)}", "danger")

        return redirect(url_for("admin_products"))

    products = get_all_products()
    return render_template("admin_products.html", products=products)


@app.route("/admin/products/edit/<product_id>", methods=["POST"])
@admin_required
def admin_edit_product(product_id: str):
    """Admin Updates Product Attributes & Pricing."""
    try:
        pname = request.form.get("product_name", "").strip()
        category = request.form.get("category", "Electronics").strip()
        cost = float(request.form.get("cost", 0.0))
        current_price = float(request.form.get("current_price", 0.0))
        stock = int(request.form.get("stock", 50))
        desc = request.form.get("description", "").strip()
        img = request.form.get("image_url", "").strip()
        comp_p = float(request.form.get("competitor_price", current_price * 1.05))
        promo = int(request.form.get("promotion", 0))

        update_product(
            product_id=product_id,
            product_name=pname,
            category=category,
            cost=cost,
            current_price=current_price,
            stock=stock,
            description=desc,
            image_url=img,
            competitor_price=comp_p,
            promotion=promo,
        )
        flash(f"Product '{product_id}' updated successfully. New Price: ₹{current_price:,.2f}", "success")
    except Exception as e:
        flash(f"Failed to update product: {str(e)}", "danger")

    return redirect(url_for("admin_products"))


@app.route("/admin/products/update-price", methods=["POST"])
@admin_required
def admin_update_price():
    """Direct API/Form price update (e.g., from Simulator approval or quick price change)."""
    try:
        data = request.get_json() if request.is_json else request.form
        product_id = data.get("product_id")
        new_price = float(data.get("new_price", 0.0))

        if not product_id or new_price <= 0:
            if request.is_json:
                return jsonify({"status": "error", "message": "Invalid product ID or price"}), 400
            flash("Invalid product ID or price.", "danger")
            return redirect(url_for("admin_products"))

        update_product_price_only(product_id, new_price)

        if request.is_json:
            return jsonify({
                "status": "success",
                "message": f"Price for {product_id} updated to ₹{new_price:,.2f}. Customer catalog refreshed.",
                "product_id": product_id,
                "new_price": new_price,
            })

        flash(f"Price for {product_id} updated to ₹{new_price:,.2f}.", "success")
        return redirect(url_for("admin_products"))
    except Exception as e:
        if request.is_json:
            return jsonify({"status": "error", "message": str(e)}), 500
        flash(f"Error updating price: {str(e)}", "danger")
        return redirect(url_for("admin_products"))


@app.route("/admin/products/delete/<product_id>", methods=["POST"])
@admin_required
def admin_delete_product(product_id: str):
    """Deletes a product from catalog."""
    try:
        delete_product(product_id)
        flash(f"Product '{product_id}' deleted successfully.", "info")
    except Exception as e:
        flash(f"Failed to delete product: {str(e)}", "danger")
    return redirect(url_for("admin_products"))


@app.route("/admin/sales")
@admin_required
def admin_sales():
    """Historical Sales Transactions from Database."""
    sales = get_recent_sales(limit=150)
    summary = get_sales_summary()
    return render_template("admin_sales.html", sales=sales, summary=summary)


@app.route("/admin/orders")
@admin_required
def admin_orders():
    """Admin Order Management."""
    orders = get_all_orders()
    return render_template("admin_orders.html", orders=orders)


@app.route("/admin/orders/update-status", methods=["POST"])
@admin_required
def admin_update_order_status():
    """Updates order fulfillment status, logs chronological history, and dispatches customer notification email."""
    try:
        order_id = int(request.form.get("order_id", 0))
        new_status = request.form.get("status", "Confirmed").strip()

        old_order = get_order_by_id(order_id)
        if not old_order:
            flash("Order not found.", "danger")
            return redirect(url_for("admin_orders"))

        old_status = old_order.get("status", "Confirmed")

        if update_order_status(order_id, new_status, updated_by="Admin"):
            # Fetch refreshed order record
            updated_order = get_order_by_id(order_id)
            
            # Send customer email notification
            try:
                mail_ok, mail_msg = send_order_status_email(updated_order, old_status, new_status)
                if mail_ok:
                    flash(f"Order #{order_id} status updated to '{new_status}'. Customer notification email dispatched.", "success")
                else:
                    flash(f"Order #{order_id} status updated to '{new_status}'. Note: {mail_msg}", "info")
            except Exception as mail_err:
                print(f"Notice: Email dispatch skipped or encountered an issue: {mail_err}")
                flash(f"Order #{order_id} status updated to '{new_status}'. Email notification could not be sent.", "warning")
        else:
            flash("Invalid status specified.", "danger")
    except Exception as e:
        flash(f"Failed to update order status: {str(e)}", "danger")

    return redirect(url_for("admin_orders"))


@app.route("/admin/profile", methods=["GET", "POST"])
@admin_required
def admin_profile():
    """Admin Profile Management."""
    user = get_user_by_id(session["user_id"])
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        new_password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name:
            flash("Name cannot be empty.", "danger")
        elif new_password and new_password != confirm_password:
            flash("New passwords do not match.", "danger")
        else:
            p_hash = generate_password_hash(new_password) if new_password else None
            update_user_profile(user["id"], name, p_hash)
            session["name"] = name
            flash("Admin profile updated successfully.", "success")
            return redirect(url_for("admin_profile"))

    return render_template("admin_profile.html", user=user)


# -----------------------------------------------------------------------------
# Customer E-Commerce Routes
# -----------------------------------------------------------------------------
@app.route("/customer/dashboard")
@customer_required
def customer_dashboard():
    """
    Customer Dashboard:
    - Welcome banner
    - Quick actions (Shop Now, My Orders, Profile)
    - Featured products with CURRENT selling prices only
    - Recent customer orders
    - Strictly NO internal business metrics (cost, profit, DML, causal elasticity).
    """
    user_id = session.get("user_id")
    products = get_all_products()
    orders = get_user_orders(user_id)[:5]
    total_orders = len(get_user_orders(user_id))
    notifications = get_customer_notifications(user_id, limit=5)

    return render_template(
        "customer_dashboard.html",
        featured_products=products[:3],
        popular_products=products[3:6] if len(products) > 3 else products,
        recent_orders=orders,
        total_orders=total_orders,
        notifications=notifications,
    )


@app.route("/customer/home")
def customer_home():
    """Customer E-Commerce Landing Page."""
    products = get_all_products()
    return render_template(
        "customer_home.html",
        featured_products=products[:3],
        popular_products=products[3:6] if len(products) > 3 else products,
    )


@app.route("/products")
def products():
    """Customer Product Catalog with Search, Category Filter, and Sorting."""
    category = request.args.get("category")
    search = request.args.get("search")
    sort_by = request.args.get("sort_by")

    prods = get_all_products(category=category, search=search, sort_by=sort_by)
    return render_template(
        "customer_products.html",
        products=prods,
        selected_category=category or "all",
        search_query=search or "",
        sort_by=sort_by or "",
    )


@app.route("/product/<product_id>")
def product_detail(product_id: str):
    """
    Customer Product Detail Page:
    Shows: Name, Image, Description, Rating, Current Price, Availability.
    Strictly hides: Cost, Profit, Causal Elasticity, Internal ML calculations.
    """
    prod = get_product_by_id(product_id)
    if not prod:
        flash("Product not found.", "danger")
        return redirect(url_for("products"))

    related_products = [p for p in get_all_products() if p["product_id"] != product_id][:3]
    return render_template("customer_product_detail.html", product=prod, related_products=related_products)


@app.route("/cart")
@customer_required
def cart():
    """Customer Shopping Cart."""
    user_id = session.get("user_id")
    cart_items = get_user_cart(user_id)
    total_amount = sum(item["subtotal"] for item in cart_items)
    return render_template("customer_cart.html", items=cart_items, total=total_amount)


@app.route("/api/cart/add", methods=["POST"])
def api_add_to_cart():
    """AJAX endpoint to add product to cart."""
    if "user_id" not in session:
        return jsonify({"status": "unauthorized", "redirect": url_for("login")}), 401

    try:
        data = request.get_json() or {}
        product_id = data.get("product_id")
        quantity = int(data.get("quantity", 1))

        if not product_id or quantity <= 0:
            return jsonify({"status": "error", "message": "Invalid product or quantity"}), 400

        add_to_cart(session["user_id"], product_id, quantity)
        new_count = get_cart_count(session["user_id"])
        return jsonify({
            "status": "success",
            "message": "Product added to cart!",
            "cart_count": new_count,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/cart/update", methods=["POST"])
@customer_required
def api_update_cart():
    """AJAX endpoint to modify quantity in cart."""
    try:
        data = request.get_json() or {}
        product_id = data.get("product_id")
        quantity = int(data.get("quantity", 1))

        update_cart_item_quantity(session["user_id"], product_id, quantity)
        new_count = get_cart_count(session["user_id"])
        return jsonify({"status": "success", "cart_count": new_count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/cart/remove", methods=["POST"])
@customer_required
def api_remove_cart():
    """AJAX endpoint to remove item from cart."""
    try:
        data = request.get_json() or {}
        product_id = data.get("product_id")
        remove_from_cart(session["user_id"], product_id)
        new_count = get_cart_count(session["user_id"])
        return jsonify({"status": "success", "cart_count": new_count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/checkout", methods=["GET", "POST"])
@customer_required
def checkout():
    """Customer Checkout Flow (Cash on Delivery)."""
    user_id = session.get("user_id")
    cart_items = get_user_cart(user_id)

    if not cart_items:
        flash("Your shopping cart is empty.", "info")
        return redirect(url_for("products"))

    total_amount = sum(item["subtotal"] for item in cart_items)
    user = get_user_by_id(user_id)

    if request.method == "POST":
        customer_name = request.form.get("name", "").strip() or user["name"]
        customer_email = request.form.get("email", "").strip() or user["email"]
        address = request.form.get("address", "").strip()
        phone = request.form.get("phone", "").strip()

        if not address or not phone:
            flash("Delivery address and phone number are required.", "danger")
            return render_template("customer_checkout.html", items=cart_items, total=total_amount, user=user)

        # Create Order in SQLite
        order_id = create_order(
            user_id=user_id,
            customer_name=customer_name,
            customer_email=customer_email,
            delivery_address=address,
            phone=phone,
            items=cart_items,
            payment_method="Cash on Delivery",
        )

        # Send Customer Order Confirmation Email
        try:
            created_order = get_order_by_id(order_id)
            mail_ok, mail_msg = send_order_confirmation_email(created_order)
            if not mail_ok:
                flash("Order placed successfully. Email notification could not be sent.", "warning")
            else:
                flash("Order placed successfully! A confirmation email has been dispatched to your inbox.", "success")
        except Exception as mail_err:
            print(f"Notice: Order confirmation email error: {mail_err}")
            flash("Order placed successfully. Email notification could not be sent.", "warning")

        return redirect(url_for("order_success", order_id=order_id))

    return render_template("customer_checkout.html", items=cart_items, total=total_amount, user=user)


@app.route("/order-success/<int:order_id>")
@customer_required
def order_success(order_id: int):
    """Order Confirmation Receipt."""
    user_id = session.get("user_id")
    order = get_order_by_id(order_id)
    if not order or order["user_id"] != user_id:
        flash("Order record not found.", "danger")
        return redirect(url_for("customer_dashboard"))

    return render_template("customer_order_success.html", order=order)


@app.route("/customer/orders")
@customer_required
def customer_orders():
    """Customer Order History (Isolated to current customer)."""
    user_id = session.get("user_id")
    orders = get_user_orders(user_id)
    return render_template("customer_orders.html", orders=orders)


@app.route("/customer/orders/<int:order_id>/track")
@login_required
def customer_order_track(order_id: int):
    """
    Visual Order Tracking Timeline:
    - Verifies order.user_id == current_user.id (or admin access)
    - Displays chronological visual timeline with current status highlighted
    - Shows order items, delivery address, estimated delivery date
    """
    order = get_order_by_id(order_id)
    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("customer_orders"))

    # Security check: Customer can only track their own order. Admin can view any order.
    if session.get("role") != "admin" and order["user_id"] != session.get("user_id"):
        flash("Access Denied: You are not authorized to view tracking for this order.", "danger")
        return redirect(url_for("customer_orders"))

    history = get_order_status_history(order_id)
    return render_template("customer_order_track.html", order=order, history=history)


@app.route("/customer/profile", methods=["GET", "POST"])
@customer_required
def customer_profile():
    """Customer Profile: Update Name & Password."""
    user = get_user_by_id(session["user_id"])
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        new_password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name:
            flash("Name cannot be blank.", "danger")
        elif new_password and new_password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            p_hash = generate_password_hash(new_password) if new_password else None
            update_user_profile(user["id"], name, p_hash)
            session["name"] = name
            flash("Profile updated successfully.", "success")
            return redirect(url_for("customer_profile"))

    return render_template("customer_profile.html", user=user)


# -----------------------------------------------------------------------------
# AI / ML / DML REST API Endpoints (Admin Controlled)
# -----------------------------------------------------------------------------
@app.route("/api/products", methods=["GET"])
def api_products():
    """Returns product catalog for UI components."""
    try:
        products = get_all_products()
        return jsonify({"status": "success", "products": products})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/summary", methods=["GET"])
@admin_required
def api_summary():
    """Returns high-level business and causal KPIs."""
    try:
        sales_sum = get_sales_summary()
        _, dml = get_models()

        causal_ate = dml.ate if (dml and dml.is_trained) else -0.0152
        ci_lower = dml.ate_ci_lower if (dml and dml.is_trained) else -0.0185
        ci_upper = dml.ate_ci_upper if (dml and dml.is_trained) else -0.0119

        recent_preds = get_recent_predictions(limit=5)

        summary_data = {
            "total_products": sales_sum.get("total_products", 6),
            "total_sales_records": sales_sum.get("total_sales_records", 12000),
            "avg_price": round(sales_sum.get("avg_price", 0.0), 2),
            "avg_demand": round(sales_sum.get("avg_demand", 0.0), 1),
            "current_revenue": round(sales_sum.get("total_revenue", 0.0), 2),
            "current_profit": round(sales_sum.get("total_profit", 0.0), 2),
            "estimated_causal_effect": round(causal_ate, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "interpretation": (
                f"Conditional on market covariates, a Rs. 1 price increase changes expected demand by "
                f"~{causal_ate:.4f} units (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])."
            ),
            "recent_predictions": recent_preds,
        }
        return jsonify({"status": "success", "data": summary_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
@admin_required
def api_predict():
    """Predicts demand given transaction context using Baseline ML."""
    try:
        data = request.get_json() or {}
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        dm, _ = get_models()
        if dm is None or not dm.is_trained:
            price = float(data.get("price", 2499.0))
            comp_p = float(data.get("competitor_price", price * 1.02))
            promo = int(data.get("promotion", 0))
            holiday = int(data.get("holiday", 0))
            pred_demand = max(1.0, round(50.0 + (0.2 * (comp_p - price)) + (15.0 * promo) + (12.0 * holiday) - (0.015 * (price - 2499.0)), 1))
            return jsonify({
                "status": "success",
                "predicted_demand": pred_demand,
                "model_used": "Heuristic Fallback (Training in progress)",
            })

        df_in = pd.DataFrame([{
            "price": float(data.get("price", 2499.0)),
            "competitor_price": float(data.get("competitor_price", 2550.0)),
            "promotion": int(data.get("promotion", 0)),
            "holiday": int(data.get("holiday", 0)),
            "inventory": int(data.get("inventory", 150)),
            "temperature": float(data.get("temperature", 24.0)),
            "day_of_week": str(data.get("day_of_week", "Wednesday")),
            "month": int(data.get("month", 5)),
            "season": str(data.get("season", "Spring")),
            "category": str(data.get("category", "Electronics")),
        }])

        pred_demand = float(dm.predict(df_in)[0])
        return jsonify({
            "status": "success",
            "predicted_demand": round(pred_demand, 1),
            "model_used": "RandomForestRegressor (Supervised)",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/recommend-price", methods=["POST"])
@admin_required
def api_recommend_price():
    """
    Computes profit-maximizing and revenue-maximizing dynamic price
    using EconML Double Machine Learning causal price elasticity.
    """
    try:
        data = request.get_json() or {}
        if not data:
            return jsonify({"status": "error", "message": "Payload is empty."}), 400

        product_id = str(data.get("product_id", "PROD_001"))
        current_price = float(data.get("current_price", 0))
        cost = float(data.get("cost", 0))
        competitor_price = float(data.get("competitor_price", current_price * 1.02))
        promotion = int(data.get("promotion", 0))
        holiday = int(data.get("holiday", 0))
        inventory = int(data.get("inventory", 150))
        temperature = float(data.get("temperature", 25.0))
        day_of_week = str(data.get("day_of_week", "Wednesday"))
        month = int(data.get("month", 6))
        season = str(data.get("season", "Summer"))

        if current_price <= 0:
            return jsonify({"status": "error", "message": "Current price must be greater than zero."}), 400
        if cost <= 0:
            return jsonify({"status": "error", "message": "Cost must be greater than zero."}), 400
        if cost >= current_price:
            return jsonify({"status": "error", "message": "Cost cannot be greater than or equal to current price."}), 400

        prod_meta = get_product_by_id(product_id)
        prod_name = prod_meta["product_name"] if prod_meta else "Product"
        category = prod_meta["category"] if prod_meta else "Electronics"

        dm, dml = get_models()
        if dm and dm.is_trained:
            df_in = pd.DataFrame([{
                "price": current_price,
                "competitor_price": competitor_price,
                "promotion": promotion,
                "holiday": holiday,
                "inventory": inventory,
                "temperature": temperature,
                "day_of_week": day_of_week,
                "month": month,
                "season": season,
                "category": category,
            }])
            baseline_demand = float(dm.predict(df_in)[0])
        else:
            baseline_demand = max(5.0, round(50.0 + (0.2 * (competitor_price - current_price)) + (14.0 * promotion) + (10.0 * holiday), 1))

        base_elasticity = dml.ate if (dml and dml.is_trained) else -0.0152
        if current_price > 10000:
            elasticity = base_elasticity * (2500.0 / current_price)
        else:
            elasticity = base_elasticity

        result = pricing_engine.optimize_price(
            current_price=current_price,
            cost=cost,
            baseline_demand=baseline_demand,
            causal_elasticity=elasticity,
            product_id=product_id,
            product_name=prod_name,
        )

        try:
            log_prediction(
                product_id=product_id,
                current_price=current_price,
                recommended_price=result["recommended_price"],
                optimal_revenue_price=result["revenue_maximizing_price"],
                expected_demand=result["expected_demand"],
                expected_revenue=result["expected_revenue"],
                expected_profit=result["expected_profit"],
                causal_elasticity=result["causal_elasticity"],
            )
        except Exception as db_err:
            print(f"Notice: Log prediction skipped ({db_err})")

        return jsonify({"status": "success", "recommendation": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/causal-effect", methods=["GET"])
@admin_required
def api_causal_effect():
    """Returns causal estimation results, CI, and methodological notes."""
    try:
        _, dml = get_models()
        if dml and dml.is_trained:
            data = dml.get_causal_effect()
        else:
            data = {
                "ate": -0.0152,
                "ci_lower": -0.0185,
                "ci_upper": -0.0119,
                "interpretation": (
                    "If price increases by Rs. 1, estimated demand changes by approximately "
                    "-0.0152 units (95% CI: [-0.0185, -0.0119]), conditional on the observed covariates."
                ),
            }

        data["methodology"] = "EconML LinearDML (Robinson Transformation with Cross-Fitting)"
        data["treatment"] = "price (Continuous Offered Price in ₹)"
        data["outcome"] = "demand (Units Sold)"
        data["confounders"] = [
            "competitor_price", "promotion", "holiday", "inventory",
            "temperature", "day_of_week", "month", "season", "category"
        ]
        data["scientific_disclaimer"] = (
            "This project demonstrates causal estimation methodology using simulated/observational-style data. "
            "It does not claim that synthetic observational data proves a real-world causal relationship."
        )

        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics", methods=["GET"])
@admin_required
def api_analytics():
    """Returns aggregated historical relationships for Chart.js."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Price vs Demand & Revenue & Profit
        cursor.execute("""
            SELECT price, demand, (price * demand) as revenue, ((price - cost) * demand) as profit, competitor_price, inventory
            FROM sales
            WHERE product_id = 'PROD_001'
            ORDER BY RANDOM()
            LIMIT 200
        """)
        scatter_rows = [dict(r) for r in cursor.fetchall()]

        # 2. Demand by Category
        cursor.execute("""
            SELECT p.category, ROUND(AVG(s.demand), 1) as avg_demand, ROUND(AVG(s.price), 2) as avg_price, COUNT(s.id) as count
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            GROUP BY p.category
        """)
        category_rows = [dict(r) for r in cursor.fetchall()]

        # 3. Monthly Demand Trend
        cursor.execute("""
            SELECT month, ROUND(AVG(demand), 1) as avg_demand, ROUND(AVG(price * demand), 2) as avg_revenue
            FROM sales
            GROUP BY month
            ORDER BY month ASC
        """)
        monthly_rows = [dict(r) for r in cursor.fetchall()]

        # 4. Promotion vs Demand
        cursor.execute("""
            SELECT promotion, ROUND(AVG(demand), 1) as avg_demand, ROUND(AVG(price), 2) as avg_price, COUNT(id) as count
            FROM sales
            GROUP BY promotion
        """)
        promo_rows = [dict(r) for r in cursor.fetchall()]

        # 5. Seasonal Demand
        cursor.execute("""
            SELECT season, ROUND(AVG(demand), 1) as avg_demand, ROUND(AVG(price), 2) as avg_price
            FROM sales
            GROUP BY season
            ORDER BY CASE season
                WHEN 'Spring' THEN 1
                WHEN 'Summer' THEN 2
                WHEN 'Autumn' THEN 3
                WHEN 'Winter' THEN 4
            END
        """)
        season_rows = [dict(r) for r in cursor.fetchall()]

        conn.close()

        analytics_data = {
            "scatter_data": scatter_rows,
            "category_data": category_rows,
            "monthly_data": monthly_rows,
            "promotion_data": promo_rows,
            "season_data": season_rows,
        }
        return jsonify({"status": "success", "analytics": analytics_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    init_db()
    get_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting EconoCausal Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
