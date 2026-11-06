"""
EconoCausal - Email Notification Service
Handles transactional order emails:
  1. Order Confirmation Email (send_order_confirmation_email)
  2. Order Status Update Email (send_order_status_email)
  3. Order Delivered Notification (send_delivery_email)

Uses standard smtplib with TLS and supports Gmail App Passwords.
Safely logs errors without crashing or blocking order creation.
"""

import os
import sys
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, Tuple, Optional, Union

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("econocausal.email")


def get_mail_config() -> Dict[str, Any]:
    """Retrieves SMTP configuration from environment variables."""
    return {
        "server": os.environ.get("MAIL_SERVER", "smtp.gmail.com"),
        "port": int(os.environ.get("MAIL_PORT", 587)),
        "username": os.environ.get("MAIL_USERNAME", "").strip(),
        "password": os.environ.get("MAIL_PASSWORD", "").strip(),
        "use_tls": os.environ.get("MAIL_USE_TLS", "True").lower() in ("true", "1", "yes"),
        "sender": os.environ.get("MAIL_DEFAULT_SENDER", "EconoCausal Orders <noreply@econocausal.com>").strip(),
        "app_base_url": os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/"),
    }


def _send_email_message(to_email: str, subject: str, text_body: str, html_body: str) -> Tuple[bool, str]:
    """
    Sends an email using standard SMTP with TLS.
    If credentials are placeholder/unconfigured, safely logs the simulated email.
    Never throws an unhandled exception.
    """
    config = get_mail_config()
    server_host = config["server"]
    server_port = config["port"]
    username = config["username"]
    password = config["password"]
    use_tls = config["use_tls"]
    sender = config["sender"]

    # Check if credentials are mock/default placeholders
    is_mock = (
        not username
        or not password
        or "your-email" in username
        or "your-app-password" in password
    )

    if is_mock:
        msg_log = (
            f"\n[EMAIL NOTIFICATION (MOCK/DEV MODE)]\n"
            f"To: {to_email}\n"
            f"Subject: {subject}\n"
            f"Sender: {sender}\n"
            f"--- Text Body ---\n{text_body}\n"
            f"----------------------------------------"
        )
        try:
            print(msg_log)
        except (UnicodeEncodeError, Exception):
            try:
                safe_encoding = sys.stdout.encoding or "utf-8"
                print(msg_log.encode(safe_encoding, errors="replace").decode(safe_encoding))
            except Exception:
                pass
        logger.info(f"Simulated email sent to {to_email}: {subject}")
        return True, "Simulated email logged (configure .env with real SMTP credentials to send live emails)."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        # Attach text & HTML parts
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # Connect to SMTP server
        with smtplib.SMTP(server_host, server_port, timeout=10) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.sendmail(sender, [to_email], msg.as_string())

        logger.info(f"Email successfully delivered to {to_email}: {subject}")
        return True, "Email sent successfully."
    except Exception as e:
        err_msg = f"Failed to send email to {to_email}: {str(e)}"
        print(f"[EMAIL ERROR]: {err_msg}")
        logger.error(err_msg)
        return False, err_msg


# -----------------------------------------------------------------------------
# Transactional Order Emails
# -----------------------------------------------------------------------------
def send_order_confirmation_email(order: Union[int, Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Sends order confirmation email to the customer upon checkout.
    Accepts either order_id (int) or order dictionary.
    """
    try:
        if isinstance(order, (int, str)):
            from database.database import get_order_by_id
            order_data = get_order_by_id(int(order))
            if not order_data:
                return False, f"Order #{order} not found."
            order = order_data

        order_id = order.get("id")
        customer_name = order.get("customer_name", "Customer")
        to_email = order.get("customer_email", "").strip()
        order_date = order.get("order_date", "")
        est_delivery = order.get("estimated_delivery", "3-5 Business Days")
        delivery_address = order.get("delivery_address", "")
        total_amount = float(order.get("total_amount", 0.0))
        items = order.get("items", [])
        config = get_mail_config()
        track_url = f"{config['app_base_url']}/customer/orders/{order_id}/track"

        if not to_email:
            return False, "No recipient email address found on order."

        subject = f"EconoCausal - Order Confirmation #{order_id}"

        # Build items text & HTML
        items_text_lines = []
        items_html_rows = []
        subtotal = 0.0

        for item in items:
            pname = item.get("product_name", "Product")
            qty = int(item.get("quantity", 1))
            price = float(item.get("price", item.get("current_price", 0.0)))
            item_subtotal = float(item.get("subtotal", price * qty))
            subtotal += item_subtotal

            items_text_lines.append(f"- {pname} x {qty} @ ₹{price:,.2f} = ₹{item_subtotal:,.2f}")
            items_html_rows.append(f"""
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{pname}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center;">{qty}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right;">₹{price:,.2f}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: bold;">₹{item_subtotal:,.2f}</td>
                </tr>
            """)

        items_text_block = "\n".join(items_text_lines)
        items_html_block = "".join(items_html_rows)

        # Plain Text Fallback
        text_body = f"""Hello {customer_name},

Thank you for your order!

Order ID: #{order_id}
Order Date: {order_date}

Products:
{items_text_block}

Subtotal: ₹{subtotal:,.2f}
Shipping: FREE
Total Amount: ₹{total_amount:,.2f}

Delivery Address:
{delivery_address}

Current Order Status:
Order Confirmed

Estimated Delivery:
{est_delivery}

Track Order:
{track_url}

Regards,
EconoCausal Team
"""

        # Rich HTML Email
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #0f172a, #1e293b); color: #ffffff; padding: 25px; text-align: center; }}
                .header h1 {{ margin: 0 0 5px; font-size: 22px; font-weight: 800; }}
                .header p {{ margin: 0; font-size: 13px; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; }}
                .body {{ padding: 25px; }}
                .status-badge {{ display: inline-block; background-color: #d1fae5; color: #065f46; font-weight: bold; font-size: 13px; padding: 5px 12px; border-radius: 20px; margin: 10px 0; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
                .table th {{ background: #f1f5f9; padding: 10px; text-align: left; font-weight: 700; color: #334155; }}
                .btn {{ display: inline-block; background-color: #2563eb; color: #ffffff !important; font-weight: bold; text-decoration: none; padding: 12px 24px; border-radius: 8px; margin: 20px 0 10px; text-align: center; }}
                .footer {{ background: #f8fafc; padding: 20px; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>EconoCausal</h1>
                    <p>Dynamic Pricing Intelligence</p>
                </div>
                <div class="body">
                    <h2>Thank you for your order!</h2>
                    <p>Hello <strong>{customer_name}</strong>,</p>
                    <p>Your order <strong>#{order_id}</strong> has been successfully placed and confirmed.</p>
                    
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div><strong>Order Date:</strong> {order_date}</div>
                        <div><strong>Current Status:</strong> <span class="status-badge">Order Confirmed</span></div>
                        <div><strong>Estimated Delivery:</strong> <strong>{est_delivery}</strong></div>
                        <div style="margin-top: 5px;"><strong>Delivery Address:</strong> {delivery_address}</div>
                    </div>

                    <h3>Order Summary</h3>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Product</th>
                                <th style="text-align: center;">Qty</th>
                                <th style="text-align: right;">Price</th>
                                <th style="text-align: right;">Subtotal</th>
                            </tr>
                        </thead>
                        <tbody>
                            {items_html_block}
                        </tbody>
                        <tfoot>
                            <tr>
                                <td colspan="3" style="padding: 10px; text-align: right; font-weight: bold;">Subtotal:</td>
                                <td style="padding: 10px; text-align: right; font-weight: bold;">₹{subtotal:,.2f}</td>
                            </tr>
                            <tr>
                                <td colspan="3" style="padding: 10px; text-align: right; font-weight: bold;">Shipping:</td>
                                <td style="padding: 10px; text-align: right; font-weight: bold; color: #059669;">FREE</td>
                            </tr>
                            <tr style="font-size: 16px; background-color: #f8fafc;">
                                <td colspan="3" style="padding: 12px; text-align: right; font-weight: 800;">Total Payable:</td>
                                <td style="padding: 12px; text-align: right; font-weight: 800; color: #047857;">₹{total_amount:,.2f}</td>
                            </tr>
                        </tfoot>
                    </table>

                    <div style="text-align: center;">
                        <a href="{track_url}" class="btn">Track My Order</a>
                    </div>
                </div>
                <div class="footer">
                    <p>Regards,<br><strong>EconoCausal Team</strong></p>
                    <p>© 2026 EconoCausal - AI-Powered Dynamic Pricing Framework</p>
                </div>
            </div>
        </body>
        </html>
        """

        return _send_email_message(to_email, subject, text_body, html_body)
    except Exception as e:
        logger.error(f"Error preparing confirmation email: {e}")
        return False, str(e)


def send_order_status_email(order: Union[int, Dict[str, Any]], old_status: str, new_status: str) -> Tuple[bool, str]:
    """
    Sends order status update notification to the customer.
    Accepts either order_id (int) or order dictionary.
    """
    try:
        if isinstance(order, (int, str)):
            from database.database import get_order_by_id
            order_data = get_order_by_id(int(order))
            if not order_data:
                return False, f"Order #{order} not found."
            order = order_data

        order_id = order.get("id")
        customer_name = order.get("customer_name", "Customer")
        to_email = order.get("customer_email", "").strip()
        est_delivery = order.get("estimated_delivery", "In 3-5 Days")
        config = get_mail_config()
        track_url = f"{config['app_base_url']}/customer/orders/{order_id}/track"

        if not to_email:
            return False, "No recipient email address."

        # Status specific phrasing
        status_subject_verbs = {
            "Confirmed": "Has Been Confirmed",
            "Processing": "Is Now Being Processed",
            "Shipped": "Has Been Shipped",
            "Out for Delivery": "Is Out for Delivery",
            "Delivered": "Has Been Delivered",
            "Cancelled": "Has Been Cancelled",
        }
        subject_verb = status_subject_verbs.get(new_status, f"Status Updated to {new_status}")
        subject = f"EconoCausal - Your Order #{order_id} {subject_verb}"

        status_messages = {
            "Confirmed": "Your order has been confirmed.",
            "Processing": "Your order is now being processed.",
            "Shipped": "Your order has been shipped.",
            "Out for Delivery": "Your order is out for delivery.",
            "Delivered": "Your order has been delivered successfully.",
            "Cancelled": "Your order has been cancelled.",
        }
        status_msg = status_messages.get(new_status, f"Your order status has changed from {old_status} to {new_status}.")

        text_body = f"""Hello {customer_name},

{status_msg}

Current Status:
{new_status}

Estimated Delivery:
{est_delivery}

Track your order:
{track_url}

Thank you for shopping with EconoCausal.

Regards,
EconoCausal Team
"""

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #0f172a, #1e293b); color: #ffffff; padding: 25px; text-align: center; }}
                .header h1 {{ margin: 0 0 5px; font-size: 22px; font-weight: 800; }}
                .header p {{ margin: 0; font-size: 13px; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; }}
                .body {{ padding: 25px; }}
                .status-card {{ background: #f8fafc; border: 2px solid {'#ef4444' if new_status == 'Cancelled' else '#10b981' if new_status == 'Delivered' else '#3b82f6'}; border-radius: 8px; padding: 18px; margin: 20px 0; }}
                .btn {{ display: inline-block; background-color: #2563eb; color: #ffffff !important; font-weight: bold; text-decoration: none; padding: 12px 24px; border-radius: 8px; margin: 15px 0 10px; text-align: center; }}
                .footer {{ background: #f8fafc; padding: 20px; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>EconoCausal</h1>
                    <p>Dynamic Pricing Intelligence</p>
                </div>
                <div class="body">
                    <h2>Order Status Update</h2>
                    <p>Hello <strong>{customer_name}</strong>,</p>
                    <p>{status_msg}</p>

                    <div class="status-card">
                        <div style="font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 700;">Order ID: #{order_id}</div>
                        <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin: 4px 0 10px;">
                            Status: <span style="color: {'#dc2626' if new_status == 'Cancelled' else '#059669' if new_status == 'Delivered' else '#2563eb'};">{new_status}</span>
                        </div>
                        <div><strong>Estimated Delivery:</strong> {est_delivery}</div>
                    </div>

                    <div style="text-align: center;">
                        <a href="{track_url}" class="btn">Track Order</a>
                    </div>
                </div>
                <div class="footer">
                    <p>Thank you for shopping with EconoCausal.<br><strong>EconoCausal Team</strong></p>
                    <p>© 2026 EconoCausal - AI-Powered Dynamic Pricing Framework</p>
                </div>
            </div>
        </body>
        </html>
        """

        return _send_email_message(to_email, subject, text_body, html_body)
    except Exception as e:
        logger.error(f"Error preparing status update email: {e}")
        return False, str(e)


def send_delivery_email(order: Union[int, Dict[str, Any]]) -> Tuple[bool, str]:
    """Helper for delivered status notification."""
    return send_order_status_email(order, old_status="Out for Delivery", new_status="Delivered")
