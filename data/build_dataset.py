"""
Fast, self-contained dataset generator for EconoCausal
Generates 12,000 realistic records directly to data/sales_data.csv
Runs in under 1 second without heavy dependencies.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

def build_sales_data(n_records: int = 12000, output_path: str = "data/sales_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(42)

    # 6 Realistic products across 3 categories
    products = [
        {"id": "PROD_001", "name": "Pro Wireless Headphones", "category": "Electronics", "base_price": 2499.0, "cost": 1350.0, "base_demand": 50, "elasticity": -0.018},
        {"id": "PROD_002", "name": "Ultra HD Smart TV 43-inch", "category": "Electronics", "base_price": 24999.0, "cost": 16000.0, "base_demand": 24, "elasticity": -0.0011},
        {"id": "PROD_003", "name": "Ergonomic Mesh Office Chair", "category": "Home & Office", "base_price": 6499.0, "cost": 3600.0, "base_demand": 38, "elasticity": -0.0045},
        {"id": "PROD_004", "name": "Smart Fitness Tracker Watch", "category": "Wearables", "base_price": 3199.0, "cost": 1700.0, "base_demand": 65, "elasticity": -0.014},
        {"id": "PROD_005", "name": "Noise-Cancelling True Wireless Earbuds", "category": "Electronics", "base_price": 1899.0, "cost": 950.0, "base_demand": 80, "elasticity": -0.024},
        {"id": "PROD_006", "name": "Standing Electric Desk 120cm", "category": "Home & Office", "base_price": 14999.0, "cost": 9200.0, "base_demand": 19, "elasticity": -0.0013},
    ]

    fieldnames = [
        "date", "product_id", "product_name", "category", "price", "cost",
        "demand", "competitor_price", "promotion", "holiday", "inventory",
        "temperature", "day_of_week", "month", "season"
    ]

    start_date = datetime(2025, 1, 1)
    records_per_prod = n_records // len(products)
    all_rows = []

    for prod in products:
        p_id = prod["id"]
        p_name = prod["name"]
        cat = prod["category"]
        base_p = prod["base_price"]
        cost = prod["cost"]
        base_d = prod["base_demand"]
        elasticity = prod["elasticity"]

        for i in range(records_per_prod):
            cur_date = start_date + timedelta(days=i % 730)
            month = cur_date.month
            day_of_week = cur_date.strftime("%A")
            is_weekend = 1 if cur_date.weekday() >= 5 else 0

            # Season mapping
            if month in [12, 1, 2]:
                season = "Winter"
                temp = round(random.gauss(16.0, 3.0), 1)
            elif month in [3, 4, 5]:
                season = "Spring"
                temp = round(random.gauss(26.0, 3.5), 1)
            elif month in [6, 7, 8]:
                season = "Summer"
                temp = round(random.gauss(34.0, 3.0), 1)
            else:
                season = "Autumn"
                temp = round(random.gauss(24.0, 3.0), 1)
            temp = max(10.0, min(45.0, temp))

            # Holiday probability
            holiday_prob = 0.15 if month in [10, 11, 12, 1] else 0.04
            holiday = 1 if random.random() < holiday_prob else 0

            # Promotion probability
            promo_prob = 0.30 if (holiday or is_weekend) else 0.12
            promotion = 1 if random.random() < promo_prob else 0

            # Inventory level
            inventory = int(max(15, min(480, random.gauss(180, 45) - (30 if holiday else 0))))

            # Season multiplier
            season_mult = 1.25 if season in ["Winter", "Autumn"] else 0.95

            # Competitor pricing
            comp_ratio = random.gauss(1.02, 0.05)
            competitor_price = round(base_p * comp_ratio, 2)

            # Endogenous historical pricing:
            # Managers raise prices on holidays, high competitor price, peak season, but discount during promo
            price_shock = (
                0.05 * (season_mult - 1.0) * base_p
                + 0.06 * (holiday * base_p)
                + 0.18 * (competitor_price - base_p)
                - 0.10 * (promotion * base_p)
                - 0.03 * ((inventory - 180) / 180.0) * base_p
                + random.gauss(0, 0.025 * base_p)
            )
            price = round(max(cost * 1.15, min(base_p * 1.40, base_p + price_shock)), 2)

            # Structural Causal Demand
            demand_base = base_d * season_mult * (1.32 if holiday else 1.0) * (1.28 if promotion else 1.0) * (1.12 if is_weekend else 1.0)
            comp_effect = 0.25 * ((competitor_price - base_p) / base_p) * base_d
            causal_effect = elasticity * (price - base_p)
            noise = random.gauss(0, max(2.0, base_d * 0.06))

            demand = int(max(1, round(demand_base + comp_effect + causal_effect + noise)))

            all_rows.append({
                "date": cur_date.strftime("%Y-%m-%d"),
                "product_id": p_id,
                "product_name": p_name,
                "category": cat,
                "price": price,
                "cost": cost,
                "demand": demand,
                "competitor_price": competitor_price,
                "promotion": promotion,
                "holiday": holiday,
                "inventory": inventory,
                "temperature": temp,
                "day_of_week": day_of_week,
                "month": month,
                "season": season,
            })

    # Sort rows by date and product_id
    all_rows.sort(key=lambda r: (r["date"], r["product_id"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"SUCCESS: Generated {len(all_rows):,} records -> {output_path}")

if __name__ == "__main__":
    build_sales_data()
