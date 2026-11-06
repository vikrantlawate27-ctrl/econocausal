"""
EconoCausal Data Generation Script
Generates a realistic synthetic sales dataset (12,000 records)
with price endogeneity, competitor interactions, promotions, seasonal cycles,
and realistic microeconomic causal price sensitivity.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sales_dataset(n_records: int = 12000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    os.makedirs("data", exist_ok=True)

    # 1. Product catalog definition
    products = [
        {"id": "PROD_001", "name": "Pro Wireless Headphones", "category": "Electronics", "base_price": 2499.0, "cost": 1350.0, "base_demand": 48.0, "price_elasticity": -0.016},
        {"id": "PROD_002", "name": "Ultra HD Smart TV 43-inch", "category": "Electronics", "base_price": 24999.0, "cost": 16000.0, "base_demand": 22.0, "price_elasticity": -0.00095},
        {"id": "PROD_003", "name": "Ergonomic Mesh Office Chair", "category": "Home & Office", "base_price": 6499.0, "cost": 3600.0, "base_demand": 35.0, "price_elasticity": -0.0042},
        {"id": "PROD_004", "name": "Smart Fitness Tracker Watch", "category": "Wearables", "base_price": 3199.0, "cost": 1700.0, "base_demand": 62.0, "price_elasticity": -0.0135},
        {"id": "PROD_005", "name": "Noise-Cancelling True Wireless Earbuds", "category": "Electronics", "base_price": 1899.0, "cost": 950.0, "base_demand": 75.0, "price_elasticity": -0.022},
        {"id": "PROD_006", "name": "Standing Electric Desk 120cm", "category": "Home & Office", "base_price": 14999.0, "cost": 9200.0, "base_demand": 18.0, "price_elasticity": -0.0012},
    ]

    start_date = datetime(2025, 1, 1)
    date_range = [start_date + timedelta(days=int(i)) for i in range(n_records // len(products) + 50)]

    records = []
    
    for prod in products:
        p_id = prod["id"]
        p_name = prod["name"]
        cat = prod["category"]
        base_p = prod["base_price"]
        cost = prod["cost"]
        base_q = prod["base_demand"]
        true_theta = prod["price_elasticity"]

        # Number of daily observations for this product
        n_obs = n_records // len(products)

        for i in range(n_obs):
            current_date = date_range[i % len(date_range)]
            month = current_date.month
            day_of_week = current_date.strftime("%A")
            is_weekend = 1 if current_date.weekday() >= 5 else 0

            # Season mapping
            if month in [12, 1, 2]:
                season = "Winter"
                temp = rng.normal(16.5, 3.5)
            elif month in [3, 4, 5]:
                season = "Spring"
                temp = rng.normal(26.0, 4.0)
            elif month in [6, 7, 8]:
                season = "Summer"
                temp = rng.normal(33.5, 3.8)
            else:
                season = "Autumn"
                temp = rng.normal(24.0, 3.2)
            
            temp = float(np.round(np.clip(temp, 10.0, 44.0), 1))

            # Holiday probability (higher in Oct-Dec & Jan)
            holiday_prob = 0.14 if month in [10, 11, 12, 1] else 0.04
            holiday = 1 if rng.random() < holiday_prob else 0

            # Promotion flag
            promo_prob = 0.28 if (holiday == 1 or is_weekend == 1) else 0.12
            promotion = 1 if rng.random() < promo_prob else 0

            # Inventory level (15 to 450 units)
            inventory = int(np.clip(rng.normal(180, 50) - (25 if holiday else 0), 15, 450))

            # Seasonal demand multiplier
            season_mult = 1.25 if season in ["Winter", "Autumn"] else 0.95

            # Competitor pricing
            comp_ratio = rng.normal(1.02, 0.06)
            competitor_price = float(np.round(base_p * comp_ratio, 2))

            # ENDOGENEITY IN HISTORICAL PRICING:
            # Retail managers systematically raise prices during holidays, high competitor prices,
            # and low inventory, but offer discounts during official promotions.
            pricing_shock = (
                0.04 * (season_mult - 1.0) * base_p
                + 0.05 * (holiday * base_p)
                + 0.15 * (competitor_price - base_p)
                - 0.08 * (promotion * base_p)
                - 0.03 * ((inventory - 180) / 180.0) * base_p
                + rng.normal(0, 0.03 * base_p)
            )
            price = float(np.round(np.clip(base_p + pricing_shock, cost * 1.15, base_p * 1.45), 2))

            # STRUCTURAL CAUSAL DEMAND:
            # 1. Base demand modulated by season, holiday, promo, weekend
            demand_intercept = base_q * season_mult * (1.30 if holiday else 1.0) * (1.28 if promotion else 1.0) * (1.12 if is_weekend else 1.0)
            
            # 2. Competitor cross-price effect (if competitor is expensive, our demand rises)
            comp_effect = 0.22 * ((competitor_price - base_p) / base_p) * base_q

            # 3. TRUE CAUSAL PRICE EFFECT: (price - base_price) * true_theta * base_q
            # Downward sloping demand curve
            causal_price_effect = true_theta * (price - base_p)

            # 4. Unobserved shock
            noise = rng.normal(0, max(2.0, base_q * 0.08))

            demand = int(np.maximum(1, np.round(demand_intercept + comp_effect + causal_price_effect + noise)))

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "product_id": p_id,
                "product_name": p_name,
                "category": cat,
                "price": price,
                "cost": float(cost),
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

    df = pd.DataFrame(records)
    # Shuffle records chronologically or slightly mixed
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

    csv_path = "data/sales_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df):,} records successfully -> {csv_path}")
    return df

if __name__ == "__main__":
    generate_sales_dataset()
