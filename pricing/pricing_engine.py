"""
EconoCausal Dynamic Pricing Engine
Generates candidate prices around current catalog price, projects expected demand
using causal elasticity (DML), evaluates expected revenue and gross profit surfaces,
and identifies profit-maximizing and revenue-maximizing recommendations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class DynamicPricingEngine:
    """
    Microeconomically grounded dynamic pricing engine.
    Finds revenue-maximizing and profit-maximizing optimal prices.
    """

    def __init__(
        self,
        min_margin_pct: float = 0.10,
        price_step_pct: float = 0.05,
        search_radius_pct: float = 0.25,
    ):
        self.min_margin_pct = min_margin_pct
        self.price_step_pct = price_step_pct
        self.search_radius_pct = search_radius_pct

    def generate_candidate_prices(self, current_price: float, cost: float) -> List[float]:
        """
        Generates discrete candidate prices around current price (e.g. +/- 25% in 5% intervals).
        Guarantees that prices do not drop below marginal cost * (1 + min_margin_pct).
        """
        min_price = max(cost * (1.0 + self.min_margin_pct), current_price * (1.0 - self.search_radius_pct))
        step = max(5.0, round(current_price * self.price_step_pct, 0))
        max_price = max(min_price + step, current_price * (1.0 + self.search_radius_pct))
        prices = list(np.arange(min_price, max_price + (step / 2.0), step))

        # Include current price if it meets or exceeds the minimum margin floor
        min_margin_floor = round(cost * (1.0 + self.min_margin_pct), 2)
        if current_price >= min_margin_floor and current_price not in prices:
            prices.append(current_price)

        prices = sorted(list(set([round(float(p), 2) for p in prices if round(float(p), 2) >= min_margin_floor])))
        return prices

    def optimize_price(
        self,
        current_price: float,
        cost: float,
        baseline_demand: float,
        causal_elasticity: float,
        product_id: str = "PROD_001",
        product_name: str = "Product",
    ) -> Dict[str, Any]:
        """
        Evaluates candidate prices, calculates demand, revenue, profit,
        and determines optimal pricing policies.
        """
        candidates = self.generate_candidate_prices(current_price, cost)
        grid: List[Dict[str, Any]] = []

        max_profit = -float("inf")
        max_revenue = -float("inf")
        best_profit_price = current_price
        best_revenue_price = current_price

        # Demand projection formula:
        # Demand(P) = Baseline_Demand + theta * (P - Current_Price)
        # Note: causal_elasticity (theta) is negative
        theta = float(causal_elasticity)
        if theta >= 0:
            theta = -0.01  # Safe microeconomic fallback ensuring downward sloping demand

        for p in candidates:
            delta_p = p - current_price
            # Demand decreases as price increases
            projected_demand = max(1.0, baseline_demand + (theta * delta_p))
            projected_demand = round(float(projected_demand), 1)

            revenue = round(p * projected_demand, 2)
            profit = round((p - cost) * projected_demand, 2)

            is_curr = (p == current_price)
            if profit > max_profit:
                max_profit = profit
                best_profit_price = p

            if revenue > max_revenue:
                max_revenue = revenue
                best_revenue_price = p

            grid.append({
                "price": p,
                "demand": projected_demand,
                "revenue": revenue,
                "profit": profit,
                "margin_pct": round(((p - cost) / p) * 100.0, 1),
                "is_current": is_curr,
                "is_best_profit": False,
                "is_best_revenue": False,
            })

        # Mark optimal rows
        for row in grid:
            if row["price"] == best_profit_price:
                row["is_best_profit"] = True
            if row["price"] == best_revenue_price:
                row["is_best_revenue"] = True

        # Current price baseline metrics
        curr_row = next((r for r in grid if r["is_current"]), grid[len(grid) // 2])
        best_row = next((r for r in grid if r["is_best_profit"]), grid[len(grid) // 2])
        rev_row = next((r for r in grid if r["is_best_revenue"]), grid[len(grid) // 2])

        profit_uplift_pct = (
            ((best_row["profit"] - curr_row["profit"]) / max(1.0, curr_row["profit"])) * 100.0
            if curr_row["profit"] > 0
            else 0.0
        )
        revenue_uplift_pct = (
            ((best_row["revenue"] - curr_row["revenue"]) / max(1.0, curr_row["revenue"])) * 100.0
            if curr_row["revenue"] > 0
            else 0.0
        )

        return {
            "product_id": product_id,
            "product_name": product_name,
            "current_price": current_price,
            "cost": cost,
            "causal_elasticity": round(theta, 4),
            "recommended_price": best_profit_price,
            "expected_demand": best_row["demand"],
            "expected_revenue": best_row["revenue"],
            "expected_profit": best_row["profit"],
            "profit_uplift_pct": round(profit_uplift_pct, 2),
            "revenue_uplift_pct": round(revenue_uplift_pct, 2),
            "revenue_maximizing_price": best_revenue_price,
            "revenue_at_max_revenue_price": rev_row["revenue"],
            "profit_at_max_revenue_price": rev_row["profit"],
            "baseline_demand": curr_row["demand"],
            "baseline_revenue": curr_row["revenue"],
            "baseline_profit": curr_row["profit"],
            "candidate_grid": grid,
        }
