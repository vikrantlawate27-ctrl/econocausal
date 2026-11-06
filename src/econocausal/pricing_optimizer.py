"""
EconoCausal Pricing Optimizer Module
Derives profit-maximizing and revenue-maximizing dynamic prices using causal elasticity
estimates subject to operational business guardrails and margin constraints.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class PricingOptimizer:
    """
    Microeconomically grounded dynamic pricing optimizer.
    Finds analytical or bounded optimal prices using causal demand elasticity theta(X).
    """

    def __init__(
        self,
        min_margin_pct: float = 0.15,
        max_price_change_pct: float = 0.30,
        absolute_price_floor: float = 40.0,
        absolute_price_ceiling: float = 140.0,
    ):
        self.min_margin_pct = min_margin_pct
        self.max_price_change_pct = max_price_change_pct
        self.absolute_price_floor = absolute_price_floor
        self.absolute_price_ceiling = absolute_price_ceiling

    def optimize_price_analytical(
        self,
        baseline_price: float,
        baseline_demand: float,
        causal_elasticity: float,
        unit_cost: float,
        objective: str = "profit",  # 'profit' or 'revenue'
    ) -> Dict[str, Any]:
        """
        Derives the optimal price for a single transaction or customer context.
        
        Profit maximization:
            max_P (P - c) * [Q_0 + theta * (P - P_0)]
            d(Profit)/dP = Q_0 + theta*(P - P_0) + (P - c)*theta = 0
            P* = (P_0 + c) / 2 - Q_0 / (2 * theta)
            
        Revenue maximization:
            max_P P * [Q_0 + theta * (P - P_0)]
            d(Revenue)/dP = Q_0 + theta*(P - P_0) + P*theta = 0
            P* = P_0 / 2 - Q_0 / (2 * theta)
        """
        theta = float(causal_elasticity)
        q0 = float(baseline_demand)
        p0 = float(baseline_price)
        cost = float(unit_cost)

        # Economic law safeguard: demand elasticity must be negative
        if theta >= -1e-4:
            theta = -0.50

        # Unconstrained optimal price
        if objective == "revenue":
            p_unconstrained = (p0 / 2.0) - (q0 / (2.0 * theta))
        else:  # 'profit'
            p_unconstrained = ((p0 + cost) / 2.0) - (q0 / (2.0 * theta))

        # Operational Guardrails & Business Constraints
        # 1. Cost floor with guaranteed minimum margin
        cost_floor = cost * (1.0 + self.min_margin_pct)
        # 2. Maximum allowable change from baseline
        lower_bound = max(
            self.absolute_price_floor,
            cost_floor,
            p0 * (1.0 - self.max_price_change_pct),
        )
        upper_bound = min(
            self.absolute_price_ceiling,
            p0 * (1.0 + self.max_price_change_pct),
        )

        p_optimal = float(np.clip(p_unconstrained, lower_bound, upper_bound))

        # Projected expected demand at optimal price
        expected_demand = max(0.0, q0 + theta * (p_optimal - p0))
        expected_revenue = p_optimal * expected_demand
        expected_profit = (p_optimal - cost) * expected_demand

        # Baseline comparison
        base_profit = (p0 - cost) * q0
        base_revenue = p0 * q0
        profit_uplift_pct = (
            ((expected_profit - base_profit) / max(1e-4, base_profit)) * 100.0
            if base_profit > 0
            else 0.0
        )

        return {
            "optimal_price": round(p_optimal, 2),
            "unconstrained_price": round(p_unconstrained, 2),
            "baseline_price": round(p0, 2),
            "price_delta": round(p_optimal - p0, 2),
            "price_delta_pct": round(((p_optimal - p0) / p0) * 100.0, 2),
            "expected_demand": round(expected_demand, 2),
            "baseline_demand": round(q0, 2),
            "expected_profit": round(expected_profit, 2),
            "baseline_profit": round(base_profit, 2),
            "profit_uplift_pct": round(profit_uplift_pct, 2),
            "expected_revenue": round(expected_revenue, 2),
            "unit_cost": round(cost, 2),
            "causal_elasticity": round(theta, 4),
            "bounds": (round(lower_bound, 2), round(upper_bound, 2)),
        }

    def batch_optimize(
        self,
        df: pd.DataFrame,
        causal_elasticities: np.ndarray,
        baseline_demands: np.ndarray,
        objective: str = "profit",
    ) -> pd.DataFrame:
        """
        Executes vector-accelerated price optimization across an entire test portfolio.
        """
        n = len(df)
        p0 = df["price"].values
        cost = df["unit_cost"].values
        theta = np.minimum(causal_elasticities, -0.05)
        q0 = baseline_demands

        if objective == "revenue":
            p_unconstrained = (p0 / 2.0) - (q0 / (2.0 * theta))
        else:
            p_unconstrained = ((p0 + cost) / 2.0) - (q0 / (2.0 * theta))

        # Enforce guardrails
        cost_floor = cost * (1.0 + self.min_margin_pct)
        lower_bound = np.maximum(
            self.absolute_price_floor,
            np.maximum(cost_floor, p0 * (1.0 - self.max_price_change_pct)),
        )
        upper_bound = np.minimum(
            self.absolute_price_ceiling,
            p0 * (1.0 + self.max_price_change_pct),
        )

        p_opt = np.clip(p_unconstrained, lower_bound, upper_bound)
        q_opt = np.maximum(0.0, q0 + theta * (p_opt - p0))
        rev_opt = p_opt * q_opt
        prof_opt = (p_opt - cost) * q_opt

        res_df = df.copy()
        res_df["opt_price"] = np.round(p_opt, 2)
        res_df["opt_demand"] = np.round(q_opt, 2)
        res_df["opt_revenue"] = np.round(rev_opt, 2)
        res_df["opt_profit"] = np.round(prof_opt, 2)
        res_df["opt_profit_gain"] = np.round(prof_opt - df["profit"].values, 2)

        return res_df
