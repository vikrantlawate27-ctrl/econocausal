"""
EconoCausal Counterfactual Policy Evaluator Module
Simulates and evaluates out-of-sample business revenue and profit under counterfactual pricing policies:
1. Historical Human / Rule-based Policy
2. Naive OLS Dynamic Pricing Policy
3. EconoCausal Double Machine Learning Policy
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class CounterfactualEvaluator:
    """
    Evaluates counterfactual business outcomes (Revenue, Profit, Margin, Uplift)
    under alternative pricing policies against true structural demand.
    """

    def __init__(self, test_df: pd.DataFrame, output_dir: str = "outputs/figures"):
        self.df = test_df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_policies(
        self,
        dml_optimal_prices: np.ndarray,
        naive_optimal_prices: np.ndarray,
        fixed_price: float = 75.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Simulates ground-truth counterfactual demand for each policy:
            Q_sim = max(0, Q_base + theta_true * (P_policy - P_base))
        """
        p_base = self.df["price"].values
        q_base = self.df["demand"].values
        cost = self.df["unit_cost"].values
        true_theta = self.df["true_theta"].values

        # 1. Historical Baseline
        q_hist = q_base
        rev_hist = p_base * q_hist
        prof_hist = (p_base - cost) * q_hist

        # 2. Fixed Constant Price Policy
        p_fix = np.full_like(p_base, fixed_price)
        q_fix = np.maximum(0.0, q_base + true_theta * (p_fix - p_base))
        rev_fix = p_fix * q_fix
        prof_fix = (p_fix - cost) * q_fix

        # 3. Naive ML Policy (biased by endogeneity)
        p_naive = np.clip(naive_optimal_prices, 40.0, 140.0)
        q_naive = np.maximum(0.0, q_base + true_theta * (p_naive - p_base))
        rev_naive = p_naive * q_naive
        prof_naive = (p_naive - cost) * q_naive

        # 4. EconoCausal DML Policy
        p_dml = np.clip(dml_optimal_prices, 40.0, 140.0)
        q_dml = np.maximum(0.0, q_base + true_theta * (p_dml - p_base))
        rev_dml = p_dml * q_dml
        prof_dml = (p_dml - cost) * q_dml

        # Policy Metrics Summary
        policies = {
            "Historical Baseline": {
                "avg_price": float(np.mean(p_base)),
                "total_demand": float(np.sum(q_hist)),
                "total_revenue": float(np.sum(rev_hist)),
                "total_profit": float(np.sum(prof_hist)),
                "profit_margin_pct": float((np.sum(prof_hist) / np.sum(rev_hist)) * 100.0),
                "profit_uplift_pct": 0.0,
            },
            "Fixed Price ($75)": {
                "avg_price": float(np.mean(p_fix)),
                "total_demand": float(np.sum(q_fix)),
                "total_revenue": float(np.sum(rev_fix)),
                "total_profit": float(np.sum(prof_fix)),
                "profit_margin_pct": float((np.sum(prof_fix) / np.sum(rev_fix)) * 100.0),
                "profit_uplift_pct": float(((np.sum(prof_fix) - np.sum(prof_hist)) / np.sum(prof_hist)) * 100.0),
            },
            "Naive ML Pricing": {
                "avg_price": float(np.mean(p_naive)),
                "total_demand": float(np.sum(q_naive)),
                "total_revenue": float(np.sum(rev_naive)),
                "total_profit": float(np.sum(prof_naive)),
                "profit_margin_pct": float((np.sum(prof_naive) / np.sum(rev_naive)) * 100.0),
                "profit_uplift_pct": float(((np.sum(prof_naive) - np.sum(prof_hist)) / np.sum(prof_hist)) * 100.0),
            },
            "EconoCausal (DML)": {
                "avg_price": float(np.mean(p_dml)),
                "total_demand": float(np.sum(q_dml)),
                "total_revenue": float(np.sum(rev_dml)),
                "total_profit": float(np.sum(prof_dml)),
                "profit_margin_pct": float((np.sum(prof_dml) / np.sum(rev_dml)) * 100.0),
                "profit_uplift_pct": float(((np.sum(prof_dml) - np.sum(prof_hist)) / np.sum(prof_hist)) * 100.0),
            },
        }

        comparison_df = pd.DataFrame(policies).T
        comparison_df.index.name = "Policy"

        # Store simulation columns
        sim_df = self.df.copy()
        sim_df["price_hist"] = p_base
        sim_df["profit_hist"] = prof_hist
        sim_df["price_naive"] = p_naive
        sim_df["profit_naive"] = prof_naive
        sim_df["price_dml"] = p_dml
        sim_df["profit_dml"] = prof_dml

        return comparison_df, sim_df

    def plot_counterfactual_comparison(self, comparison_df: pd.DataFrame) -> str:
        """Creates bar charts comparing profit, revenue, and uplift across policies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Gross Profit Comparison
        palette = ["#78909C", "#FFA726", "#EF5350", "#2E7D32"]
        bars = axes[0].bar(
            comparison_df.index,
            comparison_df["total_profit"] / 1000.0,
            color=palette,
            width=0.55,
            edgecolor="#333",
            linewidth=1,
        )
        axes[0].set_title("Total Portfolio Gross Profit ($k)\n(Across Test Horizon)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Profit ($ Thousands)")
        axes[0].tick_params(axis="x", rotation=15)

        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(
                f"${height:,.1f}k",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Profit Uplift Percentage
        uplift_palette = ["#9E9E9E", "#FFA726", "#EF5350", "#2E7D32"]
        bars2 = axes[1].bar(
            comparison_df.index,
            comparison_df["profit_uplift_pct"],
            color=uplift_palette,
            width=0.55,
            edgecolor="#333",
            linewidth=1,
        )
        axes[1].set_title("Gross Profit Uplift vs. Historical Baseline (%)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Uplift (%)")
        axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1].tick_params(axis="x", rotation=15)

        for bar in bars2:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:+.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontweight="bold",
            )

        fig.tight_layout()
        path = os.path.join(self.output_dir, "policy_profit_comparison.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
