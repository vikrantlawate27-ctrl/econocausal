"""
EconoCausal Exploratory Data Analysis (EDA) Module
Analyzes transactional distributions, confounding variables, and demonstrates
the classic price endogeneity paradox (omitted variable bias).
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class PricingEDA:
    """
    Performs comprehensive Exploratory Data Analysis on pricing transactions,
    highlighting endogeneity, demand confounding, and segment elasticity heterogeneity.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs/figures"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Configure seaborn style
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    def run_summary(self) -> Dict[str, Any]:
        """Calculates core statistical and confounding metrics."""
        summary = {
            "total_records": len(self.df),
            "missing_values": int(self.df.isnull().sum().sum()),
            "mean_price": float(self.df["price"].mean()),
            "std_price": float(self.df["price"].std()),
            "mean_demand": float(self.df["demand"].mean()),
            "std_demand": float(self.df["demand"].std()),
            "mean_revenue": float(self.df["revenue"].mean()),
            "mean_profit": float(self.df["profit"].mean()),
            # Raw observational correlation between price and demand
            "raw_price_demand_corr": float(self.df["price"].corr(self.df["demand"])),
            # Confounder correlations with price
            "competitor_price_corr": float(self.df["price"].corr(self.df["competitor_price"])),
            "seasonality_corr": float(self.df["price"].corr(self.df["seasonality"])),
            "marketing_spend_corr": float(self.df["price"].corr(self.df["marketing_spend"])),
        }
        return summary

    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generates publication-quality charts for the project report and deliverables."""
        saved_paths = {}

        # 1. Price vs. Demand: Raw Confounded View vs. Stratified View
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Subplot A: Raw Observational (shows apparent flat or positive trend due to confounding)
        sns.regplot(
            data=self.df.sample(min(2500, len(self.df)), random_state=42),
            x="price",
            y="demand",
            scatter_kws={"alpha": 0.25, "color": "#4A90E2", "s": 18},
            line_kws={"color": "#D0021B", "linewidth": 2.5, "label": f"Naive Fit (r={self.df['price'].corr(self.df['demand']):.2f})"},
            ax=axes[0],
        )
        axes[0].set_title("A: Naive Observational Price vs. Demand\n(Severely Biased by Seasonality & Marketing)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Price ($)")
        axes[0].set_ylabel("Demand (Units)")
        axes[0].legend(loc="upper right")

        # Subplot B: Stratified by Customer Tier (controlling for heterogeneity)
        tiers = {0: "Bronze (High Sensitivity)", 1: "Silver", 2: "Gold", 3: "Platinum (Low Sensitivity)"}
        df_sample = self.df.sample(min(3000, len(self.df)), random_state=42).copy()
        df_sample["tier_label"] = df_sample["customer_tier"].map(tiers)

        palette = ["#E65100", "#FB8C00", "#1E88E5", "#0D47A1"]
        sns.scatterplot(
            data=df_sample,
            x="price",
            y="demand",
            hue="tier_label",
            alpha=0.35,
            s=22,
            palette=palette,
            ax=axes[1],
        )
        axes[1].set_title("B: Demand Distribution by Customer Loyalty Tier\n(Heterogeneous Price Responsiveness)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Price ($)")
        axes[1].set_ylabel("Demand (Units)")
        axes[1].legend(title="Tier Segment", loc="upper right")

        fig.tight_layout()
        path1 = os.path.join(self.output_dir, "eda_price_demand_confounding.png")
        fig.savefig(path1, dpi=300)
        plt.close(fig)
        saved_paths["price_demand_confounding"] = path1

        # 2. Confounder Correlation Matrix
        fig, ax = plt.subplots(figsize=(9, 7))
        corr_cols = [
            "price", "demand", "seasonality", "competitor_price",
            "inventory_level", "marketing_spend", "customer_tier", "urgency_score"
        ]
        corr_matrix = self.df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            linewidths=0.7,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Confounder & Feature Correlation Matrix\n(Illustrating Endogeneity Drivers)", fontsize=12, fontweight="bold")
        path2 = os.path.join(self.output_dir, "eda_correlation_matrix.png")
        fig.savefig(path2, dpi=300)
        plt.close(fig)
        saved_paths["correlation_matrix"] = path2

        # 3. Revenue & Profit Surfaces
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(self.df["price"], bins=40, kde=True, color="#2b5c8f", ax=axes[0])
        axes[0].set_title("Observed Price Distribution ($)", fontweight="bold")
        axes[0].set_xlabel("Price ($)")

        sns.histplot(self.df["profit"], bins=40, kde=True, color="#2e7d32", ax=axes[1])
        axes[1].set_title("Observed Transaction Profit Distribution ($)", fontweight="bold")
        axes[1].set_xlabel("Gross Profit ($)")

        fig.tight_layout()
        path3 = os.path.join(self.output_dir, "eda_distributions.png")
        fig.savefig(path3, dpi=300)
        plt.close(fig)
        saved_paths["distributions"] = path3

        return saved_paths


if __name__ == "__main__":
    try:
        from econocausal.data_generator import DynamicPricingDataGenerator
    except ImportError:
        from data_generator import DynamicPricingDataGenerator
    df, meta = DynamicPricingDataGenerator().generate()
    eda = PricingEDA(df)
    summary = eda.run_summary()
    print("EDA Summary:", summary)
    charts = eda.generate_all_visualizations()
    print("Generated charts:", charts)
