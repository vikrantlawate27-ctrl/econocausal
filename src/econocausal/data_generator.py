"""
EconoCausal Data Generator Module
Simulates realistic e-commerce and retail transaction datasets with price endogeneity,
confounders (demand shifters), and customer segment heterogeneity.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    n_samples: int = 25000
    random_state: int = 42
    base_cost: float = 45.0
    base_price: float = 75.0
    true_ate_elasticity: float = -1.85


class DynamicPricingDataGenerator:
    """
    Generates synthetic dynamic pricing observational data exhibiting:
    1. Endogenous pricing: Prices are determined by confounders (seasonality, competitor prices, inventory, marketing).
    2. Heterogeneous price elasticity (CATE): Elasticity varies by customer loyalty tier and purchase urgency.
    3. Structural demand with known ground-truth causal parameters for empirical validation.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        np.random.seed(self.config.random_state)

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Generates full transaction dataset with observed features, treatment (price),
        outcome (demand/sales), and true causal ground truth parameters.
        """
        n = self.config.n_samples
        rng = np.random.default_rng(self.config.random_state)

        # -------------------------------------------------------------
        # 1. Confounders W (Exogenous demand & price shifters)
        # -------------------------------------------------------------
        # Seasonality index [-1 to 1] (e.g. holiday peaks vs. off-peak)
        t = np.linspace(0, 4 * np.pi, n)
        seasonality = 0.8 * np.sin(t) + 0.3 * np.cos(2 * t) + rng.normal(0, 0.15, n)

        # Competitor pricing ($50 - $120)
        competitor_price = 70.0 + 15.0 * seasonality + rng.normal(0, 8.0, n)
        competitor_price = np.clip(competitor_price, 40.0, 140.0)

        # Inventory level (units in stock, 10 to 500)
        inventory_level = 250 - 60 * seasonality + rng.normal(0, 40, n)
        inventory_level = np.clip(inventory_level, 10, 500)

        # Marketing spend ($k/day)
        marketing_spend = 15.0 + 8.0 * seasonality + rng.exponential(4.0, n)

        # Day of week (0=Mon ... 6=Sun)
        day_of_week = rng.integers(0, 7, n)
        is_weekend = (day_of_week >= 5).astype(float)

        # Store rating (3.0 to 5.0)
        store_rating = np.clip(rng.normal(4.3, 0.4, n), 3.0, 5.0)

        # -------------------------------------------------------------
        # 2. Heterogeneity Features X (Customer state & segments)
        # -------------------------------------------------------------
        # Customer loyalty tier: 0=Bronze (sensitive), 1=Silver, 2=Gold, 3=Platinum (inelastic)
        tier_probs = [0.40, 0.30, 0.20, 0.10]
        customer_tier = rng.choice([0, 1, 2, 3], size=n, p=tier_probs)

        # Urgency score [0 to 1]: 0 = casual browser, 1 = immediate need
        urgency_score = np.clip(rng.beta(2, 3, n) + 0.15 * is_weekend, 0.05, 0.98)

        # Device type: Mobile (0), Desktop (1)
        device_type = rng.choice([0, 1], size=n, p=[0.65, 0.35])

        # -------------------------------------------------------------
        # 3. Ground Truth Heterogeneous Causal Price Sensitivity theta(X)
        # -------------------------------------------------------------
        # Base sensitivity theta_0 = -2.6 (Bronze) up to -0.9 (Platinum)
        # Higher urgency makes demand less elastic (closer to 0)
        true_theta = (
            -2.60
            + 0.50 * customer_tier
            + 0.65 * urgency_score
            - 0.15 * device_type
        )
        # Mean true causal treatment effect (ATE)
        true_ate = float(np.mean(true_theta))

        # -------------------------------------------------------------
        # 4. Observational Treatment Assignment (Endogenous Price Setting)
        # -------------------------------------------------------------
        # Business pricing rule in history: managers raised prices during high demand
        # (peak season, high marketing, low inventory, high competitor price).
        price_confounding = (
            self.config.base_price
            + 0.30 * (competitor_price - 70.0)
            + 11.5 * seasonality
            - 0.035 * (inventory_level - 250)
            + 0.45 * (marketing_spend - 15.0)
            + 3.2 * store_rating
            + 2.5 * customer_tier  # Segment-targeted pricing
        )
        # Add historical policy exploration noise (manager discretion / test bands)
        policy_noise = rng.normal(0, 4.5, n)
        price = np.clip(price_confounding + policy_noise, 35.0, 150.0)

        # Marginal Cost
        unit_cost = self.config.base_cost + 0.05 * (competitor_price - 70.0) + rng.normal(0, 1.5, n)
        unit_cost = np.clip(unit_cost, 25.0, 65.0)

        # -------------------------------------------------------------
        # 5. Structural Demand Generation (Outcome Y)
        # -------------------------------------------------------------
        # Base demand influenced by confounders W
        baseline_demand_shift = (
            85.0
            + 18.0 * seasonality
            + 0.42 * (competitor_price - 70.0)  # Cross-price elasticity: competitor higher -> our demand higher
            + 0.85 * marketing_spend
            + 5.0 * is_weekend
            + 7.5 * (store_rating - 4.0)
            + 4.0 * customer_tier
        )

        # Causal price effect on demand: theta(X) * (price - base_price)
        causal_price_effect = true_theta * (price - self.config.base_price)

        # Unobserved demand shock epsilon
        demand_noise = rng.normal(0, 6.0, n)

        # Observed Demand (units sold)
        demand = baseline_demand_shift + causal_price_effect + demand_noise
        demand = np.maximum(0.0, np.round(demand, 2))

        # Revenue and Gross Profit
        revenue = np.round(demand * price, 2)
        profit = np.round(demand * (price - unit_cost), 2)

        # -------------------------------------------------------------
        # 6. Assemble DataFrame
        # -------------------------------------------------------------
        df = pd.DataFrame({
            # Confounders W
            "seasonality": np.round(seasonality, 4),
            "competitor_price": np.round(competitor_price, 2),
            "inventory_level": np.round(inventory_level, 1),
            "marketing_spend": np.round(marketing_spend, 2),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "store_rating": np.round(store_rating, 2),
            # Moderators X
            "customer_tier": customer_tier,
            "urgency_score": np.round(urgency_score, 4),
            "device_type": device_type,
            # Treatment T
            "price": np.round(price, 2),
            "unit_cost": np.round(unit_cost, 2),
            # Outcome Y
            "demand": demand,
            "revenue": revenue,
            "profit": profit,
            # Ground Truth for Benchmarking
            "true_theta": np.round(true_theta, 4),
        })

        metadata = {
            "n_samples": n,
            "true_ate": true_ate,
            "true_theta_min": float(np.min(true_theta)),
            "true_theta_max": float(np.max(true_theta)),
            "base_cost": self.config.base_cost,
            "base_price": self.config.base_price,
            "confounders": ["seasonality", "competitor_price", "inventory_level", "marketing_spend", "day_of_week", "is_weekend", "store_rating"],
            "moderators": ["customer_tier", "urgency_score", "device_type"],
            "treatment": "price",
            "outcome": "demand",
        }

        return df, metadata


if __name__ == "__main__":
    generator = DynamicPricingDataGenerator()
    df, meta = generator.generate()
    print(f"Generated {len(df)} transactions.")
    print(f"True Average Treatment Effect (Elasticity ATE): {meta['true_ate']:.4f}")
    print(df.head())
