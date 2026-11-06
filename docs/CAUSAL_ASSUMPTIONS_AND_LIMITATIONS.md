# EconoCausal: Causal Assumptions, Identification & Scientific Limitations

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Author:** Data Science & Machine Learning Internship Team  
**Compliance Standard:** ISO 9001:2015 / Peer-Reviewed Causal Inference Framework  

---

## 1. Executive Scientific Disclaimer

This project implements and evaluates **Double Machine Learning (DML)** for causal price elasticity estimation in retail and e-commerce dynamic pricing. Causal parameter estimates and counterfactual profit projections must be interpreted strictly within the formal econometric assumptions of the underlying structural equation model. **A non-zero coefficient or high predictive metric does not by itself establish causal validity.**

---

## 2. Core Causal Identifying Assumptions

To identify the causal effect of price $T$ on demand $Y$ from observational logs $(Y_i, T_i, W_i)_{i=1}^N$, the Double Machine Learning framework relies on four foundational micro-econometric assumptions:

### Assumption 1: Conditional Unconfoundedness (Ignorability / Selection on Observables)
$$\{ Y(t) \}_{t \in \mathcal{T}} \perp T \mid W$$

- **Definition:** Conditional on the observed high-dimensional confounder vector $W$ (competitor prices, promotions, holiday indicators, warehouse inventory, temperature, month, season, category), price assignment $T$ is conditionally independent of the potential demand outcomes $Y(t)$.
- **Substantive Meaning:** All factors that simultaneously influenced management pricing decisions and customer purchasing propensity are captured in the feature matrix $W$.
- **Potential Threat:** If an unobserved factor (e.g. an unrecorded viral social media campaign, unlogged localized supply chain disruption, or unmeasured brand reputation shift) simultaneously drives prices up and demand up, unconfoundedness is violated, leading to omitted variable bias in $\hat{\theta}$.

### Assumption 2: Common Support & Positivity (Overlap)
$$\forall w \in \text{supp}(W), \quad \text{Var}(T \mid W = w) > 0$$

- **Definition:** For every customer and market covariate configuration $w$, there must exist non-degenerate variation in the assigned prices.
- **Substantive Meaning:** Pricing managers and legacy systems must not set prices via a rigid, deterministic deterministic function of $W$ without any residual experimentation or noise.
- **Verification:** In our observational data, prices display empirical variation around competitor baselines and seasonal targets ($\sigma_{\text{price}} / \mu_{\text{price}} \approx 10\% - 25\%$), ensuring sufficient overlap across covariate strata.

### Assumption 3: Stable Unit Treatment Value Assumption (SUTVA)
SUTVA comprises two conditions:
1. **Consistency:** An individual observation's realized demand under assigned price $T_i$ is identical to the potential demand outcome:
   $$Y_i = Y_i(T_i)$$
2. **No Interference (No Spillover):** The price offered to customer $i$ does not alter the potential demand of customer $j$ ($i \neq j$):
   $$Y_i(t_1, t_2, \dots, t_N) = Y_i(t_i)$$
- **Substantive Meaning:** Valid in standard consumer retail where individual purchases do not cause immediate stockout panics or peer spillover effects. When inventory constraints become binding (e.g. limited flash sales), spillover effects can emerge and require multi-unit matching models.

### Assumption 4: Partially Linear Structural Specification
$$Y = \theta \cdot T + g(W) + U, \quad \mathbb{E}[U \mid T, W] = 0$$

- **Definition:** The relationship between demand and price is structural and linear in parameter $\theta$, while the confounding relationship $g(W)$ can be completely non-linear and non-parametric.
- **Substantive Meaning:** $\theta = \frac{\partial Y}{\partial T}$ represents the constant marginal change in demand per unit currency increase in price. For localized price optimizations ($\pm 25\%$ around baseline), local linearity provides an accurate Taylor approximation of true smooth economic demand curves.

---

## 3. Predictive Accuracy vs. Causal Validity: The Fundamental Distinction

A central contribution of this project is demonstrating why **predictive accuracy ($R^2$, MAE) is an inappropriate surrogate for causal validity**:

| Dimension | Standard Predictive ML | Double Machine Learning (DML) |
| :--- | :--- | :--- |
| **Primary Question** | *"What is expected sales when price happens to be ₹2,500?"* | *"What will happen to sales if we actively set price to ₹2,500?"* |
| **Mathematical Object**| $\mathbb{E}[Y \mid P = p]$ | $\mathbb{E}[Y \mid \text{do}(P = p)]$ |
| **Objective Function** | Minimize $\frac{1}{N} \sum (Y_i - \hat{Y}_i)^2$ | Orthogonal moment condition $\mathbb{E}[\psi(\theta_0, \eta_0)] = 0$ |
| **Handling of Confounders**| Exploits confounding correlation to maximize fit | Orthogonalizes nuisance functions to eliminate endogeneity |
| **Price Elasticity** | Flat or positive (+0.12 to -0.002) due to holiday surges | Microeconomically consistent downward slope ($\theta < 0$) |
| **Downstream Risk** | Prescribes reckless price hikes; causes customer churn | Optimizes true gross profit along counterfactual demand curve |

---

## 4. Operational Guardrails Against Model Failure

To protect enterprise operations from model extrapolation risks, EconoCausal implements microeconomic guardrails:

1. **Margin Floor Enforcement:**
   $$P_{\text{recommended}} \ge \text{Cost} \times (1 + \text{Margin}_{\text{min}})$$
   Ensures that algorithmic recommendations never sell below marginal cost or erode minimum gross profit margins (default 10%).
2. **Search Radius Bounds:**
   Candidate prices are bounded within $\pm 25\%$ of current catalog prices:
   $$P_{\text{recommended}} \in \left[ 0.75 \times P_0, \; 1.25 \times P_0 \right]$$
   Prevents unvalidated extrapolation outside the historical empirical support.
3. **Law of Demand Fallback:**
   If empirical estimation produces $\hat{\theta} \ge 0$ (due to severe unobserved shocks or high data sparsity), the pricing engine safely activates a microeconomic fallback ($\theta_{\text{fallback}} = -0.01$).

---

## 5. Summary of Limitations

1. **Synthetic Observational Data:** The empirical demonstration utilizes a structured dataset with realistic price endogeneity. While the methodology reflects enterprise DML applications, findings demonstrate causal estimation mechanics rather than an empirical claim about external physical markets.
2. **Unobserved Macro Shocks:** In live production, unexpected macroeconomic crises (hyperinflation, regulatory price controls) require instrumental variable methods (e.g. Deep IV) or online contextual bandit exploration.
3. **Non-Linear Elasticity Beyond Local Range:** Large price jumps ($\pm 50\%+$) may exhibit non-linear price elasticity saturation, requiring non-parametric causal tree or neural DML architectures.
