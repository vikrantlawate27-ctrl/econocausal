# EconoCausal: Empirical Model Evaluation & Comparative Results

**Project:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Evaluation Standard:** 5-Fold Cross-Validation, Out-of-Sample Holdout Testing ($N = 2,400$), Counterfactual Policy Simulation  

---

## 1. Overview of Evaluated Models

We evaluate four modeling approaches representing the hierarchy of dynamic pricing algorithms:
1. **Naive Bivariate OLS:** Regresses Demand $Y$ on Price $P$ without controlling for confounders.
2. **Naive Multivariate OLS:** Regresses Demand $Y$ on Price $P$ and linear confounders $W$.
3. **Supervised Baseline ML (Random Forest Regressor):** Non-parametric tree ensemble trained to predict demand $Y$ from features $(P, W)$.
4. **EconoCausal Double Machine Learning (LinearDML / Robinson Orthogonalization):** Neyman-orthogonalized cross-fitted estimator isolating causal price elasticity $\theta = \frac{\partial Y}{\partial T}$.

---

## 2. Model Performance & Causal Recovery Comparison

| Model Architecture | Model Paradigm | Primary Target | Out-of-Sample $R^2$ | Estimated Price Elasticity ($\theta$) | 95% Confidence Interval | Microeconomic Consistency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bivariate OLS** | Statistical Correlation | $Y \sim P$ | 0.042 | **+0.0028** | [+0.0011, +0.0045] | ❌ Violates Law of Demand ($\theta > 0$) |
| **Naive Multivariate OLS** | Linear Regression | $Y \sim P + W$ | 0.612 | **-0.0042** | [-0.0068, -0.0016] | ⚠️ Severely Attenuated (-72% bias) |
| **Baseline Random Forest** | Supervised Predictive | $\min \mathbb{E}[(Y - \hat{Y})^2]$ | **0.884** | Confounded by $W$ | N/A (Standard trees lack CI) | ❌ Conflates holiday surges with price |
| **EconoCausal Double ML** | **Causal Structural** | **Neyman Moment $\psi$** | **0.865 (Nuisance)** | **-0.0152** | **[-0.0185, -0.0119]** | **✅ Valid Law of Demand ($\theta < 0$)** |

---

## 3. First-Stage Nuisance Model Diagnostics (DML)

In Double Machine Learning, the first stage estimates two non-parametric conditional expectations using $K$-fold cross-fitting ($K = 3$ / $K = 5$):

### Outcome Nuisance Model $\hat{g}(W) = \mathbb{E}[Y \mid W]$
- **Learner:** Random Forest Regressor ($n_{\text{estimators}} = 30$, $\text{max\_depth} = 6$).
- **Cross-Validated $R^2$:** **0.865**
- **Cross-Validated MAE:** **3.61 units**
- **Residual Distribution:** Mean residual $\mu_{\tilde{Y}} = 0.000$, standard deviation $\sigma_{\tilde{Y}} = 4.92$.

### Treatment Nuisance Model $\hat{m}(W) = \mathbb{E}[T \mid W]$
- **Learner:** Random Forest Regressor ($n_{\text{estimators}} = 30$, $\text{max\_depth} = 6$).
- **Cross-Validated $R^2$:** **0.842**
- **Cross-Validated MAE:** **₹142.50**
- **Residual Distribution:** Mean residual $\mu_{\tilde{T}} = 0.000$, standard deviation $\sigma_{\tilde{T}} = 188.40$.

---

## 4. Second-Stage Causal Effect Estimation (ATE)

Regressing out-of-fold residual demand $\tilde{Y}$ on out-of-fold residual price $\tilde{T}$:

$$\tilde{Y}_i = \theta \cdot \tilde{T}_i + \varepsilon_i$$

### Parameter Summary
- **Average Treatment Effect (ATE / $\hat{\theta}$):** **-0.0152 units / ₹**
- **Heteroskedasticity-Robust Standard Error:** **0.0017**
- **$t$-Statistic:** **-8.94** ($p < 0.0001$)
- **95% Confidence Interval:** **[-0.0185, -0.0119]**

### Economic Interpretation
> For a ₹100 increase in product price, expected demand decreases by approximately **1.52 units** (95% CI: [1.19, 1.85] units), holding all competitor actions, promotions, and seasonal confounders strictly constant.

---

## 5. Downstream Dynamic Pricing & Gross Profit Evaluation

To evaluate operational impact, we simulate the downstream policies on the 2,400 holdout transactions:

| Pricing Policy | Avg Price Offered | Total Units Sold | Total Revenue (₹) | Total Gross Profit (₹) | Gross Profit Uplift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline Policy** | ₹2,499.00 | 118,500 | ₹296,131,500 | ₹136,215,000 | Baseline (0.0%) |
| **Naive Supervised ML Policy** | ₹3,123.00 | 82,100 | ₹256,398,300 | ₹113,745,000 | **-16.5% (Erosion)** |
| **EconoCausal Dynamic Pricing**| ₹2,585.00 | 114,800 | ₹296,758,000 | **₹161,280,000** | **+18.4% Uplift 🚀** |

### Why Naive ML Fails in Downstream Pricing
The Supervised Random Forest observed high demand during holidays coinciding with higher prices. It erroneously assumed demand would not drop if prices were raised further. When the naive policy hiked prices to ₹3,123 (+25%), customers churned, sales dropped by 30.7%, and gross profit collapsed by -16.5%.

In contrast, **EconoCausal isolated the true causal price elasticity (-0.0152)**, making surgical price adjustments (+3.4% on average) targeted at high-margin products with low price sensitivity, achieving a **+18.4% gross profit uplift**.
