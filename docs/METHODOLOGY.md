# EconoCausal: Double Machine Learning & Dynamic Pricing Methodology

**Project Name:** EconoCausal — Dynamic Pricing via Double Machine Learning  
**Domain:** Causal Inference, Microeconomics, and Applied Machine Learning  
**Compliance Standard:** Chernozhukov et al. (2018), Robinson (1988)  

---

## 1. The Microeconomic Pricing Problem

In enterprise dynamic pricing, a seller seeks to establish an optimal price $P^*$ for each product or transaction context that maximizes expected gross profit subject to marginal cost $c$:

$$\max_{P} \Pi(P) = (P - c) \cdot Q(P)$$

Alternatively, under market-share acquisition objectives, the objective function may target total revenue:

$$\max_{P} R(P) = P \cdot Q(P)$$

The fundamental unknown is the **structural causal demand function** $Q(P)$:
$$Q(P) = \mathbb{E}[Y \mid \text{do}(P = p), W]$$

Where:
- $Y$: Demand (quantity of units sold).
- $P$: Price offered to the customer (treatment).
- $W$: Observed context and confounders (competitor price, inventory, promotions, holidays, seasonality).
- $\text{do}(P = p)$: Pearl's causal intervention operator, representing the act of actively setting price rather than passively observing it.

---

## 2. The Endogeneity Paradox: Why Standard Supervised ML Fails

In historical sales transaction logs, pricing decisions are not assigned uniformly at random. Pricing managers and legacy algorithmic heuristics systematically adjust prices based on market conditions:
1. **Surge / Peak Periods:** Prices are increased during holidays, festivals, and high-demand seasons.
2. **Promotional Clearances:** Prices are cut when inventory is high or demand is sluggish.
3. **Competitor Following:** Prices track competitor price increases.

Mathematically, price $T$ is correlated with demand shifters $W$ that directly affect sales volume $Y$:
$$\text{Cov}(T, W) \neq 0 \quad \text{and} \quad \frac{\partial Y}{\partial W} \neq 0$$

### The Failure of Conditional Expectation
Standard machine learning models (e.g. Ridge Regression, XGBoost, Random Forests, Deep Neural Networks) estimate the conditional expectation:
$$\hat{Y}_{\text{ML}} = \mathbb{E}[Y \mid T, W]$$

Because high prices coincide with high demand during holiday surges, standard ML observes $T \uparrow$ and $Y \uparrow$ simultaneously. As a consequence:
- The model estimates an **attenuated or positive price slope**: $\frac{\partial \hat{Y}}{\partial T} \ge 0$.
- This violates the fundamental microeconomic **Law of Demand**:
  $$\frac{\partial Q}{\partial P} < 0$$
- If downstream profit optimization uses this naive model:
  $$\max_{P} (P - c) \cdot \hat{Y}_{\text{ML}}(P)$$
  The algorithm recommends continuous price hikes, falsely predicting that demand will remain robust. In practice, this leads to catastrophic customer churn and collapsed enterprise revenue.

---

## 3. The Double Machine Learning (DML) Formulation

To overcome endogeneity and regularization bias, EconoCausal implements **Double/Debiased Machine Learning (DML)** introduced by Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins (2018).

### 3.1 Partially Linear Model (Robinson, 1988)
We model observed demand through a partially linear structural equation:

$$Y_i = \theta \cdot T_i + g(W_i) + U_i, \quad \mathbb{E}[U_i \mid T_i, W_i] = 0$$
$$T_i = m(W_i) + V_i, \quad \mathbb{E}[V_i \mid W_i] = 0$$

Where:
- $Y_i \in \mathbb{R}$: Observed demand (units sold).
- $T_i \in \mathbb{R}$: Continuous price treatment.
- $W_i \in \mathbb{R}^d$: Confounders (competitor price, promotions, holiday indicators, inventory, temperature, month, season, category).
- $g(W_i) = \mathbb{E}[Y_i \mid W_i]$: Non-parametric outcome nuisance function.
- $m(W_i) = \mathbb{E}[T_i \mid W_i]$: Non-parametric treatment nuisance function.
- $\theta$: The parameter of interest—**Average Treatment Effect (ATE)** or causal price elasticity $\frac{\partial Y}{\partial T}$.

---

## 4. Neyman Orthogonality

A naive plug-in approach estimates $\hat{g}(W)$ using machine learning, plugs it into $Y - \hat{g}(W) = \theta T$, and runs regression. However, non-parametric ML estimators introduce **regularization bias** (from tree depth constraints, shrinkage, or feature selection) of order $n^{-1/4}$ or slower, which contaminates $\hat{\theta}$ with first-order bias:

$$\sqrt{n}(\hat{\theta}_{\text{naive}} - \theta_0) \to \infty$$

### The Orthogonal Score
Chernozhukov et al. construct an estimating equation using a Neyman-orthogonal score:

$$\psi(Y, T, W; \theta, \eta) = \Big( (Y - g(W)) - \theta (T - m(W)) \Big) \cdot (T - m(W))$$

Where $\eta = (g, m)$ represents the nuisance parameters. The score satisfies the **Neyman Orthogonality Condition**:

$$\left. \frac{\partial}{\partial \eta} \mathbb{E}\left[ \psi(Y, T, W; \theta_0, \eta) \right] \right|_{\eta = \eta_0} = 0$$

Because the Gateaux derivative with respect to first-stage nuisance estimation errors vanishes at the true parameter values, first-stage estimation errors do not contaminate the second-stage causal estimator.

---

## 5. Robinson's Residual-on-Residual Regression & $K$-Fold Cross-Fitting

### Step 1: $K$-Fold Sample Partitioning
To prevent overfitting in the nuisance models from inducing correlation between residual errors, we use $K$-fold cross-fitting ($K = 3$ or $5$).
The dataset $\mathcal{D}$ is partitioned into $K$ disjoint subsets $\{I_k\}_{k=1}^K$.

### Step 2: Nuisance Model Training
For each fold $k \in \{1, \dots, K\}$:
1. Train the outcome nuisance model $\hat{g}^{(-k)}$ on $\mathcal{D} \setminus I_k$ to predict $Y$ from $W$.
2. Train the treatment nuisance model $\hat{m}^{(-k)}$ on $\mathcal{D} \setminus I_k$ to predict $T$ from $W$.
3. Compute out-of-fold residuals strictly on holdout partition $I_k$:
   $$\tilde{Y}_i = Y_i - \hat{g}^{(-k)}(W_i), \quad \forall i \in I_k$$
   $$\tilde{T}_i = T_i - \hat{m}^{(-k)}(W_i), \quad \forall i \in I_k$$

### Step 3: Second-Stage Orthogonal Regression
Pool the out-of-fold residuals across all $K$ folds and fit the residualized linear model:

$$\tilde{Y}_i = \theta \cdot \tilde{T}_i + \varepsilon_i$$

Using OLS / Ridge on residuals:
$$\hat{\theta} = \frac{\sum_{i=1}^N \tilde{T}_i \tilde{Y}_i}{\sum_{i=1}^N \tilde{T}_i^2}$$

### Step 4: Asymptotic Normality and Confidence Intervals
Under cross-fitting and Neyman orthogonality, $\hat{\theta}$ obeys $\sqrt{N}$-rate asymptotic normality:

$$\sqrt{N}(\hat{\theta} - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, \frac{\mathbb{E}[\tilde{T}^2 \varepsilon^2]}{(\mathbb{E}[\tilde{T}^2])^2}\right)$$

The heteroskedasticity-robust standard error is:
$$\widehat{\text{SE}}(\hat{\theta}) = \frac{\sqrt{\frac{1}{N} \sum_{i=1}^N \tilde{T}_i^2 \hat{\varepsilon}_i^2}}{\sum_{i=1}^N \tilde{T}_i^2}$$

And the 95% Wald confidence interval is:
$$\text{CI}_{0.95} = \left[ \hat{\theta} - 1.96 \cdot \widehat{\text{SE}}(\hat{\theta}), \; \hat{\theta} + 1.96 \cdot \widehat{\text{SE}}(\hat{\theta}) \right]$$

---

## 6. Microeconomic Optimization & Dynamic Pricing Engine

Once the causal elasticity $\theta$ is estimated, demand projection for any candidate price $P$ around catalog baseline $P_0$ with baseline demand $Q_0$ is:

$$Q(P) = \max\Big(1.0, \; Q_0 + \theta \cdot (P - P_0)\Big)$$

### Analytical Closed-Form Unconstrained Solution
Profit is:
$$\Pi(P) = (P - c) \cdot \left[ Q_0 + \theta(P - P_0) \right]$$

Setting the first-order condition $\frac{d\Pi}{dP} = 0$:
$$Q_0 + \theta(P - P_0) + \theta(P - c) = 0$$
$$2\theta P = \theta P_0 + \theta c - Q_0$$
$$P^* = \frac{P_0 + c}{2} - \frac{Q_0}{2\theta}$$

### Operational Business Guardrails
To prevent unbounded price volatility and protect enterprise brand equity, the dynamic pricing engine applies configurable constraints:
1. **Search Radius:** Candidate prices evaluate within $P_0 \times (1 \pm \delta_{\text{radius}})$, typically $\pm 25\%$.
2. **Margin Floor:** $P \ge c \times (1 + \text{margin}_{\text{min}})$, guaranteeing minimum 10% gross margin.
3. **Discretization Grid:** Evaluated in standard commercial price increments ($\Delta P = \max(5, \text{round}(0.05 \cdot P_0))$).
4. **Law of Demand Fallback:** If empirical estimation yields $\theta \ge 0$ due to unobserved shocks, the engine enforces a conservative microeconomic fallback ($\theta_{\text{fallback}} = -0.01$).

---

## 7. Counterfactual Policy Evaluation

To evaluate pricing policies before production deployment, we perform counterfactual evaluation on holdout test data:
1. **Historical Baseline Policy:** Performance under actual historical observed prices $P_{\text{hist}}$ and demands $Y_{\text{hist}}$.
2. **Naive ML Policy:** Prices chosen using standard Random Forest demand predictions without orthogonalization.
3. **EconoCausal DML Policy:** Prices chosen using Neyman-orthogonalized causal elasticity.

Gross profit for policy $\pi$ over $M$ evaluation transactions is:
$$\Pi_{\text{total}}(\pi) = \sum_{j=1}^M \left( P_j^{(\pi)} - c_j \right) \cdot Q_j\left(P_j^{(\pi)}\right)$$

Uplift relative to historical baseline is computed as:
$$\text{Uplift}_{\Pi} = \frac{\Pi_{\text{total}}(\pi) - \Pi_{\text{total}}(\text{historical})}{\Pi_{\text{total}}(\text{historical})} \times 100\%$$
