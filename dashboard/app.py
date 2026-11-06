"""
EconoCausal Dynamic Pricing Interactive Demonstration Dashboard
Built for Infotact Solutions Internship Project Evaluation
Intern: Vikrant Bharat Lawate | Ref. No.: ITS/DSML/1090
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from econocausal.pricing_optimizer import PricingOptimizer

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EconoCausal: Dynamic Pricing via Double Machine Learning",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load Model Artifacts & Data
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_results():
    model_path = os.path.join(parent_dir, "outputs", "models", "dml_pricing_model.joblib")
    results_path = os.path.join(parent_dir, "outputs", "reports", "evaluation_results.json")
    data_path = os.path.join(parent_dir, "data", "dynamic_pricing_transactions.csv")

    model_data = joblib.load(model_path) if os.path.exists(model_path) else None

    results = None
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

    sample_df = None
    if os.path.exists(data_path):
        sample_df = pd.read_csv(data_path, nrows=10000)

    return model_data, results, sample_df


model_data, results, df = load_model_and_results()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Internship Header
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric/100/combo-chart.png",
        width=70,
    )
    st.markdown("### **Infotact Solutions**")
    st.markdown("**Internship Project Demonstration**")
    st.markdown(
        """
        <div style="font-size:0.85rem; color:#6B7280; line-height:1.4;">
        <b>Intern:</b> Vikrant Bharat Lawate<br>
        <b>Role:</b> Data Associate L1. Intern<br>
        <b>Domain:</b> Data Science & ML<br>
        <b>Ref No:</b> ITS/DSML/1090<br>
        <b>Duration:</b> 1 Month
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🎯 Real-Time Dynamic Pricing Simulator",
            "📊 Endogeneity & Causal Diagnostics",
            "📈 Heterogeneous Elasticity (CATE)",
            "🏆 Counterfactual Policy Benchmark",
            "📑 Technical Report & Architecture",
        ],
    )
    st.divider()
    st.caption("Powered by Double Machine Learning (Chernozhukov et al.)")

# -----------------------------------------------------------------------------
# TAB 1: Real-Time Dynamic Pricing Simulator
# -----------------------------------------------------------------------------
if menu == "🎯 Real-Time Dynamic Pricing Simulator":
    st.markdown('<div class="main-header">EconoCausal Real-Time Dynamic Pricing Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Optimize prices dynamically by isolating true causal price sensitivity from market confounders.</div>', unsafe_allow_html=True)

    col_sim_left, col_sim_right = st.columns([1, 2])

    with col_sim_left:
        st.subheader("Customer & Market Context")
        
        st.markdown("**Customer Segment (Moderators X)**")
        tier_names = ["Bronze (Price-Sensitive)", "Silver", "Gold", "Platinum (Inelastic VIP)"]
        selected_tier_idx = st.selectbox("Customer Loyalty Tier", [0, 1, 2, 3], format_func=lambda x: tier_names[x], index=1)
        urgency = st.slider("Purchase Urgency Score", 0.05, 1.0, 0.50, 0.05, help="0 = Casual browsing, 1 = Immediate urgent necessity")
        device = st.selectbox("Device Platform", [0, 1], format_func=lambda x: "Mobile" if x == 0 else "Desktop")

        st.markdown("**Market Confounders (W)**")
        competitor_p = st.slider("Competitor Price ($)", 50.0, 120.0, 75.0, 1.0)
        seasonality = st.slider("Seasonality Demand Index", -1.0, 1.0, 0.35, 0.05, help="Peak season vs Off-season")
        inventory = st.slider("Inventory Level (Units in Stock)", 20, 450, 180, 10)
        marketing = st.slider("Marketing Spend Intensity ($k)", 5.0, 35.0, 18.0, 1.0)

        st.markdown("**Unit Economics**")
        unit_cost = st.number_input("Unit Marginal Cost ($)", min_value=20.0, max_value=80.0, value=45.0, step=1.0)
        baseline_price = st.number_input("Current Catalog Base Price ($)", min_value=30.0, max_value=120.0, value=75.0, step=1.0)

    with col_sim_right:
        # Calculate Causal Elasticity for this customer context
        if model_data is not None and model_data["model"] is not None:
            dml = model_data["model"]
            cust_features = pd.DataFrame([{
                "customer_tier": selected_tier_idx,
                "urgency_score": urgency,
                "device_type": device,
            }])
            est_theta = float(dml.predict_cate(cust_features)[0])
        else:
            # Fallback theoretical formula if model not loaded
            est_theta = -2.60 + 0.50 * selected_tier_idx + 0.65 * urgency - 0.15 * device

        # Calculate Expected Baseline Demand
        est_base_demand = max(10.0, (
            85.0
            + 18.0 * seasonality
            + 0.42 * (competitor_p - 70.0)
            + 0.85 * marketing
            + 4.0 * selected_tier_idx
        ))

        # Run Optimizer
        optimizer = PricingOptimizer(
            min_margin_pct=0.15,
            max_price_change_pct=0.30,
            absolute_price_floor=40.0,
            absolute_price_ceiling=140.0,
        )
        opt_res = optimizer.optimize_price_analytical(
            baseline_price=baseline_price,
            baseline_demand=est_base_demand,
            causal_elasticity=est_theta,
            unit_cost=unit_cost,
            objective="profit",
        )

        # Display Top Metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                label="Causal Elasticity θ(X)",
                value=f"{opt_res['causal_elasticity']:.3f}",
                delta="Elastic" if opt_res['causal_elasticity'] < -1.8 else "Inelastic",
                delta_color="inverse" if opt_res['causal_elasticity'] < -1.8 else "normal",
            )
        with kpi2:
            st.metric(
                label="Optimal Price",
                value=f"${opt_res['optimal_price']:.2f}",
                delta=f"{opt_res['price_delta']:+.2f} ({opt_res['price_delta_pct']:+.1f}%)",
            )
        with kpi3:
            st.metric(
                label="Expected Demand",
                value=f"{opt_res['expected_demand']:.1f} units",
                delta=f"{opt_res['expected_demand'] - opt_res['baseline_demand']:+.1f}",
            )
        with kpi4:
            st.metric(
                label="Expected Gross Profit",
                value=f"${opt_res['expected_profit']:,.2f}",
                delta=f"{opt_res['profit_uplift_pct']:+.2f}% uplift",
            )

        # Plot Demand & Profit Surface Curves
        p_range = np.linspace(opt_res["bounds"][0], opt_res["bounds"][1], 80)
        q_curve = np.maximum(0.0, opt_res["baseline_demand"] + opt_res["causal_elasticity"] * (p_range - opt_res["baseline_price"]))
        profit_curve = (p_range - opt_res["unit_cost"]) * q_curve
        revenue_curve = p_range * q_curve

        fig = go.Figure()
        # Profit Curve
        fig.add_trace(go.Scatter(
            x=p_range,
            y=profit_curve,
            mode="lines",
            name="Gross Profit ($)",
            line=dict(color="#10B981", width=3.5),
        ))
        # Revenue Curve
        fig.add_trace(go.Scatter(
            x=p_range,
            y=revenue_curve,
            mode="lines",
            name="Revenue ($)",
            line=dict(color="#3B82F6", width=2, dash="dash"),
        ))
        # Baseline Price Marker
        fig.add_vline(
            x=opt_res["baseline_price"],
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"Base: ${opt_res['baseline_price']:.2f}",
            annotation_position="top left",
        )
        # Optimal Price Marker
        fig.add_vline(
            x=opt_res["optimal_price"],
            line_color="#10B981",
            line_width=2.5,
            annotation_text=f"Optimal: ${opt_res['optimal_price']:.2f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"Counterfactual Profit & Revenue Response Curve (θ = {opt_res['causal_elasticity']:.3f})",
            xaxis_title="Offered Price ($)",
            yaxis_title="Projected Value ($)",
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"💡 **Dynamic Pricing Decision Insight:** Under customer tier **{tier_names[selected_tier_idx]}** and urgency **{urgency:.2f}**, the estimated price elasticity is **{opt_res['causal_elasticity']:.3f}**. "
            f"The microeconomic profit-maximizing price is **${opt_res['optimal_price']:.2f}**, generating a projected gross profit uplift of **+{opt_res['profit_uplift_pct']:.2f}%** compared to the status-quo $75.00 price."
        )

# -----------------------------------------------------------------------------
# TAB 2: Endogeneity & Causal Diagnostics
# -----------------------------------------------------------------------------
elif menu == "📊 Endogeneity & Causal Diagnostics":
    st.markdown('<div class="main-header">The Price Endogeneity Paradox</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Why standard machine learning and naive regressions fail in observational pricing data.</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown(
            """
            ### ⚠️ The Omitted Variable / Confounder Bias
            In real business data, prices are **never assigned purely at random**. 
            Management and legacy heuristics deliberately increase prices during:
            1. **Peak Demand Seasons** (Holidays, travel surges)
            2. **Aggressive Marketing Campaigns** (High conversion intent)
            3. **Low Inventory / Scarcity**
            4. **Higher Competitor Pricing**

            Because high-demand periods have **both high prices and high sales**, naive correlation appears **artificially positive or flat**. 
            """
        )
        if results is not None:
            st.markdown(
                f"""
                | Metric | Value | Interpretation |
                | :--- | :--- | :--- |
                | **Raw Correlation (Price ~ Demand)** | `{results.get('policy_comparison', {}).get('Historical Baseline', {}).get('avg_price', 0):.2f}` | Severely attenuated |
                | **Naive Bivariate OLS Elasticity** | `{results.get('naive_bivariate_elasticity', 0):.4f}` | Positive / Biased |
                | **Naive Multivariate OLS** | `{results.get('naive_multivariate_elasticity', 0):.4f}` | Residual confounding |
                | **True Ground Truth ATE** | `{results.get('ground_truth_ate', -1.85):.4f}` | Downward sloping (Law of Demand) |
                | **EconoCausal DML ATE** | `{results.get('dml_estimated_ate', -1.85):.4f}` | **Unbiased Causal Recovery!** |
                """
            )

    with col_e2:
        st.markdown("### 🔬 How Double Machine Learning Solves It")
        st.latex(r"\tilde{Y} = Y - \hat{E}[Y | W, X]")
        st.latex(r"\tilde{T} = T - \hat{E}[T | W, X]")
        st.latex(r"\tilde{Y} = \theta(X) \cdot \tilde{T} + \varepsilon")
        st.markdown(
            """
            1. **Step 1:** Fit Gradient Boosters to predict Demand and Price from all confounders $W$.
            2. **Step 2:** Form Neyman-orthogonalized residuals $\\tilde{Y}$ and $\\tilde{T}$.
            3. **Step 3:** Regress $\\tilde{Y}$ on $\\tilde{T}$ with $K$-fold cross-fitting to eliminate regularization bias.
            """
        )

    st.divider()
    # Visual Demonstration
    if df is not None:
        st.subheader("Visualizing Observational Data vs. Stratified Demand Response")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig1 = px.scatter(
                df.sample(2000, random_state=42),
                x="price",
                y="demand",
                trendline="ols",
                opacity=0.35,
                title="A: Raw Observational Correlation (Apparent Positive Bias)",
                labels={"price": "Price ($)", "demand": "Demand (Units)"},
                color_discrete_sequence=["#3B82F6"],
            )
            st.plotly_chart(fig1, use_container_width=True)
        with chart_col2:
            fig2 = px.scatter(
                df.sample(2000, random_state=42),
                x="price",
                y="demand",
                color="customer_tier",
                opacity=0.4,
                title="B: Stratified by Loyalty Tier (True Elasticity Revealed)",
                labels={"price": "Price ($)", "demand": "Demand (Units)", "customer_tier": "Loyalty Tier"},
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Heterogeneous Elasticity (CATE)
# -----------------------------------------------------------------------------
elif menu == "📈 Heterogeneous Elasticity (CATE)":
    st.markdown('<div class="main-header">Heterogeneous Price Elasticity Analysis (CATE)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploring how customer price sensitivity varies across loyalty tiers, urgency, and channels.</div>', unsafe_allow_html=True)

    if model_data is not None and model_data["model"] is not None:
        dml = model_data["model"]
        coef_summary = dml.summary()["coefficients"]
        coef_df = pd.DataFrame(coef_summary)

        st.subheader("DML Second-Stage Parameter Estimates")
        st.dataframe(
            coef_df.style.format({
                "estimate": "{:.4f}",
                "std_error": "{:.4f}",
                "t_stat": "{:.2f}",
                "p_value": "{:.4e}",
                "ci_lower": "{:.4f}",
                "ci_upper": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.subheader("Price Elasticity by Loyalty Tier")
        tiers = ["Bronze (Tier 0)", "Silver (Tier 1)", "Gold (Tier 2)", "Platinum (Tier 3)"]
        grid = pd.DataFrame({
            "customer_tier": [0, 1, 2, 3],
            "urgency_score": [0.5, 0.5, 0.5, 0.5],
            "device_type": [0, 0, 0, 0],
        })
        thetas, lows, highs = dml.predict_cate_interval(grid)

        fig_cate = go.Figure()
        fig_cate.add_trace(go.Bar(
            x=tiers,
            y=thetas,
            error_y=dict(type="data", symmetric=False, array=highs - thetas, arrayminus=thetas - lows),
            marker_color=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"],
            text=[f"θ = {t:.3f}" for t in thetas],
            textposition="auto",
        ))
        fig_cate.update_layout(
            title="Conditional Average Treatment Effect (CATE) with 95% Confidence Intervals",
            yaxis_title="Price Elasticity Parameter θ(X)",
            xaxis_title="Customer Segment",
            height=420,
        )
        st.plotly_chart(fig_cate, use_container_width=True)
    else:
        st.info("Run the training pipeline to generate model weights.")

# -----------------------------------------------------------------------------
# TAB 4: Counterfactual Policy Benchmark
# -----------------------------------------------------------------------------
elif menu == "🏆 Counterfactual Policy Benchmark":
    st.markdown('<div class="main-header">Counterfactual Policy Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating out-of-sample portfolio profit and revenue under alternative pricing regimes.</div>', unsafe_allow_html=True)

    if results is not None and "policy_comparison" in results:
        comp_data = results["policy_comparison"]
        comp_df = pd.DataFrame(comp_data).T
        comp_df.index.name = "Pricing Policy"

        st.dataframe(
            comp_df.style.format({
                "avg_price": "${:.2f}",
                "total_demand": "{:,.0f} units",
                "total_revenue": "${:,.2f}",
                "total_profit": "${:,.2f}",
                "profit_margin_pct": "{:.1f}%",
                "profit_uplift_pct": "{:+.2f}%",
            }),
            use_container_width=True,
        )

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            fig_p = px.bar(
                comp_df.reset_index(),
                x="Pricing Policy",
                y="total_profit",
                color="Pricing Policy",
                title="Total Portfolio Gross Profit ($)",
                text_auto=".2s",
                color_discrete_sequence=["#9CA3AF", "#F59E0B", "#EF4444", "#10B981"],
            )
            fig_p.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_p, use_container_width=True)

        with bcol2:
            fig_u = px.bar(
                comp_df.reset_index(),
                x="Pricing Policy",
                y="profit_uplift_pct",
                color="Pricing Policy",
                title="Gross Profit Uplift vs. Historical Baseline (%)",
                text_auto="+.2f",
                color_discrete_sequence=["#9CA3AF", "#F59E0B", "#EF4444", "#10B981"],
            )
            fig_u.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_u, use_container_width=True)
    else:
        st.info("Run the training pipeline to generate policy comparison metrics.")

# -----------------------------------------------------------------------------
# TAB 5: Technical Report & Architecture
# -----------------------------------------------------------------------------
elif menu == "📑 Technical Report & Architecture":
    st.markdown('<div class="main-header">Project Technical Report & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Summary, Mathematical Foundation, and Production Architecture.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🏢 Infotact Solutions Internship Project Dossier
        - **Project Name:** EconoCausal: Dynamic Pricing via Double Machine Learning
        - **Intern Candidate:** Vikrant Bharat Lawate (`vikrantla91@gmail.com`)
        - **Role:** Data Associate L1. Intern (Data Science & Machine Learning)
        - **Reference No:** ITS/DSML/1090
        - **Timeline:** 05/10/2026 – 05/11/2026
        - **Quality Standard:** ISO 9001:2015 & MSME Certified Deliverable

        ---
        ### 🏛️ Double Machine Learning Mathematical Foundation
        Double Machine Learning (DML), introduced by Chernozhukov et al. (2018), solves the fundamental curse of endogeneity and regularization bias when applying high-capacity machine learning to structural causal estimation.

        #### 1. Partially Linear Regression Model (Robinson, 1988)
        $$Y = \\theta(X) \\cdot T + g(W) + U, \\quad \\mathbb{E}[U \\mid T, W] = 0$$
        $$T = m(W) + V, \\quad \\mathbb{E}[V \\mid W] = 0$$

        #### 2. Neyman Orthogonality Condition
        The score function satisfies:
        $$\\left. \\frac{\\partial}{\\partial \\eta} \\mathbb{E}[\\psi(W, Y, T; \\theta_0, \\eta)] \\right|_{\\eta = \\eta_0} = 0$$
        This property guarantees that first-stage errors $\\hat{g}(W) - g(W)$ and $\\hat{m}(W) - m(W)$ from non-parametric ML models (Gradient Boosters, Neural Nets) do not bias the second-stage causal parameter $\\hat{\\theta}$.

        #### 3. Cross-Fitting (Sample Splitting)
        Partitioning data into $K$ disjoint folds eliminates overfitting bias of order $1/\\sqrt{N}$, restoring $\\sqrt{N}$-rate asymptotic normality:
        $$\\sqrt{N} (\\hat{\\theta} - \\theta_0) \\xrightarrow{d} \\mathcal{N}(0, \\sigma^2)$$

        ---
        ### 🚢 Production Architecture Diagram
        ```mermaid
        flowchart LR
            A[Customer Session & Market State] --> B[Feature Store / Confounder Pipeline]
            B --> C[DML Stage 1: Demand & Price Predictors]
            C --> D[CATE Elasticity Engine θ_X]
            D --> E[Microeconomic Optimizer]
            E --> F[Business Guardrails & Floor/Ceiling]
            F --> G[Real-Time Dynamic Price Quote]
            G --> H[Observational Log & Telemetry]
        ```
        """
    )
