/**
 * EconoCausal: Dynamic Pricing via Double Machine Learning
 * Frontend Controller & Chart.js Visualizations
 */

document.addEventListener("DOMContentLoaded", () => {
    // Determine active page
    const pathname = window.location.pathname;

    if (pathname.includes("/dashboard") || pathname === "/dashboard") {
        initDashboardPage();
    } else if (pathname.includes("/analysis")) {
        initAnalysisPage();
    } else if (pathname.includes("/simulator")) {
        initSimulatorPage();
    }
});

// -----------------------------------------------------------------------------
// DASHBOARD PAGE
// -----------------------------------------------------------------------------
async function initDashboardPage() {
    try {
        const [sumRes, analRes] = await Promise.all([
            fetch("/api/summary").then(r => r.json()),
            fetch("/api/analytics").then(r => r.json())
        ]);

        if (sumRes.status === "success") {
            const data = sumRes.data;
            document.getElementById("kpi-total-products").textContent = data.total_products;
            document.getElementById("kpi-total-records").textContent = Number(data.total_sales_records).toLocaleString();
            document.getElementById("kpi-avg-price").textContent = "₹" + Number(data.avg_price).toLocaleString();
            document.getElementById("kpi-avg-demand").textContent = data.avg_demand + " units";
            document.getElementById("kpi-revenue").textContent = "₹" + Number(data.current_revenue).toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById("kpi-causal-effect").textContent = data.estimated_causal_effect + " / ₹";
            document.getElementById("causal-interpretation").textContent = data.interpretation;
        }

        if (analRes.status === "success") {
            const a = analRes.analytics;
            renderPriceDemandChart(a.scatter_data);
            renderCategoryDemandChart(a.category_data);
            renderMonthlyTrendChart(a.monthly_data);
        }
    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
}

function renderPriceDemandChart(scatterData) {
    const ctx = document.getElementById("chart-price-demand");
    if (!ctx) return;

    // Sort by price for smooth line/scatter
    const sorted = [...scatterData].sort((a, b) => a.price - b.price);
    const prices = sorted.map(d => d.price);
    const demands = sorted.map(d => d.demand);

    new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                label: "Transaction Observations (Price vs Demand)",
                data: sorted.map(d => ({ x: d.price, y: d.demand })),
                backgroundColor: "rgba(37, 99, 235, 0.4)",
                borderColor: "rgba(37, 99, 235, 0.8)",
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top" },
                tooltip: {
                    callbacks: {
                        label: (c) => `Price: ₹${c.raw.x.toFixed(2)}, Demand: ${c.raw.y} units`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: "Offered Price (₹)" } },
                y: { title: { display: true, text: "Quantity Sold (Units)" } }
            }
        }
    });
}

function renderCategoryDemandChart(categoryData) {
    const ctx = document.getElementById("chart-category-demand");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: categoryData.map(c => c.category),
            datasets: [{
                label: "Average Daily Demand",
                data: categoryData.map(c => c.avg_demand),
                backgroundColor: ["#3b82f6", "#10b981", "#8b5cf6"],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Average Demand (Units)" } }
            }
        }
    });
}

function renderMonthlyTrendChart(monthlyData) {
    const ctx = document.getElementById("chart-monthly-trend");
    if (!ctx) return;

    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    new Chart(ctx, {
        type: "line",
        data: {
            labels: monthlyData.map(m => monthNames[m.month - 1]),
            datasets: [{
                label: "Monthly Avg Demand",
                data: monthlyData.map(m => m.avg_demand),
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.15)",
                fill: true,
                tension: 0.35,
                borderWidth: 3,
                pointRadius: 5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: false, title: { display: true, text: "Avg Demand Units" } }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// ANALYSIS PAGE
// -----------------------------------------------------------------------------
async function initAnalysisPage() {
    try {
        const [analRes, causalRes] = await Promise.all([
            fetch("/api/analytics").then(r => r.json()),
            fetch("/api/causal-effect").then(r => r.json())
        ]);

        if (causalRes.status === "success") {
            const c = causalRes.data;
            document.getElementById("causal-ate-val").textContent = `${c.ate} / ₹`;
            document.getElementById("causal-ci-val").textContent = `[${c.ci_lower}, ${c.ci_upper}]`;
            document.getElementById("causal-interp-val").textContent = c.interpretation;
        }

        if (analRes.status === "success") {
            const a = analRes.analytics;
            renderPriceRevenueProfitChart(a.scatter_data);
            renderCompetitorChart(a.scatter_data);
            renderPromoDemandChart(a.promotion_data);
            renderSeasonDemandChart(a.season_data);
            renderInventoryDemandChart(a.scatter_data);
        }
    } catch (err) {
        console.error("Error in analysis:", err);
    }
}

function renderPriceRevenueProfitChart(scatterData) {
    const ctx = document.getElementById("chart-price-revenue-profit");
    if (!ctx) return;

    const sorted = [...scatterData].sort((a, b) => a.price - b.price);

    new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Gross Profit (₹)",
                    data: sorted.map(d => ({ x: d.price, y: d.profit })),
                    backgroundColor: "rgba(16, 185, 129, 0.6)",
                    borderColor: "#10b981",
                    pointRadius: 4,
                },
                {
                    label: "Revenue (₹)",
                    data: sorted.map(d => ({ x: d.price, y: d.revenue })),
                    backgroundColor: "rgba(59, 130, 246, 0.4)",
                    borderColor: "#3b82f6",
                    pointRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Price (₹)" } },
                y: { title: { display: true, text: "Financial Outcome (₹)" } }
            }
        }
    });
}

function renderCompetitorChart(scatterData) {
    const ctx = document.getElementById("chart-competitor-demand");
    if (!ctx) return;

    new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                label: "Competitor Price vs Our Demand",
                data: scatterData.map(d => ({ x: d.competitor_price, y: d.demand })),
                backgroundColor: "rgba(139, 92, 246, 0.5)",
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Competitor Price (₹)" } },
                y: { title: { display: true, text: "Our Demand (Units)" } }
            }
        }
    });
}

function renderPromoDemandChart(promoData) {
    const ctx = document.getElementById("chart-promo-demand");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["No Promotion (Baseline)", "Active Promotion Campaign"],
            datasets: [{
                label: "Average Demand",
                data: promoData.map(p => p.avg_demand),
                backgroundColor: ["#94a3b8", "#10b981"],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Average Demand (Units)" } }
            }
        }
    });
}

function renderSeasonDemandChart(seasonData) {
    const ctx = document.getElementById("chart-season-demand");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: seasonData.map(s => s.season),
            datasets: [{
                label: "Average Seasonal Demand",
                data: seasonData.map(s => s.avg_demand),
                backgroundColor: ["#38bdf8", "#facc15", "#f97316", "#818cf8"],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Units Sold" } }
            }
        }
    });
}

function renderInventoryDemandChart(scatterData) {
    const ctx = document.getElementById("chart-inventory-demand");
    if (!ctx) return;

    new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                label: "Inventory Level vs Demand",
                data: scatterData.map(d => ({ x: d.inventory, y: d.demand })),
                backgroundColor: "rgba(245, 158, 11, 0.5)",
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Inventory Level (Units)" } },
                y: { title: { display: true, text: "Demand (Units)" } }
            }
        }
    });
}

// -----------------------------------------------------------------------------
// PRICING SIMULATOR PAGE
// -----------------------------------------------------------------------------
let simChart = null;

async function initSimulatorPage() {
    const productSelect = document.getElementById("sim-product-select");
    const form = document.getElementById("simulator-form");

    if (!productSelect || !form) return;

    // Load products into dropdown
    try {
        const res = await fetch("/api/products").then(r => r.json());
        if (res.status === "success") {
            productSelect.innerHTML = "";
            res.products.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.product_id;
                opt.textContent = `${p.product_name} (${p.category})`;
                opt.dataset.price = p.base_price;
                opt.dataset.cost = p.cost;
                opt.dataset.category = p.category;
                productSelect.appendChild(opt);
            });

            // Set initial defaults
            updateProductDefaults();
        }
    } catch (err) {
        console.error("Error loading products:", err);
    }

    productSelect.addEventListener("change", updateProductDefaults);

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await runSimulation();
    });

    // Run first calculation automatically
    setTimeout(runSimulation, 400);
}

function updateProductDefaults() {
    const select = document.getElementById("sim-product-select");
    const opt = select.options[select.selectedIndex];
    if (!opt) return;

    const basePrice = parseFloat(opt.dataset.price);
    const cost = parseFloat(opt.dataset.cost);

    document.getElementById("sim-current-price").value = basePrice;
    document.getElementById("sim-cost").value = cost;
    document.getElementById("sim-competitor-price").value = Math.round(basePrice * 1.03);
}

async function runSimulation() {
    const submitBtn = document.getElementById("sim-submit-btn");
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = "⏳ Calculating Optimal Price...";
    }

    const payload = {
        product_id: document.getElementById("sim-product-select").value,
        current_price: parseFloat(document.getElementById("sim-current-price").value),
        cost: parseFloat(document.getElementById("sim-cost").value),
        competitor_price: parseFloat(document.getElementById("sim-competitor-price").value),
        promotion: parseInt(document.getElementById("sim-promotion").value),
        holiday: parseInt(document.getElementById("sim-holiday").value),
        inventory: parseInt(document.getElementById("sim-inventory").value),
        temperature: parseFloat(document.getElementById("sim-temperature")?.value || 25.0),
        day_of_week: document.getElementById("sim-day-of-week")?.value || "Wednesday",
        month: parseInt(document.getElementById("sim-month")?.value || 6),
        season: document.getElementById("sim-season")?.value || "Summer",
    };

    try {
        const res = await fetch("/api/recommend-price", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        }).then(r => r.json());

        if (res.status === "success") {
            const rec = res.recommendation;
            displaySimulationResults(rec);
        } else {
            alert("Error in simulation: " + res.message);
        }
    } catch (err) {
        console.error("Simulation error:", err);
        alert("Failed to compute pricing recommendation.");
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = "🎯 Calculate Optimal Price";
        }
    }
}

function displaySimulationResults(rec) {
    document.getElementById("res-current-price").textContent = "₹" + rec.current_price.toLocaleString();
    document.getElementById("res-recommended-price").textContent = "₹" + rec.recommended_price.toLocaleString();
    document.getElementById("res-expected-demand").textContent = rec.expected_demand + " units";
    document.getElementById("res-expected-revenue").textContent = "₹" + Number(rec.expected_revenue).toLocaleString();
    document.getElementById("res-expected-profit").textContent = "₹" + Number(rec.expected_profit).toLocaleString();

    const upliftEl = document.getElementById("res-profit-uplift");
    if (upliftEl) {
        upliftEl.textContent = `${rec.profit_uplift_pct >= 0 ? "+" : ""}${rec.profit_uplift_pct}% Profit Uplift`;
        upliftEl.className = rec.profit_uplift_pct >= 0 ? "text-positive" : "text-negative";
    }

    document.getElementById("res-revenue-opt-price").textContent = "₹" + rec.revenue_maximizing_price.toLocaleString();
    document.getElementById("res-causal-elasticity").textContent = `${rec.causal_elasticity} / ₹`;

    // Render candidate comparison table
    const tableBody = document.getElementById("candidate-table-body");
    tableBody.innerHTML = "";

    rec.candidate_grid.forEach(row => {
        const tr = document.createElement("tr");
        if (row.is_best_profit) tr.classList.add("highlight-best");
        else if (row.is_current) tr.classList.add("highlight-current");

        tr.innerHTML = `
            <td>
                ₹${row.price.toLocaleString()}
                ${row.is_best_profit ? ' <span class="badge badge-secondary">Optimal Profit</span>' : ''}
                ${row.is_current ? ' <span class="badge badge-dark">Current</span>' : ''}
                ${row.is_best_revenue && !row.is_best_profit ? ' <span class="badge badge-primary">Max Revenue</span>' : ''}
            </td>
            <td>${row.demand} units</td>
            <td>₹${Number(row.revenue).toLocaleString()}</td>
            <td>₹${Number(row.profit).toLocaleString()}</td>
            <td>${row.margin_pct}%</td>
        `;
        tableBody.appendChild(tr);
    });

    // Wire up Dynamic Price Approval to Live Store
    const applyBox = document.getElementById("apply-price-box");
    const approveBtn = document.getElementById("btn-approve-price");
    const approveVal = document.getElementById("btn-approve-price-val");
    const statusText = document.getElementById("price-applied-status");

    if (applyBox && approveBtn && approveVal) {
        applyBox.style.display = "block";
        approveVal.textContent = `₹${rec.recommended_price.toLocaleString()}`;
        approveBtn.disabled = false;
        if (statusText) statusText.style.display = "none";

        approveBtn.onclick = async () => {
            approveBtn.disabled = true;
            approveBtn.innerHTML = "⏳ Updating Store Catalog...";
            try {
                const prodId = document.getElementById("sim-product-select").value;
                const updateRes = await fetch("/admin/products/update-price", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        product_id: prodId,
                        new_price: rec.recommended_price
                    })
                }).then(r => r.json());

                if (updateRes.status === "success") {
                    approveBtn.innerHTML = "✅ Price Approved & Live";
                    if (statusText) {
                        statusText.style.display = "inline";
                        statusText.textContent = `Catalog updated! Customers now see ₹${rec.recommended_price.toLocaleString()}`;
                    }
                    // Update input current price
                    document.getElementById("sim-current-price").value = rec.recommended_price;
                    document.getElementById("res-current-price").textContent = `₹${rec.recommended_price.toLocaleString()}`;
                } else {
                    alert("Error updating price: " + updateRes.message);
                    approveBtn.disabled = false;
                    approveBtn.innerHTML = `<i class="fa-solid fa-tag"></i> Approve & Apply Price (₹${rec.recommended_price.toLocaleString()})`;
                }
            } catch (err) {
                console.error("Error approving price:", err);
                approveBtn.disabled = false;
                approveBtn.innerHTML = `<i class="fa-solid fa-tag"></i> Approve & Apply Price (₹${rec.recommended_price.toLocaleString()})`;
            }
        };
    }

    // Render dynamic response curve chart
    renderSimulatorCurve(rec.candidate_grid, rec.recommended_price, rec.current_price);
}

function renderSimulatorCurve(grid, bestPrice, currPrice) {
    const ctx = document.getElementById("chart-simulator-curve");
    if (!ctx) return;

    if (simChart) simChart.destroy();

    const labels = grid.map(g => `₹${g.price}`);
    const profits = grid.map(g => g.profit);
    const revenues = grid.map(g => g.revenue);

    simChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Expected Gross Profit (₹)",
                    data: profits,
                    borderColor: "#10b981",
                    backgroundColor: "rgba(16, 185, 129, 0.15)",
                    fill: true,
                    tension: 0.3,
                    borderWidth: 3,
                    pointRadius: grid.map(g => (g.price === bestPrice ? 7 : 3)),
                    pointBackgroundColor: grid.map(g => (g.price === bestPrice ? "#059669" : "#10b981")),
                },
                {
                    label: "Expected Revenue (₹)",
                    data: revenues,
                    borderColor: "#3b82f6",
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (c) => `${c.dataset.label}: ₹${c.raw.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: "Candidate Price (₹)" } },
                y: { title: { display: true, text: "Value (₹)" } }
            }
        }
    });
}
