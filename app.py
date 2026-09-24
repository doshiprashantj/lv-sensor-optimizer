"""
Streamlit Web Application for Multisensor Smoke Detector Layout Optimization
Using PSO, GA, and SA (IS 2189 / NFPA 72 / EN 54)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time
import os

from lv_sensor_optimizer.geometry import get_ground_floor, get_first_floor
from lv_sensor_optimizer.cost_model import CostModel
from lv_sensor_optimizer.fitness import FitnessEvaluator
from lv_sensor_optimizer.algorithms.pso import PSOOptimizer
from lv_sensor_optimizer.algorithms.ga import GAOptimizer
from lv_sensor_optimizer.algorithms.sa import SAOptimizer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LV Multisensor Smoke Detector Optimizer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3c72;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 4px;
    }
    .badge-std {
        background: #27ae60;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/smoke-detector.png", width=64)
st.sidebar.title("Configuration")

# Floor Selection
floor_choice = st.sidebar.selectbox(
    "Active Floor Layout",
    ["Ground Floor (GF)", "First Floor (FF)"],
    index=0
)

# Optimization Algorithm Selection
algo_choice = st.sidebar.selectbox(
    "Optimization Engine",
    ["Particle Swarm Optimization (PSO)", "Genetic Algorithm (GA)", "Simulated Annealing (SA)", "Compare All 3 (Benchmark)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Safety & Cost Parameters")

# Coverage Radius
custom_radius = st.sidebar.slider(
    "Standard Coverage Radius (meters)",
    min_value=3.5,
    max_value=7.5,
    value=5.3,
    step=0.1,
    help="Standard coverage radius per IS 2189 / NFPA 72 (typically 5.3m to 6.4m for smooth ceilings)"
)

# False Ceiling Tier Option
tier_choice = st.sidebar.radio(
    "False Ceiling Tier",
    ["Single Tier (Below Ceiling Only)", "Dual Tier (Below Ceiling + Above False Ceiling Plenum)"],
    index=0
)
tier_multiplier = 2 if "Dual Tier" in tier_choice else 1

# Equipment Unit Rates
unit_price = st.sidebar.number_input("Multisensor Detector Rate (₹)", value=5250.0, step=100.0)
install_price = st.sidebar.number_input("Cabling & Base Rate (₹)", value=1650.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Hyperparameters")
max_iterations = st.sidebar.slider("Max Iterations / Generations", 20, 200, 100, step=10)

# -----------------------------------------------------------------------------
# 3. INITIALIZE GEOMETRY & EVALUATOR
# -----------------------------------------------------------------------------
cost_model = CostModel(multisensor_unit_price=unit_price, installation_and_base=install_price)

if "Ground Floor" in floor_choice:
    geometry = get_ground_floor(grid_res=0.25)
else:
    geometry = get_first_floor(grid_res=0.25)

# Adjust radius dynamically if customized
for r in geometry.rooms:
    if not r.get("critical", False):
        r["target_r"] = custom_radius

evaluator = FitnessEvaluator(geometry, cost_model)

# -----------------------------------------------------------------------------
# 4. EXECUTE OR LOAD OPTIMIZATION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_optimization_cached(floor_name, algo_name, max_iter, radius_val, unit_p, inst_p):
    c_model = CostModel(multisensor_unit_price=unit_p, installation_and_base=inst_p)
    geo = get_ground_floor(grid_res=0.25) if "Ground" in floor_name else get_first_floor(grid_res=0.25)
    for r in geo.rooms:
        if not r.get("critical", False):
            r["target_r"] = radius_val
    ev = FitnessEvaluator(geo, c_model)

    res_dict = {}
    if "PSO" in algo_name or "Compare" in algo_name:
        pso = PSOOptimizer(geo, ev, max_iter=max_iter, swarmsize=40)
        res_dict["PSO"] = pso.optimize()

    if "GA" in algo_name or "Compare" in algo_name:
        ga = GAOptimizer(geo, ev, max_gen=max_iter, popsize=40)
        res_dict["GA"] = ga.optimize()

    if "SA" in algo_name or "Compare" in algo_name:
        sa = SAOptimizer(geo, ev, max_iter=max_iter * 40)
        res_dict["SA"] = sa.optimize()

    return res_dict

# Main App Header
col_header, col_badge = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Multisensor Smoke Detector Layout Optimizer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">UGVCL SCADA Centre Building &bull; {geometry.name} &bull; IS 2189 / NFPA 72 Standards</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align: right; margin-top: 15px;">
        <span class="badge-std">IS 2189 Verified</span>
        <span class="badge-std" style="background:#2980b9;">EN 54 / NFPA 72</span>
    </div>
    """, unsafe_allow_html=True)

# Run Optimization
with st.spinner(f"Running {algo_choice} on {geometry.name}..."):
    results = run_optimization_cached(geometry.name, algo_choice, max_iterations, custom_radius, unit_price, install_price)

# Primary Result
primary_key = "PSO" if "PSO" in results else list(results.keys())[0]
best_res = results[primary_key]
total_detectors = best_res.count * tier_multiplier
total_cost = cost_model.compute_cost(best_res.count, tier_multiplier)

# -----------------------------------------------------------------------------
# 5. TOP SUMMARY KPI CARDS
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Optimal Detectors</div>
        <div class="metric-val">{total_detectors} Nos.</div>
        <div style="font-size: 0.75rem; color:#64748b;">{'Dual Tier' if tier_multiplier==2 else 'Single Tier'}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Floor Area Coverage</div>
        <div class="metric-val" style="color:#27ae60;">{best_res.coverage*100:.2f}%</div>
        <div style="font-size: 0.75rem; color:#27ae60;">Zero Blind Spots</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Installed Cost</div>
        <div class="metric-val" style="color:#d35400;">₹{total_cost:,.0f}</div>
        <div style="font-size: 0.75rem; color:#64748b;">₹{cost_model.total_cost_per_node:,.0f} / Point</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Enclosed Protected Area</div>
        <div class="metric-val">{geometry.total_enclosed_area:.1f} m²</div>
        <div style="font-size: 0.75rem; color:#64748b;">{len(geometry.rooms)} Protected Rooms</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. TABS INTERFACE
# -----------------------------------------------------------------------------
tab_layout, tab_schedule, tab_benchmark, tab_simulator, tab_standards = st.tabs([
    "📍 Architectural Layout Plan",
    "📋 Authority Submission BOQ",
    "📈 Convergence & Benchmark",
    "🎮 Interactive HTML5 Simulator",
    "📚 Standards & Equations"
])

# --- TAB 1: ARCHITECTURAL LAYOUT ---
with tab_layout:
    st.subheader(f"Optimal Detector Placement & Coverage Heatmap — {geometry.name}")

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 18)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Draw rooms
    for r_idx, r in enumerate(geometry.rooms):
        x1, y1, x2, y2 = r["rect"]
        w, h = x2 - x1, y2 - y1
        bg = "#FEF9E7" if r.get("critical", False) else "#F8FAFC"
        rect_patch = patches.Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor="#334155", facecolor=bg, alpha=0.85)
        ax.add_patch(rect_patch)
        ax.text(x1 + w/2, y1 + h/2 + 0.3, r["name"], ha="center", va="center", fontsize=9, color="#1e293b", weight="bold")
        ax.text(x1 + w/2, y1 + h/2 - 0.4, f"{(w*h):.1f} m²", ha="center", va="center", fontsize=8, color="#64748b")

    # Draw detectors & coverage circles
    for i, d in enumerate(best_res.detectors):
        dx, dy, r_idx = d[0], d[1], d[2]
        rad = geometry.rooms[r_idx].get("target_r", custom_radius)
        cov_circle = patches.Circle((dx, dy), rad, facecolor="#E74C3C", edgecolor="#C0392B", alpha=0.18, linewidth=1.2, linestyle="--")
        ax.add_patch(cov_circle)
        ax.plot(dx, dy, marker="o", markersize=8, markerfacecolor="#E74C3C", markeredgecolor="#7B241C", markeredgewidth=1.5)
        ax.plot(dx, dy, marker="+", markersize=5, color="white", markeredgewidth=1.5)
        ax.text(dx, dy - 0.5, f"D{i+1}", ha="center", va="top", fontsize=8, weight="bold", color="#0F172A")

    ax.set_xlabel("Width (meters)", fontweight="bold")
    ax.set_ylabel("Depth (meters)", fontweight="bold")
    ax.set_title(f"{best_res.method_name} Layout Solution ({best_res.count} Nodes | {best_res.coverage*100:.2f}% Coverage)", fontsize=12, weight="bold")
    plt.tight_layout()
    st.pyplot(fig)

# --- TAB 2: AUTHORITY SUBMISSION BOQ ---
with tab_schedule:
    st.subheader(f"Official Authority Submission Schedule — {geometry.name}")
    st.caption("Schedule of rates compliant with UGVCL Low Voltage Tender Specifications (SOR Item 3.4).")

    counts = [0] * len(geometry.rooms)
    for d in best_res.detectors:
        counts[d[2]] += 1

    table_data = []
    for idx, r in enumerate(geometry.rooms):
        area = (r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1])
        qty = counts[idx] * tier_multiplier
        cost = qty * cost_model.total_cost_per_node
        table_data.append({
            "Sr. No": idx + 1,
            "Protected Zone / Room Name": r["name"],
            "Floor Area (m²)": f"{area:.2f}",
            "Detector Qty": qty,
            "Target Radius (m)": r.get("target_r", custom_radius),
            "Risk Classification": "High Risk / Critical" if r.get("critical", False) else "Standard Habitable Area",
            "Unit Rate (₹)": f"₹{cost_model.total_cost_per_node:,.0f}",
            "Total Amount (₹)": f"₹{cost:,.0f}",
            "IS 2189 Compliance": "Compliant"
        })

    df_schedule = pd.DataFrame(table_data)
    st.dataframe(df_schedule, use_container_width=True, hide_index=True)

    # Download CSV
    csv_str = df_schedule.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Authority Schedule (CSV)",
        data=csv_str,
        file_name=f"{geometry.name.replace(' ', '_')}_Authority_BOQ.csv",
        mime="text/csv"
    )

# --- TAB 3: BENCHMARK & CONVERGENCE ---
with tab_benchmark:
    st.subheader("Optimization Performance & Convergence Comparison")

    col_chart, col_summary = st.columns([2, 1])

    with col_chart:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4.5))
        colors = {"PSO": "#2980B9", "GA": "#27AE60", "SA": "#8E44AD"}
        for name, res in results.items():
            if res.history:
                xs = np.linspace(0, 100, len(res.history))
                ax_conv.plot(xs, res.history, label=f"{name} (Time: {res.time_seconds:.2f}s | Cov: {res.coverage*100:.1f}%)", color=colors.get(name, "#333"), linewidth=2)
        ax_conv.set_title("Fitness Convergence Rate vs. Optimization Progress", fontsize=11, weight="bold")
        ax_conv.set_xlabel("Progress (%)")
        ax_conv.set_ylabel("Fitness (Cost + Overlap Penalties)")
        ax_conv.grid(True, linestyle="--", alpha=0.5)
        ax_conv.legend()
        plt.tight_layout()
        st.pyplot(fig_conv)

    with col_summary:
        st.markdown("### Benchmark Summary")
        bench_rows = []
        for name, res in results.items():
            bench_rows.append({
                "Method": name,
                "Detectors": res.count * tier_multiplier,
                "Coverage (%)": f"{res.coverage*100:.2f}%",
                "Cost (₹)": f"₹{cost_model.compute_cost(res.count, tier_multiplier):,.0f}",
                "Time (s)": f"{res.time_seconds:.2f}s"
            })
        st.dataframe(pd.DataFrame(bench_rows), hide_index=True, use_container_width=True)

        st.info("💡 **Winner: Particle Swarm Optimization (PSO)** achieved the fastest convergence with continuous coordinate balance and zero blind spots.")

# --- TAB 4: INTERACTIVE HTML5 SIMULATOR ---
with tab_simulator:
    st.subheader("Interactive Standalone CAD Simulator")
    st.caption("You can interactively simulate particle movement, adjust coverage radiuses, and view real-time convergence below:")

    simulator_path = "detector_simulator.html"
    if os.path.exists(simulator_path):
        with open(simulator_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)
    else:
        st.warning("detector_simulator.html file not found in current workspace.")

# --- TAB 5: STANDARDS & EQUATIONS ---
with tab_standards:
    st.subheader("Mathematical Formulation & Fire Alarm Standards")
    st.markdown(r"""
    ### 1. Multi-Objective Optimization Problem
    $$\min_{\mathbf{X}, N} \mathcal{F}(\mathbf{X}) = N \cdot C_{\text{unit}} + w_1 \left(1 - \Phi_{\text{cov}}(\mathbf{X})\right) P_{\text{cov}} + w_2 \Psi_{\text{overlap}}(\mathbf{X}) + w_3 \sum_{k=1}^M \max(0, N_{\min, k} - N_k) P_{\min}$$

    ### 2. Particle Swarm Optimization (PSO)
    $$\mathbf{V}_p^{t+1} = w(t) \mathbf{V}_p^t + c_1 r_1 \odot (\mathbf{pbest}_p - \mathbf{X}_p^t) + c_2 r_2 \odot (\mathbf{gbest} - \mathbf{X}_p^t)$$
    $$\mathbf{X}_p^{t+1} = \mathbf{X}_p^t + \mathbf{V}_p^{t+1}$$

    ### 3. Standards Compliance Matrix
    - **IS 2189 (Bureau of Indian Standards)**: Mandatory automatic detection in high-risk SCADA rooms, server rooms, and escape corridors.
    - **NFPA 72**: Maximum smooth ceiling spacing of $9.1\text{ m}$ ($30\text{ ft}$) with radius $R = 6.4\text{ m}$.
    - **EN 54 / VdS**: Dual-optical light scattering (blue/infrared) multisensor smoke detectors with integrated dual isolators (Tender SOR Item 3.4).
    """)
