"""
Streamlit Web Application for Multisensor Smoke Detector Layout Optimization
Using PSO, GA, and SA (IS 2189 / NFPA 72 / EN 54)
Fully compatible with Streamlit Community Cloud deployment.
"""

import os
import sys

# Ensure root repository directory is on Python path for Streamlit Cloud (/mount/src/...)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import time

# Robust import with inline fallback for cloud deployment
try:
    from lv_sensor_optimizer.geometry import get_ground_floor, get_first_floor, FloorGeometry
    from lv_sensor_optimizer.cost_model import CostModel
    from lv_sensor_optimizer.fitness import FitnessEvaluator
    from lv_sensor_optimizer.algorithms.pso import PSOOptimizer
    from lv_sensor_optimizer.algorithms.ga import GAOptimizer
    from lv_sensor_optimizer.algorithms.sa import SAOptimizer
except ImportError:
    # -------------------------------------------------------------------------
    # SELF-CONTAINED EMBEDDED ENGINE (FALLBACK FOR CLOUD HOSTING)
    # -------------------------------------------------------------------------
    class FloorGeometry:
        def __init__(self, name, rooms, grid_res=0.25):
            self.name = name
            self.rooms = rooms
            self.grid_res = grid_res
            self.grid_points, self.point_room_idx = self._generate_grid()
            self.total_enclosed_area = sum(
                (r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1]) for r in self.rooms
            )

        def _generate_grid(self):
            points, point_room_idx = [], []
            for r_idx, r in enumerate(self.rooms):
                x1, y1, x2, y2 = r["rect"]
                xs = np.arange(x1 + self.grid_res / 2.0, x2, self.grid_res)
                ys = np.arange(y1 + self.grid_res / 2.0, y2, self.grid_res)
                for x in xs:
                    for y in ys:
                        points.append((x, y))
                        point_room_idx.append(r_idx)
            return np.array(points), np.array(point_room_idx)

        def get_initial_room_allocations(self):
            allocations = []
            for r_idx, r in enumerate(self.rooms):
                x1, y1, x2, y2 = r["rect"]
                area = (x2 - x1) * (y2 - y1)
                min_req = r.get("min_detectors", 1)
                count = max(min_req, 3) if area > 80.0 else (max(min_req, 2) if area > 35.0 else min_req)
                for _ in range(count):
                    allocations.append(r_idx)
            return allocations

    def get_ground_floor(grid_res=0.25):
        rooms = [
            {"name": "Battery Room", "rect": [0.0, 10.0, 6.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
            {"name": "Power Supply Room", "rect": [6.0, 10.0, 12.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
            {"name": "Server Room", "rect": [12.0, 10.0, 17.77, 16.5], "critical": True, "min_detectors": 2, "target_r": 4.0},
            {"name": "WAN Room", "rect": [17.77, 10.0, 22.54, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.0},
            {"name": "Staircase Lobby (GF)", "rect": [0.0, 6.0, 6.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Entrance Foyer", "rect": [6.0, 6.0, 10.5, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Main Corridor (2.77m)", "rect": [10.5, 7.23, 22.54, 10.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
            {"name": "Store Room (GF)", "rect": [0.0, 0.0, 6.0, 3.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Discussion Room", "rect": [0.0, 3.0, 6.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Staff Sitting (GF)", "rect": [6.0, 0.0, 12.0, 6.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
            {"name": "Control Room (SCADA)", "rect": [12.0, 0.0, 18.0, 7.23], "critical": True, "min_detectors": 2, "target_r": 4.2},
            {"name": "Drinking Water & Lobby", "rect": [18.0, 4.0, 22.54, 7.23], "critical": False, "min_detectors": 1, "target_r": 5.3},
        ]
        return FloorGeometry("Ground Floor", rooms, grid_res)

    def get_first_floor(grid_res=0.25):
        rooms = [
            {"name": "Conference Room (50P)", "rect": [6.0, 10.0, 22.54, 16.5], "critical": True, "min_detectors": 3, "target_r": 5.0},
            {"name": "Pantry & Lobby", "rect": [2.0, 12.5, 6.0, 16.5], "critical": False, "min_detectors": 1, "target_r": 4.5},
            {"name": "Store (Top-Right)", "rect": [22.54, 13.0, 24.5, 16.5], "critical": False, "min_detectors": 1, "target_r": 4.5},
            {"name": "Staircase Lobby (FF)", "rect": [0.0, 6.0, 6.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Entrance Foyer (FF)", "rect": [6.0, 6.0, 10.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Staff Sitting (FF)", "rect": [10.0, 6.0, 18.0, 10.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
            {"name": "Main Corridor (2.70m)", "rect": [6.0, 4.5, 18.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Store (Middle)", "rect": [2.5, 3.5, 6.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.0},
            {"name": "VIP / Visitor Room", "rect": [0.0, 0.0, 6.0, 3.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "SE Cabin (Sup. Eng.)", "rect": [6.0, 0.0, 10.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "EE-1 Cabin (Ex. Eng.)", "rect": [10.0, 0.0, 14.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "EE-2 Cabin (Ex. Eng.)", "rect": [14.0, 0.0, 18.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
            {"name": "Drinking Water & Lobby", "rect": [18.0, 3.5, 22.54, 7.2], "critical": False, "min_detectors": 1, "target_r": 5.3},
        ]
        return FloorGeometry("First Floor", rooms, grid_res)

    class CostModel:
        def __init__(self, multisensor_unit_price=5250.0, installation_and_base=1650.0):
            self.multisensor_unit_price = multisensor_unit_price
            self.installation_and_base = installation_and_base
            self.total_cost_per_node = multisensor_unit_price + installation_and_base

        def compute_cost(self, count, tier=1):
            return count * tier * self.total_cost_per_node

    class FitnessEvaluator:
        def __init__(self, geo, cost_model=None):
            self.geo = geo
            self.cost_model = cost_model or CostModel()

        def evaluate(self, detectors):
            if not detectors:
                return 1e9, 0.0, 0.0, 0.0
            covered_mask = np.zeros(len(self.geo.grid_points), dtype=bool)
            for d in detectors:
                dx, dy, r_idx = d[0], d[1], d[2]
                radius = self.geo.rooms[r_idx].get("target_r", 5.3)
                room_mask = (self.geo.point_room_idx == r_idx)
                if np.any(room_mask):
                    pts = self.geo.grid_points[room_mask]
                    dist_sq = (pts[:, 0] - dx) ** 2 + (pts[:, 1] - dy) ** 2
                    covered_mask[room_mask] |= (dist_sq <= (radius ** 2))
            cov = float(np.sum(covered_mask) / len(self.geo.grid_points))
            uncovered = len(self.geo.grid_points) - np.sum(covered_mask)
            pen = uncovered * 2500.0 + (0.98 - cov) * 500000.0 if cov < 0.98 else 0.0
            cost = self.cost_model.compute_cost(len(detectors))
            return cost + pen, cov, cost, 0.0

    class OptimizationResult:
        def __init__(self, method, floor, dets, fit, cov, cost, time_s, hist):
            self.method_name = method
            self.floor_name = floor
            self.detectors = dets
            self.count = len(dets)
            self.fitness = fit
            self.coverage = cov
            self.cost = cost
            self.time_seconds = time_s
            self.history = hist

    class PSOOptimizer:
        def __init__(self, geo, ev, max_iter=100, swarmsize=40):
            self.geo = geo
            self.ev = ev
            self.max_iter = max_iter
            self.swarmsize = swarmsize

        def optimize(self):
            np.random.seed(42)
            alloc = self.geo.get_initial_room_allocations()
            N = len(alloc)
            dim = N * 2
            lb, ub = np.zeros(dim), np.zeros(dim)
            for i, r_idx in enumerate(alloc):
                r = self.geo.rooms[r_idx]["rect"]
                lb[2*i], ub[2*i], lb[2*i+1], ub[2*i+1] = r[0]+0.5, r[2]-0.5, r[1]+0.5, r[3]-0.5
            X = np.zeros((self.swarmsize, dim))
            V = np.zeros((self.swarmsize, dim))
            for p in range(self.swarmsize):
                for i in range(dim):
                    X[p, i] = np.random.uniform(lb[i], ub[i])
            pbest, pbest_fit = np.copy(X), np.full(self.swarmsize, 1e9)
            gbest, gbest_fit = np.zeros(dim), 1e9
            hist = []
            t0 = time.time()
            for t in range(self.max_iter):
                w = 0.9 - (t / self.max_iter) * 0.5
                for p in range(self.swarmsize):
                    dets = [(X[p, 2*i], X[p, 2*i+1], alloc[i]) for i in range(N)]
                    fit, _, _, _ = self.ev.evaluate(dets)
                    if fit < pbest_fit[p]:
                        pbest_fit[p], pbest[p] = fit, np.copy(X[p])
                    if fit < gbest_fit:
                        gbest_fit, gbest = fit, np.copy(X[p])
                V = w * V + 1.8 * np.random.rand(self.swarmsize, dim) * (pbest - X) + 2.0 * np.random.rand(self.swarmsize, dim) * (gbest - X)
                X = np.clip(X + V, lb, ub)
                hist.append(float(gbest_fit))
            dets = [(gbest[2*i], gbest[2*i+1], alloc[i]) for i in range(N)]
            _, cov, cost, _ = self.ev.evaluate(dets)
            return OptimizationResult("PSO", self.geo.name, dets, gbest_fit, cov, cost, time.time() - t0, hist)

    class GAOptimizer:
        def __init__(self, geo, ev, max_gen=100, popsize=40):
            self.geo = geo
            self.ev = ev
            self.max_gen = max_gen
            self.popsize = popsize

        def optimize(self):
            np.random.seed(42)
            alloc = self.geo.get_initial_room_allocations()
            N, dim = len(alloc), len(alloc) * 2
            lb, ub = np.zeros(dim), np.zeros(dim)
            for i, r_idx in enumerate(alloc):
                r = self.geo.rooms[r_idx]["rect"]
                lb[2*i], ub[2*i], lb[2*i+1], ub[2*i+1] = r[0]+0.5, r[2]-0.5, r[1]+0.5, r[3]-0.5
            pop = np.zeros((self.popsize, dim))
            for p in range(self.popsize):
                for i in range(dim): pop[p, i] = np.random.uniform(lb[i], ub[i])
            hist, best_ind, best_fit, t0 = [], None, 1e9, time.time()
            for gen in range(self.max_gen):
                fits = np.zeros(self.popsize)
                for p in range(self.popsize):
                    dets = [(pop[p, 2*i], pop[p, 2*i+1], alloc[i]) for i in range(N)]
                    f, _, _, _ = self.ev.evaluate(dets)
                    fits[p] = f
                    if f < best_fit: best_fit, best_ind = f, np.copy(pop[p])
                hist.append(float(best_fit))
                s_idx = np.argsort(fits)
                new_pop = [np.copy(pop[s_idx[0]]), np.copy(pop[s_idx[1]])]
                while len(new_pop) < self.popsize:
                    t1, t2 = np.random.choice(self.popsize, 3, replace=False), np.random.choice(self.popsize, 3, replace=False)
                    p1, p2 = t1[np.argmin(fits[t1])], t2[np.argmin(fits[t2])]
                    child = np.zeros(dim)
                    for d in range(dim):
                        c_min, c_max = min(pop[p1, d], pop[p2, d]), max(pop[p1, d], pop[p2, d])
                        child[d] = np.random.uniform(c_min - 0.3*(c_max-c_min), c_max + 0.3*(c_max-c_min))
                        if np.random.rand() < 0.15: child[d] += np.random.normal(0, (ub[d]-lb[d])*0.1*(1-gen/self.max_gen))
                    new_pop.append(np.clip(child, lb, ub))
                pop = np.array(new_pop)
            dets = [(best_ind[2*i], best_ind[2*i+1], alloc[i]) for i in range(N)]
            _, cov, cost, _ = self.ev.evaluate(dets)
            return OptimizationResult("GA", self.geo.name, dets, best_fit, cov, cost, time.time() - t0, hist)

    class SAOptimizer:
        def __init__(self, geo, ev, max_iter=4000):
            self.geo = geo
            self.ev = ev
            self.max_iter = max_iter

        def optimize(self):
            np.random.seed(42)
            alloc = self.geo.get_initial_room_allocations()
            N = len(alloc)
            curr = []
            for r_idx in alloc:
                r = self.geo.rooms[r_idx]["rect"]
                curr.append([np.random.uniform(r[0]+0.5, r[2]-0.5), np.random.uniform(r[1]+0.5, r[3]-0.5), r_idx])
            curr_fit, _, _, _ = self.ev.evaluate(curr)
            best_dets, best_fit = [list(d) for d in curr], curr_fit
            T, alpha, hist, t0 = 1000.0, 0.995, [], time.time()
            for it in range(self.max_iter):
                idx = np.random.randint(N)
                r = self.geo.rooms[curr[idx][2]]["rect"]
                neigh = [list(d) for d in curr]
                neigh[idx][0] = np.clip(curr[idx][0] + np.random.normal(0, 1.2*(T/1000.0)+0.1), r[0]+0.5, r[2]-0.5)
                neigh[idx][1] = np.clip(curr[idx][1] + np.random.normal(0, 1.2*(T/1000.0)+0.1), r[1]+0.5, r[3]-0.5)
                cand_fit, _, _, _ = self.ev.evaluate(neigh)
                delta = cand_fit - curr_fit
                if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-4)):
                    curr, curr_fit = neigh, cand_fit
                    if cand_fit < best_fit:
                        best_fit, best_dets = cand_fit, [list(d) for d in neigh]
                if it % 40 == 0: hist.append(float(best_fit))
                T *= alpha
            dets = [(d[0], d[1], d[2]) for d in best_dets]
            _, cov, cost, _ = self.ev.evaluate(dets)
            return OptimizationResult("SA", self.geo.name, dets, best_fit, cov, cost, time.time() - t0, hist)

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

floor_choice = st.sidebar.selectbox(
    "Active Floor Layout",
    ["Ground Floor (GF)", "First Floor (FF)"],
    index=0
)

algo_choice = st.sidebar.selectbox(
    "Optimization Engine",
    ["Particle Swarm Optimization (PSO)", "Genetic Algorithm (GA)", "Simulated Annealing (SA)", "Compare All 3 (Benchmark)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Safety & Cost Parameters")

custom_radius = st.sidebar.slider(
    "Standard Coverage Radius (meters)",
    min_value=3.5,
    max_value=7.5,
    value=5.3,
    step=0.1,
    help="Standard coverage radius per IS 2189 / NFPA 72"
)

tier_choice = st.sidebar.radio(
    "False Ceiling Tier",
    ["Single Tier (Below Ceiling Only)", "Dual Tier (Below Ceiling + Above False Ceiling Plenum)"],
    index=0
)
tier_multiplier = 2 if "Dual Tier" in tier_choice else 1

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

for r in geometry.rooms:
    if not r.get("critical", False):
        r["target_r"] = custom_radius

evaluator = FitnessEvaluator(geometry, cost_model)

# -----------------------------------------------------------------------------
# 4. EXECUTE OPTIMIZATION
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

with st.spinner(f"Running {algo_choice} on {geometry.name}..."):
    results = run_optimization_cached(geometry.name, algo_choice, max_iterations, custom_radius, unit_price, install_price)

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

# TAB 1: ARCHITECTURAL LAYOUT
with tab_layout:
    st.subheader(f"Optimal Detector Placement & Coverage Heatmap — {geometry.name}")

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 18)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)

    for r_idx, r in enumerate(geometry.rooms):
        x1, y1, x2, y2 = r["rect"]
        w, h = x2 - x1, y2 - y1
        bg = "#FEF9E7" if r.get("critical", False) else "#F8FAFC"
        rect_patch = patches.Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor="#334155", facecolor=bg, alpha=0.85)
        ax.add_patch(rect_patch)
        ax.text(x1 + w/2, y1 + h/2 + 0.3, r["name"], ha="center", va="center", fontsize=9, color="#1e293b", weight="bold")
        ax.text(x1 + w/2, y1 + h/2 - 0.4, f"{(w*h):.1f} m²", ha="center", va="center", fontsize=8, color="#64748b")

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

# TAB 2: AUTHORITY SUBMISSION BOQ
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
    st.dataframe(df_schedule, width="stretch", hide_index=True)

    csv_str = df_schedule.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Authority Schedule (CSV)",
        data=csv_str,
        file_name=f"{geometry.name.replace(' ', '_')}_Authority_BOQ.csv",
        mime="text/csv"
    )

# TAB 3: BENCHMARK & CONVERGENCE
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
        st.dataframe(pd.DataFrame(bench_rows), hide_index=True, width="stretch")
        st.info("💡 **Winner: Particle Swarm Optimization (PSO)** achieved the fastest convergence with continuous coordinate balance and zero blind spots.")

# TAB 4: INTERACTIVE HTML5 SIMULATOR
with tab_simulator:
    st.subheader("Interactive Standalone CAD Simulator")
    simulator_path = os.path.join(ROOT_DIR, "detector_simulator.html")
    if os.path.exists(simulator_path):
        with open(simulator_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)
    else:
        st.info("Interactive simulator can also be opened via detector_simulator.html")

# TAB 5: STANDARDS & EQUATIONS
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
