"""
Multisensor Smoke Detector Layout Optimization Engine for Ground & First Floors
Using Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Simulated Annealing (SA)
Compliant with IS 2189 / NFPA 72 / EN 54 Standards
Author: LV System Design Engineering Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import os

# ==============================================================================
# 1. BUILDING GEOMETRY DEFINITION (GROUND FLOOR & FIRST FLOOR)
# ==============================================================================

# Rates from LV Tender Estimate (SOR Item 3.4 & 3.2)
COST_MULTISENSOR_UNIT = 5250.0  # INR per detector
COST_INSTALLATION_BASE = 1650.0 # INR for base, 2x1.5 sqmm FRLS cable, conduit & RI
COST_TOTAL_PER_DETECTOR = COST_MULTISENSOR_UNIT + COST_INSTALLATION_BASE # 6,900 INR

# Coverage Radii (m) based on IS 2189 / NFPA 72
# Standard open area coverage radius R = 5.5m (up to ~65-75 m2 with smooth ceiling)
# High-risk / dense server/battery/power rooms: R = 4.0m - 4.5m for faster detection
DEFAULT_DETECTOR_RADIUS = 5.3  # Standard coverage radius in meters
CRITICAL_DETECTOR_RADIUS = 4.2 # Critical room coverage radius

ROOMS_GF = [
    {"name": "Battery Room", "rect": [0.0, 10.0, 6.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
    {"name": "Power Supply Room", "rect": [6.0, 10.0, 12.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
    {"name": "Server Room", "rect": [12.0, 10.0, 17.77, 16.5], "critical": True, "min_detectors": 2, "target_r": 4.0},
    {"name": "WAN Room", "rect": [17.77, 10.0, 22.54, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.0},
    {"name": "Staircase Lobby (GF)", "rect": [0.0, 6.0, 6.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
    {"name": "Entrance Foyer", "rect": [6.0, 6.0, 10.5, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
    {"name": "Main Corridor (2.77m)", "rect": [10.5, 7.23, 22.54, 10.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
    {"name": "Store Room (GF)", "rect": [0.0, 0.0, 6.0, 3.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
    {"name": "Discussion Room", "rect": [0.0, 3.0, 6.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
    {"name": "Staff Sitting (GF)", "rect": [6.0, 0.0, 12.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
    {"name": "Control Room (SCADA)", "rect": [12.0, 0.0, 18.0, 7.23], "critical": True, "min_detectors": 2, "target_r": 4.2},
    {"name": "Drinking Water & Lobby", "rect": [18.0, 4.0, 22.54, 7.23], "critical": False, "min_detectors": 1, "target_r": 5.3},
]

ROOMS_FF = [
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

# Discretize floor domain into a high-resolution grid for accurate ray-traced / room-bounded coverage
def generate_floor_grid(rooms, grid_res=0.25):
    points = []
    point_room_idx = []
    for r_idx, r in enumerate(rooms):
        x1, y1, x2, y2 = r["rect"]
        xs = np.arange(x1 + grid_res/2, x2, grid_res)
        ys = np.arange(y1 + grid_res/2, y2, grid_res)
        for x in xs:
            for y in ys:
                points.append((x, y))
                point_room_idx.append(r_idx)
    return np.array(points), np.array(point_room_idx)

# ==============================================================================
# 2. EVALUATION / FITNESS FUNCTION
# ==============================================================================

def evaluate_placement(detectors, rooms, grid_points, point_room_idx):
    """
    detectors: list of (x, y, room_idx) or (x, y)
    Computes:
      - coverage_ratio (percentage of grid points within sensor radius inside room)
      - total_cost (detectors count * unit cost)
      - overlap_penalty
      - fitness value to minimize
    """
    if len(detectors) == 0:
        return 1e9, 0.0, 0.0, 0.0
    
    N = len(detectors)
    det_arr = np.array([[d[0], d[1]] for d in detectors])
    
    # Point-to-detector distance check with room-bounding constraint (smoke compartmentalization)
    covered_mask = np.zeros(len(grid_points), dtype=bool)
    
    for i, det in enumerate(detectors):
        dx, dy, det_r_idx = det[0], det[1], det[2]
        r_info = rooms[det_r_idx]
        radius = r_info.get("target_r", DEFAULT_DETECTOR_RADIUS)
        
        # Grid points belonging to the same room (or adjacent open corridor if configured)
        room_pts_mask = (point_room_idx == det_r_idx)
        if np.any(room_pts_mask):
            pts = grid_points[room_pts_mask]
            dists_sq = (pts[:, 0] - dx)**2 + (pts[:, 1] - dy)**2
            in_range = dists_sq <= (radius ** 2)
            covered_mask[room_pts_mask] = covered_mask[room_pts_mask] | in_range

    coverage_ratio = np.sum(covered_mask) / len(grid_points)
    
    # Check minimum detector constraints per room
    room_detector_counts = [0] * len(rooms)
    for det in detectors:
        room_detector_counts[det[2]] += 1
    
    min_det_penalty = 0.0
    for r_idx, r in enumerate(rooms):
        req = r.get("min_detectors", 1)
        if room_detector_counts[r_idx] < req:
            min_det_penalty += (req - room_detector_counts[r_idx]) * 50000.0
    
    # Calculate overlap between sensors in the same room
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if detectors[i][2] == detectors[j][2]: # same room
                dist = np.sqrt((detectors[i][0] - detectors[j][0])**2 + (detectors[i][1] - detectors[j][1])**2)
                rad = rooms[detectors[i][2]].get("target_r", DEFAULT_DETECTOR_RADIUS)
                if dist < rad * 0.75: # excessive redundancy
                    overlap_penalty += (rad * 0.75 - dist) * 1000.0
    
    # Fitness Objective: Min Cost + Heavy penalty for uncovered points + Overlap penalty
    uncovered_points = len(grid_points) - np.sum(covered_mask)
    coverage_penalty = uncovered_points * 2500.0 if uncovered_points > 0 else 0.0
    
    # If coverage < 98%, impose severe penalty
    if coverage_ratio < 0.98:
        coverage_penalty += (0.98 - coverage_ratio) * 500000.0

    total_cost = N * COST_TOTAL_PER_DETECTOR
    fitness = total_cost + coverage_penalty + min_det_penalty + overlap_penalty
    
    return fitness, coverage_ratio, total_cost, overlap_penalty

# ==============================================================================
# 3. ALGORITHM 1: PARTICLE SWARM OPTIMIZATION (PSO)
# ==============================================================================

def optimize_pso(rooms, grid_points, point_room_idx, max_iter=120, swarmsize=40):
    """
    PSO for optimal detector layout.
    Particles explore continuous coordinates per room assignment.
    """
    np.random.seed(42)
    # Determine detector allocation per room based on area and min requirements
    allocations = []
    for r_idx, r in enumerate(rooms):
        x1, y1, x2, y2 = r["rect"]
        area = (x2 - x1) * (y2 - y1)
        req = r.get("min_detectors", 1)
        if area > 80.0:
            count = max(req, 3)
        elif area > 35.0:
            count = max(req, 2)
        else:
            count = req
        for _ in range(count):
            allocations.append(r_idx)
            
    num_detectors = len(allocations)
    dim = num_detectors * 2 # (x, y) for each detector
    
    # Lower & upper bounds per detector
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i, r_idx in enumerate(allocations):
        rect = rooms[r_idx]["rect"]
        lb[2*i] = rect[0] + 0.5
        ub[2*i] = rect[2] - 0.5
        lb[2*i+1] = rect[1] + 0.5
        ub[2*i+1] = rect[3] - 0.5
        
    # Initialize particles
    X = np.zeros((swarmsize, dim))
    V = np.zeros((swarmsize, dim))
    for p in range(swarmsize):
        for i, r_idx in enumerate(allocations):
            rect = rooms[r_idx]["rect"]
            X[p, 2*i] = np.random.uniform(lb[2*i], ub[2*i])
            X[p, 2*i+1] = np.random.uniform(lb[2*i+1], ub[2*i+1])
            V[p, 2*i] = np.random.uniform(-(ub[2*i]-lb[2*i])*0.1, (ub[2*i]-lb[2*i])*0.1)
            V[p, 2*i+1] = np.random.uniform(-(ub[2*i+1]-lb[2*i+1])*0.1, (ub[2*i+1]-lb[2*i+1])*0.1)
            
    pbest = np.copy(X)
    pbest_fit = np.full(swarmsize, 1e9)
    gbest = np.zeros(dim)
    gbest_fit = 1e9
    gbest_cov = 0.0
    
    history = []
    start_time = time.time()
    
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.8, 2.0
    
    for t in range(max_iter):
        w = w_max - (t / max_iter) * (w_max - w_min)
        for p in range(swarmsize):
            # Form detector list
            dets = []
            for i, r_idx in enumerate(allocations):
                dets.append((X[p, 2*i], X[p, 2*i+1], r_idx))
            
            fit, cov, cost, _ = evaluate_placement(dets, rooms, grid_points, point_room_idx)
            
            if fit < pbest_fit[p]:
                pbest_fit[p] = fit
                pbest[p] = np.copy(X[p])
                
            if fit < gbest_fit:
                gbest_fit = fit
                gbest = np.copy(X[p])
                gbest_cov = cov
                
        # Velocity & Position Update
        r1 = np.random.rand(swarmsize, dim)
        r2 = np.random.rand(swarmsize, dim)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        
        # Boundary clipping
        for i in range(dim):
            X[:, i] = np.clip(X[:, i], lb[i], ub[i])
            
        history.append(gbest_fit)
        
    elapsed = time.time() - start_time
    
    # Extract final best detectors
    best_dets = []
    for i, r_idx in enumerate(allocations):
        best_dets.append((gbest[2*i], gbest[2*i+1], r_idx))
        
    _, final_cov, final_cost, _ = evaluate_placement(best_dets, rooms, grid_points, point_room_idx)
    
    return {
        "method": "PSO",
        "detectors": best_dets,
        "count": len(best_dets),
        "fitness": gbest_fit,
        "coverage": final_cov,
        "cost": final_cost,
        "time": elapsed,
        "history": history
    }

# ==============================================================================
# 4. ALGORITHM 2: GENETIC ALGORITHM (GA)
# ==============================================================================

def optimize_ga(rooms, grid_points, point_room_idx, max_gen=120, popsize=40, mutation_rate=0.15):
    """
    GA with Elitism, Tournament Selection, Blend Crossover, and Adaptive Gaussian Mutation.
    """
    np.random.seed(42)
    allocations = []
    for r_idx, r in enumerate(rooms):
        x1, y1, x2, y2 = r["rect"]
        area = (x2 - x1) * (y2 - y1)
        req = r.get("min_detectors", 1)
        if area > 80.0:
            count = max(req, 3)
        elif area > 35.0:
            count = max(req, 2)
        else:
            count = req
        for _ in range(count):
            allocations.append(r_idx)
            
    num_detectors = len(allocations)
    dim = num_detectors * 2
    
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i, r_idx in enumerate(allocations):
        rect = rooms[r_idx]["rect"]
        lb[2*i] = rect[0] + 0.5
        ub[2*i] = rect[2] - 0.5
        lb[2*i+1] = rect[1] + 0.5
        ub[2*i+1] = rect[3] - 0.5
        
    # Initialize Population
    population = np.zeros((popsize, dim))
    for p in range(popsize):
        for i in range(dim):
            population[p, i] = np.random.uniform(lb[i], ub[i])
            
    history = []
    best_individual = None
    best_fitness = 1e9
    start_time = time.time()
    
    for gen in range(max_gen):
        fitnesses = np.zeros(popsize)
        for p in range(popsize):
            dets = [(population[p, 2*i], population[p, 2*i+1], allocations[i]) for i in range(num_detectors)]
            fit, cov, cost, _ = evaluate_placement(dets, rooms, grid_points, point_room_idx)
            fitnesses[p] = fit
            if fit < best_fitness:
                best_fitness = fit
                best_individual = np.copy(population[p])
                
        history.append(best_fitness)
        
        # Elitism: keep top 2
        sorted_indices = np.argsort(fitnesses)
        new_population = [np.copy(population[sorted_indices[0]]), np.copy(population[sorted_indices[1]])]
        
        # Reproduction
        while len(new_population) < popsize:
            # Tournament selection
            t1, t2 = np.random.choice(popsize, 3, replace=False), np.random.choice(popsize, 3, replace=False)
            p1 = t1[np.argmin(fitnesses[t1])]
            p2 = t2[np.argmin(fitnesses[t2])]
            
            # Blend Crossover (BLX-alpha)
            alpha = 0.3
            child = np.zeros(dim)
            for d in range(dim):
                c_min = min(population[p1, d], population[p2, d])
                c_max = max(population[p1, d], population[p2, d])
                rng = c_max - c_min
                child[d] = np.random.uniform(c_min - alpha*rng, c_max + alpha*rng)
                
            # Mutation
            for d in range(dim):
                if np.random.rand() < mutation_rate:
                    sigma = (ub[d] - lb[d]) * 0.1 * (1.0 - gen / max_gen)
                    child[d] += np.random.normal(0, max(sigma, 0.05))
                    
            # Clip
            child = np.clip(child, lb, ub)
            new_population.append(child)
            
        population = np.array(new_population)
        
    elapsed = time.time() - start_time
    best_dets = [(best_individual[2*i], best_individual[2*i+1], allocations[i]) for i in range(num_detectors)]
    _, final_cov, final_cost, _ = evaluate_placement(best_dets, rooms, grid_points, point_room_idx)
    
    return {
        "method": "GA",
        "detectors": best_dets,
        "count": len(best_dets),
        "fitness": best_fitness,
        "coverage": final_cov,
        "cost": final_cost,
        "time": elapsed,
        "history": history
    }

# ==============================================================================
# 5. ALGORITHM 3: SIMULATED ANNEALING (SA)
# ==============================================================================

def optimize_sa(rooms, grid_points, point_room_idx, max_iter=4800, T_init=1000.0, alpha=0.995):
    """
    Simulated Annealing with Boltzmann Acceptance & Adaptive Neighborhood Search.
    """
    np.random.seed(42)
    allocations = []
    for r_idx, r in enumerate(rooms):
        x1, y1, x2, y2 = r["rect"]
        area = (x2 - x1) * (y2 - y1)
        req = r.get("min_detectors", 1)
        if area > 80.0:
            count = max(req, 3)
        elif area > 35.0:
            count = max(req, 2)
        else:
            count = req
        for _ in range(count):
            allocations.append(r_idx)
            
    num_detectors = len(allocations)
    
    # Initialize state
    current_dets = []
    for r_idx in allocations:
        rect = rooms[r_idx]["rect"]
        x = np.random.uniform(rect[0] + 0.5, rect[2] - 0.5)
        y = np.random.uniform(rect[1] + 0.5, rect[3] - 0.5)
        current_dets.append([x, y, r_idx])
        
    current_fit, current_cov, _, _ = evaluate_placement(current_dets, rooms, grid_points, point_room_idx)
    
    best_dets = [list(d) for d in current_dets]
    best_fit = current_fit
    
    history = []
    T = T_init
    start_time = time.time()
    
    for it in range(max_iter):
        # Pick a random sensor to perturb
        idx = np.random.randint(num_detectors)
        r_idx = current_dets[idx][2]
        rect = rooms[r_idx]["rect"]
        
        step_size = 1.2 * (T / T_init) + 0.1
        new_x = np.clip(current_dets[idx][0] + np.random.normal(0, step_size), rect[0] + 0.5, rect[2] - 0.5)
        new_y = np.clip(current_dets[idx][1] + np.random.normal(0, step_size), rect[1] + 0.5, rect[3] - 0.5)
        
        neighbor_dets = [list(d) for d in current_dets]
        neighbor_dets[idx][0] = new_x
        neighbor_dets[idx][1] = new_y
        
        cand_fit, cand_cov, _, _ = evaluate_placement(neighbor_dets, rooms, grid_points, point_room_idx)
        
        delta = cand_fit - current_fit
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-4)):
            current_dets = neighbor_dets
            current_fit = cand_fit
            if cand_fit < best_fit:
                best_fit = cand_fit
                best_dets = [list(d) for d in neighbor_dets]
                
        if it % 40 == 0:
            history.append(best_fit)
            
        T *= alpha
        
    elapsed = time.time() - start_time
    _, final_cov, final_cost, _ = evaluate_placement(best_dets, rooms, grid_points, point_room_idx)
    
    return {
        "method": "SA",
        "detectors": best_dets,
        "count": len(best_dets),
        "fitness": best_fit,
        "coverage": final_cov,
        "cost": final_cost,
        "time": elapsed,
        "history": history
    }

# ==============================================================================
# 6. VISUALIZATION & REPORT GENERATION
# ==============================================================================

def plot_layout_optimization(floor_name, rooms, grid_points, point_room_idx, results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Multisensor Smoke Detector Optimization - {floor_name}\n(UGVCL SCADA Centre Building - LV System Design)", fontsize=16, fontweight='bold')
    
    methods = [("PSO", results["PSO"], axes[0, 0]), 
               ("GA", results["GA"], axes[0, 1]), 
               ("SA", results["SA"], axes[1, 0])]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(rooms)))
    
    for name, res, ax in methods:
        ax.set_title(f"{name} Method (Detectors: {res['count']} | Coverage: {res['coverage']*100:.1f}% | Cost: ₹{res['cost']:,.0f})", fontsize=12, fontweight='bold')
        ax.set_xlim(-1, 26)
        ax.set_ylim(-1, 18)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Plot rooms
        for r_idx, r in enumerate(rooms):
            x1, y1, x2, y2 = r["rect"]
            w, h = x2 - x1, y2 - y1
            rect_patch = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='#2C3E50', facecolor='#ECF0F1', alpha=0.6)
            ax.add_patch(rect_patch)
            ax.text(x1 + w/2, y1 + h/2, r["name"], ha='center', va='center', fontsize=7, color='#2C3E50', wrap=True, weight='bold')
            
        # Plot detector coverage circles & points
        for det in res["detectors"]:
            dx, dy, det_r_idx = det[0], det[1], det[2]
            rad = rooms[det_r_idx].get("target_r", DEFAULT_DETECTOR_RADIUS)
            cov_circle = patches.Circle((dx, dy), rad, color='#E74C3C', alpha=0.18, edgecolor='#C0392B', linewidth=1.2, linestyle='--')
            ax.add_patch(cov_circle)
            ax.plot(dx, dy, marker='o', markersize=7, markerfacecolor='#E74C3C', markeredgecolor='#7B241C', markeredgewidth=1.5)
            # Response indicator symbol
            ax.plot(dx, dy, marker='+', markersize=5, color='white', markeredgewidth=1.5)
            
        ax.set_xlabel("Width (meters)")
        ax.set_ylabel("Depth (meters)")
        
    # Plot Convergence Comparison in 4th subplot
    ax_conv = axes[1, 1]
    ax_conv.set_title("Optimization Convergence Rate (Fitness vs Iterations)", fontsize=12, fontweight='bold')
    
    # Normalize iterations to percentage
    for m_name, color in [("PSO", "#2980B9"), ("GA", "#27AE60"), ("SA", "#8E44AD")]:
        hist = results[m_name]["history"]
        xs = np.linspace(0, 100, len(hist))
        ax_conv.plot(xs, hist, label=f"{m_name} (Best: ₹{results[m_name]['cost']:,.0f}, Cov: {results[m_name]['coverage']*100:.1f}%)", color=color, linewidth=2)
        
    ax_conv.set_xlabel("Optimization Progress (%)")
    ax_conv.set_ylabel("Objective Fitness (Cost + Penalties)")
    ax_conv.grid(True, linestyle='--', alpha=0.6)
    ax_conv.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layout plot to {save_path}")

# ==============================================================================
# 7. MAIN EXECUTION & BENCHMARKING
# ==============================================================================

def run_benchmarks():
    print("==================================================================")
    print("MULTISENSOR SMOKE DETECTOR OPTIMIZATION BENCHMARK (PSO vs GA vs SA)")
    print("Standards: IS 2189 / NFPA 72 / EN 54 | Project: UGVCL SCADA Centre")
    print("==================================================================")
    
    os.makedirs("results", exist_ok=True)
    
    # --- GROUND FLOOR ---
    print("\n>>> Running Optimization for GROUND FLOOR (GF)...")
    gf_points, gf_room_idx = generate_floor_grid(ROOMS_GF)
    print(f"Total GF Discretized Grid Points: {len(gf_points)} (resolution: 0.25m)")
    
    gf_pso = optimize_pso(ROOMS_GF, gf_points, gf_room_idx)
    gf_ga  = optimize_ga(ROOMS_GF, gf_points, gf_room_idx)
    gf_sa  = optimize_sa(ROOMS_GF, gf_points, gf_room_idx)
    
    gf_results = {"PSO": gf_pso, "GA": gf_ga, "SA": gf_sa}
    plot_layout_optimization("Ground Floor", ROOMS_GF, gf_points, gf_room_idx, gf_results, "results/GF_Optimization_Comparison.png")
    
    # --- FIRST FLOOR ---
    print("\n>>> Running Optimization for FIRST FLOOR (FF)...")
    ff_points, ff_room_idx = generate_floor_grid(ROOMS_FF)
    print(f"Total FF Discretized Grid Points: {len(ff_points)} (resolution: 0.25m)")
    
    ff_pso = optimize_pso(ROOMS_FF, ff_points, ff_room_idx)
    ff_ga  = optimize_ga(ROOMS_FF, ff_points, ff_room_idx)
    ff_sa  = optimize_sa(ROOMS_FF, ff_points, ff_room_idx)
    
    ff_results = {"PSO": ff_pso, "GA": ff_ga, "SA": ff_sa}
    plot_layout_optimization("First Floor", ROOMS_FF, ff_points, ff_room_idx, ff_results, "results/FF_Optimization_Comparison.png")
    
    # Export summary JSON
    summary_data = {
        "unit_rates": {
            "sensor_inr": COST_MULTISENSOR_UNIT,
            "install_inr": COST_INSTALLATION_BASE,
            "total_node_inr": COST_TOTAL_PER_DETECTOR
        },
        "ground_floor": {
            "rooms": ROOMS_GF,
            "PSO": {"count": gf_pso["count"], "coverage": gf_pso["coverage"], "cost": gf_pso["cost"], "time": gf_pso["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in gf_pso["detectors"]]},
            "GA":  {"count": gf_ga["count"], "coverage": gf_ga["coverage"], "cost": gf_ga["cost"], "time": gf_ga["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in gf_ga["detectors"]]},
            "SA":  {"count": gf_sa["count"], "coverage": gf_sa["coverage"], "cost": gf_sa["cost"], "time": gf_sa["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in gf_sa["detectors"]]}
        },
        "first_floor": {
            "rooms": ROOMS_FF,
            "PSO": {"count": ff_pso["count"], "coverage": ff_pso["coverage"], "cost": ff_pso["cost"], "time": ff_pso["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in ff_pso["detectors"]]},
            "GA":  {"count": ff_ga["count"], "coverage": ff_ga["coverage"], "cost": ff_ga["cost"], "time": ff_ga["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in ff_ga["detectors"]]},
            "SA":  {"count": ff_sa["count"], "coverage": ff_sa["coverage"], "cost": ff_sa["cost"], "time": ff_sa["time"], "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in ff_sa["detectors"]]}
        }
    }
    
    with open("results/optimization_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print("\n======================= BENCHMARK SUMMARY =======================")
    print(f"{'Floor':<14} | {'Method':<6} | {'Detectors':<10} | {'Coverage %':<12} | {'Cost (INR)':<14} | {'Exec Time (s)':<12}")
    print("-" * 76)
    for fl_name, res in [("Ground Floor", gf_results), ("First Floor", ff_results)]:
        for m in ["PSO", "GA", "SA"]:
            print(f"{fl_name:<14} | {m:<6} | {res[m]['count']:<10} | {res[m]['coverage']*100:<11.2f}% | Rs. {res[m]['cost']:<10,.0f} | {res[m]['time']:<11.2f}s")
    print("==================================================================")

if __name__ == "__main__":
    run_benchmarks()
