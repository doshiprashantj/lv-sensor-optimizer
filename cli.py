"""
Command-Line Interface (CLI) for running multisensor smoke detector layout optimizations.
"""

import argparse
import os
import sys
from .geometry import get_ground_floor, get_first_floor
from .cost_model import CostModel
from .fitness import FitnessEvaluator
from .algorithms.pso import PSOOptimizer
from .algorithms.ga import GAOptimizer
from .algorithms.sa import SAOptimizer
from .visualizer import plot_floor_optimization
from .exporter import export_authority_schedule_csv, export_summary_json


def main():
    parser = argparse.ArgumentParser(
        description="LV Multisensor Smoke Detector Layout Optimizer (PSO, GA, SA) - IS 2189 / NFPA 72"
    )
    parser.add_argument(
        "--floor",
        choices=["GF", "FF", "BOTH"],
        default="BOTH",
        help="Floor to optimize (GF = Ground Floor, FF = First Floor, BOTH = All floors)"
    )
    parser.add_argument(
        "--algo",
        choices=["PSO", "GA", "SA", "ALL"],
        default="ALL",
        help="Optimization algorithm to run"
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory for plots, CSV schedules, and JSON summaries"
    )
    parser.add_argument(
        "--grid-res",
        type=float,
        default=0.25,
        help="Discretization grid resolution in meters (default: 0.25m)"
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    floors = []
    if args.floor in ["GF", "BOTH"]:
        floors.append(get_ground_floor(args.grid_res))
    if args.floor in ["FF", "BOTH"]:
        floors.append(get_first_floor(args.grid_res))

    cost_model = CostModel()
    all_summary = {}

    print("=" * 78)
    print("LV MULTISENSOR SMOKE DETECTOR LAYOUT OPTIMIZER (IS 2189 / NFPA 72)")
    print("=" * 78)

    for floor in floors:
        print(f"\n>>> Running Optimization for {floor.name.upper()} ({len(floor.grid_points)} grid points)...")
        evaluator = FitnessEvaluator(floor, cost_model)

        results = {}
        if args.algo in ["PSO", "ALL"]:
            print("  -> Executing Particle Swarm Optimization (PSO)...")
            pso = PSOOptimizer(floor, evaluator)
            results["PSO"] = pso.optimize()

        if args.algo in ["GA", "ALL"]:
            print("  -> Executing Genetic Algorithm (GA)...")
            ga = GAOptimizer(floor, evaluator)
            results["GA"] = ga.optimize()

        if args.algo in ["SA", "ALL"]:
            print("  -> Executing Simulated Annealing (SA)...")
            sa = SAOptimizer(floor, evaluator)
            results["SA"] = sa.optimize()

        # Save plot
        plot_path = os.path.join(args.outdir, f"{floor.name.replace(' ', '_')}_Optimization.png")
        plot_floor_optimization(floor, results, save_path=plot_path)
        print(f"  [SAVED] Comparison plot: {plot_path}")

        # Save CSV for best method (PSO)
        best_res = results.get("PSO") or list(results.values())[0]
        csv_path = os.path.join(args.outdir, f"{floor.name.replace(' ', '_')}_Authority_Schedule.csv")
        export_authority_schedule_csv(floor, best_res, csv_path, cost_model)
        print(f"  [SAVED] Authority Submission CSV: {csv_path}")

        all_summary[floor.name] = {k: v.to_dict() for k, v in results.items()}

    # Save JSON summary
    json_path = os.path.join(args.outdir, "optimization_summary.json")
    export_summary_json(all_summary, json_path)
    print(f"\n[SAVED] Complete JSON summary: {json_path}")

    # Print Table
    print("\n" + "=" * 78)
    print(f"{'Floor Level':<14} | {'Method':<6} | {'Detectors':<10} | {'Coverage %':<12} | {'Cost (INR)':<14} | {'Time (s)':<10}")
    print("-" * 78)
    for fl_name, methods in all_summary.items():
        for m_name, d in methods.items():
            print(f"{fl_name:<14} | {m_name:<6} | {d['count']:<10} | {d['coverage']*100:<11.2f}% | Rs. {d['cost']:<10,.0f} | {d['time_seconds']:<9.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
