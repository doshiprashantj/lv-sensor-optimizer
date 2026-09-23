"""
Publication-quality plotting and visualization routines for detector layouts and convergence curves.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .geometry import FloorGeometry
from .algorithms.base import OptimizationResult


def plot_floor_optimization(
    geometry: FloorGeometry,
    results_by_method: Dict[str, OptimizationResult],
    save_path: str = None,
    show: bool = False
):
    """
    Plots a 4-panel comparison figure showing PSO, GA, and SA detector placements and convergence curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"Multisensor Smoke Detector Optimization - {geometry.name}\n"
        f"(UGVCL SCADA Centre Building - LV System Design - IS 2189 / NFPA 72)",
        fontsize=15,
        fontweight="bold"
    )

    panels = [
        ("PSO", results_by_method.get("PSO"), axes[0, 0]),
        ("GA", results_by_method.get("GA"), axes[0, 1]),
        ("SA", results_by_method.get("SA"), axes[1, 0])
    ]

    for name, res, ax in panels:
        if res is None:
            continue
        ax.set_title(
            f"{name} Method (Detectors: {res.count} | Coverage: {res.coverage*100:.1f}% | Cost: ₹{res.cost:,.0f})",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlim(-1, 26)
        ax.set_ylim(-1, 18)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Plot rooms
        for r_idx, r in enumerate(geometry.rooms):
            x1, y1, x2, y2 = r["rect"]
            w, h = x2 - x1, y2 - y1
            bg_col = "#FEF9E7" if r.get("critical", False) else "#ECF0F1"
            rect_patch = patches.Rectangle(
                (x1, y1), w, h, linewidth=1.5, edgecolor="#2C3E50", facecolor=bg_col, alpha=0.7
            )
            ax.add_patch(rect_patch)
            ax.text(
                x1 + w / 2.0,
                y1 + h / 2.0,
                r["name"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#2C3E50",
                wrap=True,
                weight="bold"
            )

        # Plot detector coverage circles
        for d in res.detectors:
            dx, dy, r_idx = d[0], d[1], d[2]
            rad = geometry.rooms[r_idx].get("target_r", 5.3)
            cov_circle = patches.Circle(
                (dx, dy),
                rad,
                facecolor="#E74C3C",
                edgecolor="#C0392B",
                alpha=0.16,
                linewidth=1.2,
                linestyle="--"
            )
            ax.add_patch(cov_circle)
            ax.plot(
                dx,
                dy,
                marker="o",
                markersize=7,
                markerfacecolor="#E74C3C",
                markeredgecolor="#7B241C",
                markeredgewidth=1.5
            )
            ax.plot(dx, dy, marker="+", markersize=5, color="white", markeredgewidth=1.5)

        ax.set_xlabel("Width (meters)")
        ax.set_ylabel("Depth (meters)")

    # Convergence plot
    ax_conv = axes[1, 1]
    ax_conv.set_title("Optimization Convergence Rate (Fitness vs Iterations)", fontsize=12, fontweight="bold")
    colors = {"PSO": "#2980B9", "GA": "#27AE60", "SA": "#8E44AD"}

    for name, res in results_by_method.items():
        if res is not None and res.history:
            xs = np.linspace(0, 100, len(res.history))
            ax_conv.plot(
                xs,
                res.history,
                label=f"{name} (Best: ₹{res.cost:,.0f}, Cov: {res.coverage*100:.1f}%)",
                color=colors.get(name, "#333333"),
                linewidth=2
            )

    ax_conv.set_xlabel("Optimization Progress (%)")
    ax_conv.set_ylabel("Objective Fitness (Cost + Penalties)")
    ax_conv.grid(True, linestyle="--", alpha=0.6)
    ax_conv.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_convergence_curves(results: Dict[str, OptimizationResult], save_path: str = None):
    """Plots standalone convergence curves for all algorithms."""
    plt.figure(figsize=(9, 5))
    colors = {"PSO": "#2980B9", "GA": "#27AE60", "SA": "#8E44AD"}
    for name, res in results.items():
        if res.history:
            xs = np.linspace(0, 100, len(res.history))
            plt.plot(xs, res.history, label=f"{name} ({res.time_seconds:.2f}s)", color=colors.get(name), linewidth=2)
    plt.title("Convergence Comparison Across Metaheuristics", fontsize=12, fontweight="bold")
    plt.xlabel("Progress (%)")
    plt.ylabel("Objective Fitness")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
