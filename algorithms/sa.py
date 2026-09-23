"""
Simulated Annealing (SA) Engine with Boltzmann Acceptance and Temperature-Scaled Gaussian Perturbation.
"""

import time
from typing import List, Tuple
import numpy as np
from .base import BaseOptimizer, OptimizationResult


class SAOptimizer(BaseOptimizer):
    """Simulated Annealing optimizer for sensor layout optimization."""

    def __init__(
        self,
        geometry,
        fitness_evaluator,
        max_iter: int = 4800,
        t_init: float = 1000.0,
        alpha: float = 0.995,
        seed: int = 42
    ):
        super().__init__(geometry, fitness_evaluator, seed)
        self.max_iter = max_iter
        self.t_init = t_init
        self.alpha = alpha

    def optimize(self) -> OptimizationResult:
        np.random.seed(self.seed)
        allocations = self.geo.get_initial_room_allocations()
        num_detectors = len(allocations)

        # Initialize detector coordinates
        current_dets = []
        for r_idx in allocations:
            rect = self.geo.rooms[r_idx]["rect"]
            x = np.random.uniform(rect[0] + 0.5, rect[2] - 0.5)
            y = np.random.uniform(rect[1] + 0.5, rect[3] - 0.5)
            current_dets.append([x, y, r_idx])

        current_fit, _, _, _ = self.evaluator.evaluate(current_dets)
        best_dets = [list(d) for d in current_dets]
        best_fit = current_fit

        history = []
        T = self.t_init
        start_time = time.time()

        for it in range(self.max_iter):
            # Select random detector to perturb
            idx = np.random.randint(num_detectors)
            r_idx = current_dets[idx][2]
            rect = self.geo.rooms[r_idx]["rect"]

            step_size = 1.2 * (T / self.t_init) + 0.1
            new_x = np.clip(
                current_dets[idx][0] + np.random.normal(0, step_size),
                rect[0] + 0.5,
                rect[2] - 0.5
            )
            new_y = np.clip(
                current_dets[idx][1] + np.random.normal(0, step_size),
                rect[1] + 0.5,
                rect[3] - 0.5
            )

            neighbor_dets = [list(d) for d in current_dets]
            neighbor_dets[idx][0] = new_x
            neighbor_dets[idx][1] = new_y

            cand_fit, _, _, _ = self.evaluator.evaluate(neighbor_dets)
            delta = cand_fit - current_fit

            # Metropolis Boltzmann Acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-4)):
                current_dets = neighbor_dets
                current_fit = cand_fit
                if cand_fit < best_fit:
                    best_fit = cand_fit
                    best_dets = [list(d) for d in neighbor_dets]

            if it % 40 == 0:
                history.append(float(best_fit))

            T *= self.alpha

        elapsed = time.time() - start_time
        _, final_cov, final_cost, _ = self.evaluator.evaluate(best_dets)

        return OptimizationResult(
            method_name="SA",
            floor_name=self.geo.name,
            detectors=[(d[0], d[1], d[2]) for d in best_dets],
            count=len(best_dets),
            fitness=float(best_fit),
            coverage=final_cov,
            cost=final_cost,
            time_seconds=elapsed,
            history=history
        )
