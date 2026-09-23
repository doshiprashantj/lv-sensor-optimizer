"""
Particle Swarm Optimization (PSO) Engine for continuous multi-room sensor positioning.
"""

import time
from typing import List, Tuple
import numpy as np
from .base import BaseOptimizer, OptimizationResult


class PSOOptimizer(BaseOptimizer):
    """Continuous Particle Swarm Optimization for multi-room sensor layouts."""

    def __init__(
        self,
        geometry,
        fitness_evaluator,
        max_iter: int = 120,
        swarmsize: int = 40,
        c1: float = 1.8,
        c2: float = 2.0,
        w_max: float = 0.9,
        w_min: float = 0.4,
        seed: int = 42
    ):
        super().__init__(geometry, fitness_evaluator, seed)
        self.max_iter = max_iter
        self.swarmsize = swarmsize
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min

    def optimize(self) -> OptimizationResult:
        np.random.seed(self.seed)
        allocations = self.geo.get_initial_room_allocations()
        num_detectors = len(allocations)
        dim = num_detectors * 2

        # Lower and upper spatial limits per detector
        lb = np.zeros(dim)
        ub = np.zeros(dim)
        for i, r_idx in enumerate(allocations):
            rect = self.geo.rooms[r_idx]["rect"]
            lb[2 * i] = rect[0] + 0.5
            ub[2 * i] = rect[2] - 0.5
            lb[2 * i + 1] = rect[1] + 0.5
            ub[2 * i + 1] = rect[3] - 0.5

        # Initialize Swarm
        X = np.zeros((self.swarmsize, dim))
        V = np.zeros((self.swarmsize, dim))
        for p in range(self.swarmsize):
            for i in range(num_detectors):
                X[p, 2 * i] = np.random.uniform(lb[2 * i], ub[2 * i])
                X[p, 2 * i + 1] = np.random.uniform(lb[2 * i + 1], ub[2 * i + 1])
                V[p, 2 * i] = np.random.uniform(-(ub[2 * i] - lb[2 * i]) * 0.1, (ub[2 * i] - lb[2 * i]) * 0.1)
                V[p, 2 * i + 1] = np.random.uniform(-(ub[2 * i + 1] - lb[2 * i + 1]) * 0.1, (ub[2 * i + 1] - lb[2 * i + 1]) * 0.1)

        pbest = np.copy(X)
        pbest_fit = np.full(self.swarmsize, 1e9)
        gbest = np.zeros(dim)
        gbest_fit = 1e9

        history = []
        start_time = time.time()

        for t in range(self.max_iter):
            w = self.w_max - (t / self.max_iter) * (self.w_max - self.w_min)
            for p in range(self.swarmsize):
                dets = [(X[p, 2 * i], X[p, 2 * i + 1], allocations[i]) for i in range(num_detectors)]
                fit, _, _, _ = self.evaluator.evaluate(dets)

                if fit < pbest_fit[p]:
                    pbest_fit[p] = fit
                    pbest[p] = np.copy(X[p])

                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = np.copy(X[p])

            # Velocity and position update
            r1 = np.random.rand(self.swarmsize, dim)
            r2 = np.random.rand(self.swarmsize, dim)
            V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            X = X + V

            # Boundary clipping
            for i in range(dim):
                X[:, i] = np.clip(X[:, i], lb[i], ub[i])

            history.append(float(gbest_fit))

        elapsed = time.time() - start_time
        best_dets = [(gbest[2 * i], gbest[2 * i + 1], allocations[i]) for i in range(num_detectors)]
        _, final_cov, final_cost, _ = self.evaluator.evaluate(best_dets)

        return OptimizationResult(
            method_name="PSO",
            floor_name=self.geo.name,
            detectors=best_dets,
            count=len(best_dets),
            fitness=float(gbest_fit),
            coverage=final_cov,
            cost=final_cost,
            time_seconds=elapsed,
            history=history
        )
