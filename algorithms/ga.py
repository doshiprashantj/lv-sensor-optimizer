"""
Genetic Algorithm (GA) Engine with Tournament Selection, BLX-alpha Crossover, and Adaptive Gaussian Mutation.
"""

import time
from typing import List, Tuple
import numpy as np
from .base import BaseOptimizer, OptimizationResult


class GAOptimizer(BaseOptimizer):
    """Genetic Algorithm for multi-room sensor layout optimization."""

    def __init__(
        self,
        geometry,
        fitness_evaluator,
        max_gen: int = 120,
        popsize: int = 40,
        mutation_rate: float = 0.15,
        crossover_alpha: float = 0.3,
        elitism_count: int = 2,
        seed: int = 42
    ):
        super().__init__(geometry, fitness_evaluator, seed)
        self.max_gen = max_gen
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.crossover_alpha = crossover_alpha
        self.elitism_count = elitism_count

    def optimize(self) -> OptimizationResult:
        np.random.seed(self.seed)
        allocations = self.geo.get_initial_room_allocations()
        num_detectors = len(allocations)
        dim = num_detectors * 2

        lb = np.zeros(dim)
        ub = np.zeros(dim)
        for i, r_idx in enumerate(allocations):
            rect = self.geo.rooms[r_idx]["rect"]
            lb[2 * i] = rect[0] + 0.5
            ub[2 * i] = rect[2] - 0.5
            lb[2 * i + 1] = rect[1] + 0.5
            ub[2 * i + 1] = rect[3] - 0.5

        # Initialize Population
        population = np.zeros((self.popsize, dim))
        for p in range(self.popsize):
            for i in range(dim):
                population[p, i] = np.random.uniform(lb[i], ub[i])

        history = []
        best_individual = None
        best_fitness = 1e9
        start_time = time.time()

        for gen in range(self.max_gen):
            fitnesses = np.zeros(self.popsize)
            for p in range(self.popsize):
                dets = [(population[p, 2 * i], population[p, 2 * i + 1], allocations[i]) for i in range(num_detectors)]
                fit, _, _, _ = self.evaluator.evaluate(dets)
                fitnesses[p] = fit
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = np.copy(population[p])

            history.append(float(best_fitness))

            # Elitism
            sorted_indices = np.argsort(fitnesses)
            new_pop = [np.copy(population[sorted_indices[e]]) for e in range(self.elitism_count)]

            # Selection, Crossover, and Mutation
            while len(new_pop) < self.popsize:
                t1 = np.random.choice(self.popsize, 3, replace=False)
                t2 = np.random.choice(self.popsize, 3, replace=False)
                p1 = t1[np.argmin(fitnesses[t1])]
                p2 = t2[np.argmin(fitnesses[t2])]

                # BLX-alpha Crossover
                child = np.zeros(dim)
                for d in range(dim):
                    c_min = min(population[p1, d], population[p2, d])
                    c_max = max(population[p1, d], population[p2, d])
                    rng = c_max - c_min
                    child[d] = np.random.uniform(c_min - self.crossover_alpha * rng, c_max + self.crossover_alpha * rng)

                # Adaptive Gaussian Mutation
                for d in range(dim):
                    if np.random.rand() < self.mutation_rate:
                        sigma = (ub[d] - lb[d]) * 0.1 * (1.0 - gen / self.max_gen)
                        child[d] += np.random.normal(0, max(sigma, 0.05))

                child = np.clip(child, lb, ub)
                new_pop.append(child)

            population = np.array(new_pop)

        elapsed = time.time() - start_time
        best_dets = [(best_individual[2 * i], best_individual[2 * i + 1], allocations[i]) for i in range(num_detectors)]
        _, final_cov, final_cost, _ = self.evaluator.evaluate(best_dets)

        return OptimizationResult(
            method_name="GA",
            floor_name=self.geo.name,
            detectors=best_dets,
            count=len(best_dets),
            fitness=float(best_fitness),
            coverage=final_cov,
            cost=final_cost,
            time_seconds=elapsed,
            history=history
        )
