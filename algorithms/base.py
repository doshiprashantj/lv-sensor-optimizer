"""
Base classes and result data containers for optimization algorithms.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import abc


@dataclass
class OptimizationResult:
    """Stores the execution outcome of an optimization run."""
    method_name: str
    floor_name: str
    detectors: List[Tuple[float, float, int]]
    count: int
    fitness: float
    coverage: float
    cost: float
    time_seconds: float
    history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "floor": self.floor_name,
            "count": self.count,
            "fitness": self.fitness,
            "coverage": self.coverage,
            "cost": self.cost,
            "time_seconds": self.time_seconds,
            "detectors": [[float(d[0]), float(d[1]), int(d[2])] for d in self.detectors]
        }


class BaseOptimizer(abc.ABC):
    """Abstract base class for floor layout optimizers."""

    def __init__(self, geometry, fitness_evaluator, seed: int = 42):
        self.geo = geometry
        self.evaluator = fitness_evaluator
        self.seed = seed

    @abc.abstractmethod
    def optimize(self) -> OptimizationResult:
        """Executes the optimization algorithm and returns the result."""
        pass
