"""
Multi-objective fitness evaluation module with vectorized raycast/room-bounding point coverage.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from .geometry import FloorGeometry
from .cost_model import CostModel


class FitnessEvaluator:
    """Evaluates multi-objective fitness: coverage %, cost, overlap penalty, boundary and minimum room constraints."""

    def __init__(self, geometry: FloorGeometry, cost_model: CostModel = None):
        self.geo = geometry
        self.cost_model = cost_model or CostModel()

    def evaluate(self, detectors: List[Tuple[float, float, int]]) -> Tuple[float, float, float, float]:
        """
        Evaluates a list of detectors: [(x, y, room_idx), ...].
        Returns:
            (fitness_score, coverage_ratio, total_cost, overlap_penalty)
        """
        if not detectors:
            return 1e9, 0.0, 0.0, 0.0

        n_detectors = len(detectors)
        covered_mask = np.zeros(len(self.geo.grid_points), dtype=bool)

        # Vectorized room-by-room coverage
        for d in detectors:
            dx, dy, r_idx = d[0], d[1], d[2]
            radius = self.geo.rooms[r_idx].get("target_r", 5.3)
            room_mask = (self.geo.point_room_idx == r_idx)
            if np.any(room_mask):
                pts = self.geo.grid_points[room_mask]
                dist_sq = (pts[:, 0] - dx) ** 2 + (pts[:, 1] - dy) ** 2
                in_range = dist_sq <= (radius ** 2)
                covered_mask[room_mask] = covered_mask[room_mask] | in_range

        coverage_ratio = float(np.sum(covered_mask) / len(self.geo.grid_points))

        # Room minimum sensor constraint
        room_counts = [0] * len(self.geo.rooms)
        for d in detectors:
            room_counts[d[2]] += 1

        min_det_penalty = 0.0
        for r_idx, r in enumerate(self.geo.rooms):
            req = r.get("min_detectors", 1)
            if room_counts[r_idx] < req:
                min_det_penalty += (req - room_counts[r_idx]) * 50000.0

        # Overlap penalty inside same room
        overlap_penalty = 0.0
        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                if detectors[i][2] == detectors[j][2]:
                    dist = np.sqrt(
                        (detectors[i][0] - detectors[j][0]) ** 2 + (detectors[i][1] - detectors[j][1]) ** 2
                    )
                    rad = self.geo.rooms[detectors[i][2]].get("target_r", 5.3)
                    if dist < rad * 0.75:
                        overlap_penalty += (rad * 0.75 - dist) * 1000.0

        # Uncovered grid points penalty
        uncovered = len(self.geo.grid_points) - np.sum(covered_mask)
        coverage_penalty = uncovered * 2500.0 if uncovered > 0 else 0.0
        if coverage_ratio < 0.98:
            coverage_penalty += (0.98 - coverage_ratio) * 500000.0

        total_cost = self.cost_model.compute_cost(n_detectors)
        fitness = total_cost + coverage_penalty + min_det_penalty + overlap_penalty

        return fitness, coverage_ratio, total_cost, overlap_penalty
