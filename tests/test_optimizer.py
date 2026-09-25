import sys
import os
import unittest

# Ensure package is importable when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lv_sensor_optimizer.geometry import get_ground_floor, get_first_floor
from lv_sensor_optimizer.cost_model import CostModel
from lv_sensor_optimizer.fitness import FitnessEvaluator
from lv_sensor_optimizer.algorithms.pso import PSOOptimizer
from lv_sensor_optimizer.algorithms.ga import GAOptimizer
from lv_sensor_optimizer.algorithms.sa import SAOptimizer


class TestOptimizer(unittest.TestCase):
    def test_geometry_creation(self):
        gf = get_ground_floor(grid_res=0.5)
        self.assertEqual(gf.name, "Ground Floor")
        self.assertEqual(len(gf.rooms), 12)
        self.assertGreater(len(gf.grid_points), 0)
        self.assertGreater(gf.total_enclosed_area, 300.0)

    def test_cost_model(self):
        cost_model = CostModel(multisensor_unit_price=5250.0, installation_and_base=1650.0)
        self.assertEqual(cost_model.total_cost_per_node, 6900.0)
        self.assertEqual(cost_model.compute_cost(10), 69000.0)
        self.assertEqual(cost_model.compute_cost(10, tier_multiplier=2), 138000.0)

    def test_pso_optimizer(self):
        gf = get_ground_floor(grid_res=0.5)
        evaluator = FitnessEvaluator(gf)
        pso = PSOOptimizer(gf, evaluator, max_iter=15, swarmsize=10)
        result = pso.optimize()
        self.assertEqual(result.method_name, "PSO")
        self.assertGreaterEqual(result.count, 12)
        self.assertGreaterEqual(result.coverage, 0.90)

    def test_ga_optimizer(self):
        gf = get_ground_floor(grid_res=0.5)
        evaluator = FitnessEvaluator(gf)
        ga = GAOptimizer(gf, evaluator, max_gen=15, popsize=10)
        result = ga.optimize()
        self.assertEqual(result.method_name, "GA")
        self.assertGreaterEqual(result.count, 12)
        self.assertGreaterEqual(result.coverage, 0.90)

    def test_sa_optimizer(self):
        gf = get_ground_floor(grid_res=0.5)
        evaluator = FitnessEvaluator(gf)
        sa = SAOptimizer(gf, evaluator, max_iter=300)
        result = sa.optimize()
        self.assertEqual(result.method_name, "SA")
        self.assertGreaterEqual(result.count, 12)
        self.assertGreaterEqual(result.coverage, 0.85)


if __name__ == '__main__':
    unittest.main()

