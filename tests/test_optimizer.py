"""
Unit tests for LV Sensor Optimizer package.
"""

import pytest
from lv_sensor_optimizer.geometry import get_ground_floor, get_first_floor
from lv_sensor_optimizer.cost_model import CostModel
from lv_sensor_optimizer.fitness import FitnessEvaluator
from lv_sensor_optimizer.algorithms.pso import PSOOptimizer
from lv_sensor_optimizer.algorithms.ga import GAOptimizer
from lv_sensor_optimizer.algorithms.sa import SAOptimizer


def test_geometry_creation():
    gf = get_ground_floor(grid_res=0.5)
    assert gf.name == "Ground Floor"
    assert len(gf.rooms) == 12
    assert len(gf.grid_points) > 0
    assert gf.total_enclosed_area > 300.0


def test_cost_model():
    cost_model = CostModel(multisensor_unit_price=5250.0, installation_and_base=1650.0)
    assert cost_model.total_cost_per_node == 6900.0
    assert cost_model.compute_cost(10) == 69000.0
    assert cost_model.compute_cost(10, tier_multiplier=2) == 138000.0


def test_pso_optimizer():
    gf = get_ground_floor(grid_res=0.5)
    evaluator = FitnessEvaluator(gf)
    pso = PSOOptimizer(gf, evaluator, max_iter=15, swarmsize=10)
    result = pso.optimize()
    assert result.method_name == "PSO"
    assert result.count >= 12
    assert result.coverage >= 0.90


def test_ga_optimizer():
    gf = get_ground_floor(grid_res=0.5)
    evaluator = FitnessEvaluator(gf)
    ga = GAOptimizer(gf, evaluator, max_gen=15, popsize=10)
    result = ga.optimize()
    assert result.method_name == "GA"
    assert result.count >= 12
    assert result.coverage >= 0.90


def test_sa_optimizer():
    gf = get_ground_floor(grid_res=0.5)
    evaluator = FitnessEvaluator(gf)
    sa = SAOptimizer(gf, evaluator, max_iter=300)
    result = sa.optimize()
    assert result.method_name == "SA"
    assert result.count >= 12
    assert result.coverage >= 0.85
