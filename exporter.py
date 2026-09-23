"""
Export schedules to CSV, JSON, and text for authority submission.
"""

import csv
import json
from typing import Dict, Any
from .geometry import FloorGeometry
from .algorithms.base import OptimizationResult
from .cost_model import CostModel


def export_authority_schedule_csv(
    geometry: FloorGeometry,
    result: OptimizationResult,
    output_path: str,
    cost_model: CostModel = None,
    tier_multiplier: int = 1
):
    """Exports room-by-room detector schedule to CSV format for authority submission and tender BOQ."""
    cost_model = cost_model or CostModel()
    counts = [0] * len(geometry.rooms)
    for d in result.detectors:
        counts[d[2]] += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["UGVCL SCADA CENTRE - LV FIRE DETECTION & ALARM SYSTEM"])
        writer.writerow([f"Authority Submission Schedule - {geometry.name}"])
        writer.writerow([f"Algorithm: {result.method_name}", f"Coverage: {result.coverage*100:.2f}%", f"Standards: IS 2189 / NFPA 72"])
        writer.writerow([])
        writer.writerow([
            "Sr No", "Room / Protected Zone", "Room Area (sq.m)", "Detector Qty",
            "Target Radius (m)", "Risk Level", "Unit Cost (INR)", "Total Cost (INR)"
        ])

        total_qty = 0
        total_cost = 0.0
        for idx, r in enumerate(geometry.rooms):
            area = (r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1])
            qty = counts[idx] * tier_multiplier
            total_qty += qty
            cost = qty * cost_model.total_cost_per_node
            total_cost += cost
            risk = "High Risk / Critical" if r.get("critical", False) else "Standard Area"
            target_r = r.get("target_r", 5.3)

            writer.writerow([
                idx + 1, r["name"], f"{area:.2f}", qty, target_r, risk,
                f"{cost_model.total_cost_per_node:.2f}", f"{cost:.2f}"
            ])

        writer.writerow([])
        writer.writerow(["TOTAL", "", "", total_qty, "", "", "", f"{total_cost:.2f}"])


def export_summary_json(summary_data: Dict[str, Any], output_path: str):
    """Exports complete benchmark results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
