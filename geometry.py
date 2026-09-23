"""
Floor geometry, room definitions, and spatial point grid discretization.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class FloorGeometry:
    """Represents a building floor plan with room boundaries and coverage constraints."""

    def __init__(self, name: str, rooms: List[Dict[str, Any]], grid_res: float = 0.25):
        self.name = name
        self.rooms = rooms
        self.grid_res = grid_res
        self.grid_points, self.point_room_idx = self._generate_discretized_grid()
        self.total_enclosed_area = sum(
            (r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1])
            for r in self.rooms
        )

    def _generate_discretized_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Discretizes all enclosed room areas into a uniform 2D grid for coverage calculation."""
        points = []
        point_room_idx = []
        for r_idx, r in enumerate(self.rooms):
            x1, y1, x2, y2 = r["rect"]
            xs = np.arange(x1 + self.grid_res / 2.0, x2, self.grid_res)
            ys = np.arange(y1 + self.grid_res / 2.0, y2, self.grid_res)
            for x in xs:
                for y in ys:
                    points.append((x, y))
                    point_room_idx.append(r_idx)
        return np.array(points), np.array(point_room_idx)

    def get_initial_room_allocations(self) -> List[int]:
        """Calculates initial statutory detector count per room based on area and fire safety classification."""
        allocations = []
        for r_idx, r in enumerate(self.rooms):
            x1, y1, x2, y2 = r["rect"]
            area = (x2 - x1) * (y2 - y1)
            min_req = r.get("min_detectors", 1)
            if area > 80.0:
                count = max(min_req, 3)
            elif area > 35.0:
                count = max(min_req, 2)
            else:
                count = min_req
            for _ in range(count):
                allocations.append(r_idx)
        return allocations


def get_ground_floor(grid_res: float = 0.25) -> FloorGeometry:
    """Returns Ground Floor geometry model for UGVCL SCADA Centre."""
    rooms = [
        {"name": "Battery Room", "rect": [0.0, 10.0, 6.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
        {"name": "Power Supply Room", "rect": [6.0, 10.0, 12.0, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.2},
        {"name": "Server Room", "rect": [12.0, 10.0, 17.77, 16.5], "critical": True, "min_detectors": 2, "target_r": 4.0},
        {"name": "WAN Room", "rect": [17.77, 10.0, 22.54, 16.5], "critical": True, "min_detectors": 1, "target_r": 4.0},
        {"name": "Staircase Lobby (GF)", "rect": [0.0, 6.0, 6.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Entrance Foyer", "rect": [6.0, 6.0, 10.5, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Main Corridor (2.77m)", "rect": [10.5, 7.23, 22.54, 10.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
        {"name": "Store Room (GF)", "rect": [0.0, 0.0, 6.0, 3.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Discussion Room", "rect": [0.0, 3.0, 6.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Staff Sitting (GF)", "rect": [6.0, 0.0, 12.0, 6.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
        {"name": "Control Room (SCADA)", "rect": [12.0, 0.0, 18.0, 7.23], "critical": True, "min_detectors": 2, "target_r": 4.2},
        {"name": "Drinking Water & Lobby", "rect": [18.0, 4.0, 22.54, 7.23], "critical": False, "min_detectors": 1, "target_r": 5.3},
    ]
    return FloorGeometry("Ground Floor", rooms, grid_res)


def get_first_floor(grid_res: float = 0.25) -> FloorGeometry:
    """Returns First Floor geometry model for UGVCL SCADA Centre."""
    rooms = [
        {"name": "Conference Room (50P)", "rect": [6.0, 10.0, 22.54, 16.5], "critical": True, "min_detectors": 3, "target_r": 5.0},
        {"name": "Pantry & Lobby", "rect": [2.0, 12.5, 6.0, 16.5], "critical": False, "min_detectors": 1, "target_r": 4.5},
        {"name": "Store (Top-Right)", "rect": [22.54, 13.0, 24.5, 16.5], "critical": False, "min_detectors": 1, "target_r": 4.5},
        {"name": "Staircase Lobby (FF)", "rect": [0.0, 6.0, 6.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Entrance Foyer (FF)", "rect": [6.0, 6.0, 10.0, 10.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Staff Sitting (FF)", "rect": [10.0, 6.0, 18.0, 10.0], "critical": False, "min_detectors": 2, "target_r": 5.3},
        {"name": "Main Corridor (2.70m)", "rect": [6.0, 4.5, 18.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Store (Middle)", "rect": [2.5, 3.5, 6.0, 6.0], "critical": False, "min_detectors": 1, "target_r": 5.0},
        {"name": "VIP / Visitor Room", "rect": [0.0, 0.0, 6.0, 3.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "SE Cabin (Sup. Eng.)", "rect": [6.0, 0.0, 10.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "EE-1 Cabin (Ex. Eng.)", "rect": [10.0, 0.0, 14.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "EE-2 Cabin (Ex. Eng.)", "rect": [14.0, 0.0, 18.0, 4.5], "critical": False, "min_detectors": 1, "target_r": 5.3},
        {"name": "Drinking Water & Lobby", "rect": [18.0, 3.5, 22.54, 7.2], "critical": False, "min_detectors": 1, "target_r": 5.3},
    ]
    return FloorGeometry("First Floor", rooms, grid_res)
