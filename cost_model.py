"""
Cost model based on Schedule of Rates (SOR) Tender Specifications.
"""


class CostModel:
    """Calculates capital equipment, cabling, and installation costs for multisensor detector nodes."""

    def __init__(
        self,
        multisensor_unit_price: float = 5250.0,
        installation_and_base: float = 1650.0,
        currency_symbol: str = "₹"
    ):
        self.multisensor_unit_price = multisensor_unit_price
        self.installation_and_base = installation_and_base
        self.total_cost_per_node = multisensor_unit_price + installation_and_base
        self.currency_symbol = currency_symbol

    def compute_cost(self, detector_count: int, tier_multiplier: int = 1) -> float:
        """Returns total installed cost for detector_count nodes across tier_multiplier (1=below ceiling, 2=dual tier)."""
        return detector_count * tier_multiplier * self.total_cost_per_node

    def format_cost(self, total_cost: float) -> str:
        """Formats currency in Indian Rupee format."""
        return f"{self.currency_symbol}{total_cost:,.0f}"
