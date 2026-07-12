"""Domain models for portfolios and clients.

The one non-obvious rule lives here: on CASH rows the source file's `quantity`
column IS the dollar balance and its `Purchase Price` is noise (~$50), so cash
cost basis = quantity and cash value = quantity × 1. Never multiply a cash row's
quantity by its purchase_price. See docs/data_assumptions.md §2.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class Holding(BaseModel):
    symbol: str
    security_name: str
    asset_class: str  # rich free text (16+ values) — deliberately NOT an enum
    quantity: float
    purchase_date: datetime
    purchase_price: float
    sector: str
    current_price: Optional[float] = None
    market_value: Optional[float] = None

    @property
    def is_cash(self) -> bool:
        return self.asset_class == "Cash Equivalent"

    @property
    def is_etf(self) -> bool:
        return "ETF" in self.asset_class

    @property
    def is_individual_stock(self) -> bool:
        return not self.is_cash and not self.is_etf

    @property
    def cost_basis(self) -> float:
        # Cash: quantity IS the dollar balance; its purchase_price is noise.
        if self.is_cash:
            return self.quantity
        return self.quantity * self.purchase_price


class Portfolio(BaseModel):
    client_id: str  # raw "CLT-XXX" string from the source — never renumbered
    holdings: list[Holding]
    as_of: datetime

    @property
    def symbols(self) -> list[str]:
        return [h.symbol for h in self.holdings]

    def position(self, symbol: str) -> Optional[Holding]:
        for h in self.holdings:
            if h.symbol.upper() == symbol.upper():
                return h
        return None


class ClientProfile(BaseModel):
    """Synthetic in this project — the source file has holdings only (Phase 6 seeds
    data/profiles/CLT-XXX.json, each marked "synthetic — not provided in source data")."""

    client_id: str
    name: str
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    goals: str
    risk_score: int  # 1 (capital preservation) … 10 (max risk appetite)
    time_horizon_years: int
    income_needs: bool
