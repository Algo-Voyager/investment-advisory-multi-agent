"""Data access — Repository pattern.

Agents never touch Excel/pandas directly; they depend on the abstract
`PortfolioRepository`. Today the implementation reads two xlsx files (the real
8-client book + the synthetic CLT-006/008 supplement) and unions them; moving to
Postgres later means one new class, zero agent changes.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data.models import ClientProfile, Holding, Portfolio
from app.errors.exceptions import ToolError
from app.logging import get_logger

log = get_logger(__name__)

# Anchor all data paths to the project root (this file lives at <root>/app/data/),
# so the repository works no matter where Python is launched from — terminal at the
# root, a notebook whose cwd is notebooks/, or a test runner.
_ROOT = Path(__file__).resolve().parents[2]

PORTFOLIOS_XLSX = _ROOT / "data/portfolios/portfolios.xlsx"
SUPPLEMENT_XLSX = _ROOT / "data/portfolios/synthetic_supplement.xlsx"
SHEET_NAME = "Potfolios"  # sic — the typo is in the source file; read as-is
PROFILES_DIR = _ROOT / "data/profiles"


class PortfolioRepository(ABC):
    @abstractmethod
    def get(self, client_id: str) -> Portfolio: ...

    @abstractmethod
    def client_ids(self) -> list[str]: ...


class ClientProfileRepository(ABC):
    @abstractmethod
    def get(self, client_id: str) -> Optional[ClientProfile]: ...


class ExcelPortfolioRepository(PortfolioRepository):
    """Reads the real book and unions the synthetic supplement (if present).

    The whole DataFrame is loaded once and cached in memory; `get()` filters it.
    """

    def __init__(self, path: Path = PORTFOLIOS_XLSX, supplement: Path = SUPPLEMENT_XLSX):
        self._path = path
        self._supplement = supplement
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            frames = [pd.read_excel(self._path, sheet_name=SHEET_NAME)]
            if self._supplement.exists():
                frames.append(pd.read_excel(self._supplement, sheet_name=SHEET_NAME))
            df = pd.concat(frames, ignore_index=True)
            # Source column has a space; normalize once on load.
            df = df.rename(columns={"Purchase Price": "purchase_price"})
            self._df = df
            log.info("portfolios_loaded", rows=len(df), clients=df["client_id"].nunique())
        return self._df

    def client_ids(self) -> list[str]:
        return sorted(self._load()["client_id"].unique())

    def get(self, client_id: str) -> Portfolio:
        # Interceptor (Phase 7): EVERY client-scoped tool reaches data through this
        # chokepoint, so access control enforced here covers all of them — including
        # tools written later. Raises PermissionError on cross-client access.
        from app.guardrails.access_control import verify_client_access

        verify_client_access(client_id)
        df = self._load()
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            raise ToolError(
                f"Unknown client_id '{client_id}'. Known clients: {', '.join(self.client_ids())}"
            )
        holdings = []
        for record in rows.to_dict("records"):
            record.pop("client_id", None)  # Holding has no client_id field
            holdings.append(Holding(**record))
        return Portfolio(client_id=client_id, holdings=holdings, as_of=datetime.now())


class JsonClientProfileRepository(ClientProfileRepository):
    """Reads data/profiles/CLT-XXX.json. Files are seeded in Phase 6; returns None until then."""

    def __init__(self, directory: Path = PROFILES_DIR):
        self._dir = directory

    def get(self, client_id: str) -> Optional[ClientProfile]:
        from app.guardrails.access_control import verify_client_access

        verify_client_access(client_id)  # same interceptor as portfolios
        path = self._dir / f"{client_id}.json"
        if not path.exists():
            return None
        return ClientProfile(**json.loads(path.read_text()))


# Shared default instances — tools import these instead of constructing their own.
portfolio_repo = ExcelPortfolioRepository()
profile_repo = JsonClientProfileRepository()

# Market data is also a repository from the agents' point of view — but it's backed
# by the adapter fallback chain (Chain of Responsibility, app/integrations/chain.py):
# Finnhub → Alpha Vantage → yfinance, skipping any adapter whose key is missing.
from app.integrations.chain import market_data as market_data_repo  # noqa: E402,F401
