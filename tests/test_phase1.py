"""Phase 1 tests — data models, repository, portfolio tools, and the agent.

Three tiers:
- offline  : models + repository (no network, always run)
- network  : yfinance-backed tools (skipped automatically if the market feed is unreachable)
- llm      : the live agent answer (skipped unless a real GOOGLE_API_KEY is configured)
"""

import os
from pathlib import Path

import pytest

# Only fake the key when no .env exists — env vars override .env in pydantic-settings.
if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.data.models import Holding, Portfolio  # noqa: E402
from app.data.repositories import ExcelPortfolioRepository  # noqa: E402
from app.errors.exceptions import ToolError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

_env = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
HAS_REAL_KEY = "GOOGLE_API_KEY=" in _env and "your-gemini-key" not in _env


def _network_available() -> bool:
    try:
        from app.tools.portfolio_tools import _current_price

        return _current_price("VTI") > 0
    except Exception:
        return False


# ---------------------------------------------------------------- offline
class TestHoldingModel:
    def _cash(self) -> Holding:
        return Holding(
            symbol="CASH", security_name="Money Market Fund", asset_class="Cash Equivalent",
            quantity=160050, purchase_date="2024-01-16", purchase_price=50.77, sector="Cash",
        )

    def test_cash_cost_basis_is_the_dollar_balance(self):
        cash = self._cash()
        assert cash.is_cash and not cash.is_etf
        assert cash.cost_basis == 160050  # NOT 160050 * 50.77

    def test_stock_cost_basis_is_qty_times_price(self):
        nvda = Holding(
            symbol="NVDA", security_name="NVIDIA Corporation", asset_class="Individual Stock",
            quantity=200, purchase_date="2024-01-16", purchase_price=55.018, sector="Semiconductors",
        )
        assert nvda.is_individual_stock
        assert nvda.cost_basis == pytest.approx(200 * 55.018)

    def test_every_etf_flavour_is_detected(self):
        for ac in ["Equity ETF", "Bond ETF", "Municipal Bond ETF", "Clean Energy ETF", "ESG ETF"]:
            h = Holding(symbol="X", security_name="x", asset_class=ac, quantity=1,
                        purchase_date="2024-01-01", purchase_price=1.0, sector="s")
            assert h.is_etf and not h.is_individual_stock


class TestRepository:
    repo = ExcelPortfolioRepository()

    def test_union_yields_all_ten_clients(self):
        assert self.repo.client_ids() == [f"CLT-{i:03d}" for i in range(1, 11)]

    def test_clt002_has_seven_holdings_with_normalized_price_column(self):
        p = self.repo.get("CLT-002")
        assert isinstance(p, Portfolio)
        assert len(p.holdings) == 7
        nvda = p.position("NVDA")
        assert nvda is not None and nvda.purchase_price == pytest.approx(55.018)

    def test_clt006_synthetic_client_loads_like_a_real_one(self):
        p = self.repo.get("CLT-006")
        assert any(h.is_individual_stock for h in p.holdings)
        assert any(h.is_etf for h in p.holdings)
        assert any(h.is_cash for h in p.holdings)

    def test_unknown_client_raises_clean_toolerror(self):
        with pytest.raises(ToolError, match="Unknown client_id"):
            self.repo.get("CLT-999")


# ---------------------------------------------------------------- network
@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestToolsLive:
    def test_nvda_since_purchase_return_is_positive_for_clt002(self):
        from app.tools.portfolio_tools import get_position_performance

        perf = get_position_performance("CLT-002", "NVDA")
        assert perf["held"] is True
        assert perf["return_pct"] > 0  # bought at $55.018 (post-split) — up since

    def test_portfolio_value_counts_cash_at_balance(self):
        from app.tools.portfolio_tools import get_portfolio_value

        value = get_portfolio_value("CLT-001")
        cash_rows = [p for p in value["positions"] if p["symbol"] == "CASH"]
        assert cash_rows and cash_rows[0]["market_value"] == 160050

    def test_not_held_symbol_answers_gracefully(self):
        from app.tools.portfolio_tools import get_position

        result = get_position("CLT-003", "NVDA")  # CLT-003 holds no individual stocks
        assert result["held"] is False
        assert "does not hold" in result["message"]


# ---------------------------------------------------------------- llm (live agent)
@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestPortfolioAgentLive:
    def test_what_do_i_own_distinguishes_stocks_etfs_and_cash(self):
        from app.agents.portfolio import PortfolioAgent

        answer = PortfolioAgent().answer(client_id="CLT-002", query="What do I own?").lower()
        assert "nvda" in answer or "nvidia" in answer      # an individual stock
        assert "etf" in answer or "qqq" in answer          # an ETF
        assert "cash" in answer                            # the cash position
