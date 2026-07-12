"""Phase 6 tests — risk metrics, VaR strategies, profiles, tolerance mismatch."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.data.repositories import profile_repo  # noqa: E402
from app.errors.exceptions import ToolError  # noqa: E402
from app.tools.risk_tools import (  # noqa: E402
    HistoricalVaR,
    ParametricVaR,
    _bucket,
    _holding_risk_weight,
)

ROOT = Path(__file__).resolve().parents[1]
_env = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
HAS_REAL_KEY = "GOOGLE_API_KEY=" in _env and "your-gemini-key" not in _env


def _network_available() -> bool:
    try:
        from app.tools.portfolio_tools import _current_price

        return _current_price("SPY") > 0
    except Exception:
        return False


# ---------------------------------------------------------------- profiles
class TestProfiles:
    def test_all_ten_profiles_seeded_and_loadable(self):
        for i in range(1, 11):
            profile = profile_repo.get(f"CLT-{i:03d}")
            assert profile is not None, f"missing profile CLT-{i:03d} — run scripts.seed_profiles"
            assert 1 <= profile.risk_score <= 10
            assert profile.time_horizon_years > 0

    def test_deliberate_mismatches_are_in_place(self):
        assert profile_repo.get("CLT-009").risk_tolerance == "conservative"  # holds spec tech
        assert profile_repo.get("CLT-003").risk_tolerance == "aggressive"    # holds index+cash

    def test_profile_files_carry_the_synthetic_note(self):
        import json

        data = json.loads((ROOT / "data/profiles/CLT-009.json").read_text())
        assert "synthetic" in data["_note"]


# ---------------------------------------------------------------- VaR strategies (pure math)
class TestVaRStrategies:
    # 1000 fake daily returns: N(0, 1%) — both methods must land near z·σ
    rng = np.random.RandomState(7)
    returns = pd.Series(rng.normal(0, 0.01, 1000))

    def test_parametric_matches_z_times_sigma(self):
        result = ParametricVaR().compute(self.returns, 100_000, 0.95)
        expected = 1.6449 * float(self.returns.std()) * 100
        assert result["var_1d_pct"] == pytest.approx(round(expected, 2), abs=0.01)
        assert result["var_1d_dollars"] == pytest.approx(100_000 * expected / 100, rel=0.01)

    def test_historical_is_the_empirical_quantile(self):
        result = HistoricalVaR().compute(self.returns, 100_000, 0.95)
        expected = -float(np.quantile(self.returns, 0.05)) * 100
        assert result["var_1d_pct"] == pytest.approx(round(expected, 2), abs=0.01)

    def test_methods_agree_roughly_on_normal_data(self):
        h = HistoricalVaR().compute(self.returns, 1, 0.95)["var_1d_pct"]
        p = ParametricVaR().compute(self.returns, 1, 0.95)["var_1d_pct"]
        assert h == pytest.approx(p, rel=0.15)  # normal data → both ≈ z·σ

    def test_parametric_rejects_unsupported_confidence(self):
        with pytest.raises(ToolError, match="confidence"):
            ParametricVaR().compute(self.returns, 1, 0.87)

    def test_unknown_method_is_clean_toolerror(self):
        from app.tools.risk_tools import value_at_risk

        with pytest.raises(ToolError, match="Unknown VaR method"):
            value_at_risk("CLT-001", method="quantum")


# ---------------------------------------------------------------- scoring heuristics
class TestRiskScoring:
    def test_bucket_mapping(self):
        assert _bucket("Cash Equivalent", True) == "cash"
        assert _bucket("Municipal Bond ETF", False) == "bond"
        assert _bucket("International Equity ETF", False) == "international"
        assert _bucket("Individual Stock", False) == "equity"

    def test_holding_risk_weights_order_sensibly(self):
        from app.data.models import Holding

        def h(asset_class):
            return Holding(symbol="X", security_name="x", asset_class=asset_class,
                           quantity=1, purchase_date="2024-01-01", purchase_price=1.0,
                           sector="s")

        cash = _holding_risk_weight(h("Cash Equivalent"))
        bond = _holding_risk_weight(h("Bond ETF"))
        broad = _holding_risk_weight(h("Large Cap ETF"))
        thematic = _holding_risk_weight(h("Clean Energy ETF"))
        stock = _holding_risk_weight(h("Individual Stock"))
        assert cash < bond < broad < thematic < stock  # the whole point of the scale


# ---------------------------------------------------------------- network
@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestRiskToolsLive:
    def test_tolerance_mismatch_fires_for_clt009_and_not_clt001(self):
        from app.tools.risk_tools import risk_tolerance_check

        assert risk_tolerance_check("CLT-009")["mismatch"] is True
        assert risk_tolerance_check("CLT-001")["mismatch"] is False

    def test_conservative_book_less_volatile_than_speculative_one(self):
        from app.tools.risk_tools import portfolio_volatility

        balanced = portfolio_volatility("CLT-001")["annualized_volatility_pct"]
        speculative = portfolio_volatility("CLT-009")["annualized_volatility_pct"]
        assert 0 < balanced < speculative

    def test_beta_reports_equity_sleeve_and_exclusions(self):
        from app.tools.risk_tools import portfolio_beta

        result = portfolio_beta("CLT-001")  # holds BND/VTEB + cash → real exclusions
        assert result["excluded_cash_bonds_pct"] > 20
        assert 0 < result["beta_equity_sleeve"] < 2

    def test_concentration_shows_asset_class_and_single_issuer(self):
        from app.tools.risk_tools import concentration_metrics

        result = concentration_metrics("CLT-005")  # 11 individual stocks + cash
        assert result["asset_class_split_pct"]["equity"] > 80
        assert result["single_issuer_concentration"]["weight_pct"] > 5

    def test_var_both_methods_are_sane(self):
        from app.tools.risk_tools import value_at_risk

        for method in ("historical", "parametric"):
            result = value_at_risk("CLT-002", method=method)
            assert 0 < result["var_1d_pct"] < 15


# ---------------------------------------------------------------- llm (live agent)
@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestRiskAgentLive:
    def test_clt009_gets_a_plainly_stated_mismatch(self):
        from app.agents.risk import RiskAssessmentAgent

        answer = RiskAssessmentAgent().answer(
            client_id="CLT-009",
            query="What is my risk exposure and is it aligned with my risk tolerance?").lower()
        assert "conservative" in answer          # names the profiled tolerance
        assert "mismatch" in answer or "riskier" in answer or "exceeds" in answer
