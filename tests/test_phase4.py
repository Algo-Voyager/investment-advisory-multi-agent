"""Phase 4 tests — indicators (hand-verified math), factory, tools, live agent."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.errors.exceptions import ToolError  # noqa: E402
from app.indicators import indicator_factory  # noqa: E402
from app.indicators.rsi import RSI  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_env = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
HAS_REAL_KEY = "GOOGLE_API_KEY=" in _env and "your-gemini-key" not in _env


def _network_available() -> bool:
    try:
        from app.integrations.yfinance_adapter import YFinanceAdapter

        return YFinanceAdapter().get_quote("VTI")["price"] > 0
    except Exception:
        return False


# ---------------------------------------------------------------- indicator math
class TestRSIHandVerified:
    def test_pure_uptrend_rsi_is_100(self):
        close = pd.Series(np.arange(1.0, 40.0))  # gains every day, zero losses
        assert RSI().compute(close)["value"] == 100.0

    def test_pure_downtrend_rsi_near_zero(self):
        close = pd.Series(np.arange(40.0, 1.0, -1.0))
        assert RSI().compute(close)["value"] < 1.0

    def test_wilder_smoothing_hand_calculation(self):
        # Deterministic hand-check: +1 then -1 alternating forever.
        # Wilder-smoothed avg gain and avg loss converge toward the same value
        # → RS → 1 → RSI → 50. Convergence is asymptotic and the final -1 move
        # always tilts the average slightly (observed ≈48.2), hence the ±2.5 band.
        moves = np.tile([1.0, -1.0], 200)
        close = pd.Series(100 + np.cumsum(moves))
        value = RSI().compute(close)["value"]
        assert value == pytest.approx(50.0, abs=2.5)

    def test_zones(self):
        up = pd.Series(np.arange(1.0, 40.0))
        assert RSI().compute(up)["zone"] == "overbought"
        down = pd.Series(np.arange(40.0, 1.0, -1.0))
        assert RSI().compute(down)["zone"] == "oversold"


class TestOtherIndicators:
    close = pd.Series(np.linspace(100, 120, 60))

    def test_sma_is_the_plain_mean(self):
        result = indicator_factory.get("sma_20").compute(self.close)
        assert result["value"] == pytest.approx(round(self.close.tail(20).mean(), 2))

    def test_ema_reacts_faster_than_sma_in_uptrend(self):
        sma = indicator_factory.get("sma_20").compute(self.close)["value"]
        ema = indicator_factory.get("ema_20").compute(self.close)["value"]
        assert ema > sma  # in a steady uptrend the EMA hugs price more closely

    def test_macd_matches_manual_ema_difference(self):
        result = indicator_factory.get("macd").compute(self.close)
        manual = (self.close.ewm(span=12, adjust=False).mean()
                  - self.close.ewm(span=26, adjust=False).mean()).iloc[-1]
        assert result["macd_line"] == pytest.approx(round(float(manual), 4))
        assert result["momentum"] in ("bullish", "bearish")

    def test_bollinger_middle_is_sma20_and_bands_bracket_it(self):
        result = indicator_factory.get("bollinger").compute(self.close)
        assert result["middle"] == pytest.approx(round(self.close.tail(20).mean(), 2))
        assert result["lower"] < result["middle"] < result["upper"]

    def test_atr_requires_ohlc_and_computes_on_it(self):
        with pytest.raises(ToolError, match="OHLC"):
            indicator_factory.get("atr").compute(self.close)
        frame = pd.DataFrame({"close": self.close,
                              "high": self.close + 1.0, "low": self.close - 1.0})
        result = indicator_factory.get("atr").compute(frame)
        # TR is ~2 every day (high-low) → Wilder ATR ≈ 2
        assert result["value"] == pytest.approx(2.0, abs=0.2)


class TestFactory:
    def test_parses_windowed_names(self):
        assert indicator_factory.get("sma_50").window == 50
        assert indicator_factory.get("ema_12").kind == "ema"
        assert indicator_factory.get("rsi").name == "rsi"

    def test_unknown_indicator_is_a_clean_toolerror(self):
        with pytest.raises(ToolError, match="Unknown indicator"):
            indicator_factory.get("stochastic_wizardry")


# ---------------------------------------------------------------- tools
class TestIndicatorTools:
    def test_cash_is_rejected_up_front_without_network(self):
        from app.tools.indicator_tools import technical_analysis

        result = technical_analysis("CASH", ["rsi"])
        assert result["status"] == "not_applicable"
        assert "cash position" in result["message"]

    def test_check_holding_flags_not_held(self):
        from app.tools.indicator_tools import check_holding

        assert check_holding("CLT-003", "NVDA")["held"] is False  # CLT-003: funds only
        assert check_holding("CLT-002", "NVDA")["held"] is True

    def test_registered_under_securities_agent(self):
        from app.tools.registry import tool_registry

        names = [r["tool"] for r in tool_registry.table() if r["agent"] == "securities_analysis"]
        assert {"technical_analysis", "compare_indicators", "check_holding"} <= set(names)


class TestRouterKnowsSecurities:
    def test_technical_query_routes_to_securities_first(self):
        from langchain_core.messages import HumanMessage

        from app.graph.router import AgentSpec, KeywordRoutingStrategy

        specs = [AgentSpec("portfolio", "..."), AgentSpec("market_research", "..."),
                 AgentSpec("securities_analysis", "...")]
        state = {"messages": [HumanMessage(
            content="Perform a technical analysis of my NVIDIA position including moving averages and RSI")],
            "tool_results": {}}
        assert KeywordRoutingStrategy().route(state, specs) == "securities_analysis"


# ---------------------------------------------------------------- network
@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestToolsLive:
    def test_technical_analysis_on_real_nvda(self):
        from app.tools.indicator_tools import technical_analysis

        result = technical_analysis("NVDA", ["rsi", "sma_20", "sma_50"])
        assert result["series_length"] > 50
        assert 0 <= result["indicators"]["rsi"]["value"] <= 100
        assert result["indicators"]["sma_50"]["value"] > 0
        assert "RSI" in result["summary"]

    def test_compare_sma_windows_reports_crossover_state(self):
        from app.tools.indicator_tools import compare_indicators

        result = compare_indicators("NVDA", "sma", [20, 50])
        assert "sma_20" in result["values"] and "sma_50" in result["values"]
        assert result["crossovers"][0]["state"].startswith(("bullish", "bearish"))


# ---------------------------------------------------------------- llm (live agent)
@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestSecuritiesAgentLive:
    def test_nvda_analysis_cites_real_numbers_for_clt005(self):
        import re

        from app.agents.securities_analysis import SecuritiesAnalysisAgent

        answer = SecuritiesAnalysisAgent().answer(
            client_id="CLT-005",
            query="Perform a technical analysis of my NVIDIA position including moving averages and RSI indicators")
        assert "rsi" in answer.lower()
        assert re.search(r"\d+\.?\d*", answer)  # contains actual numbers

    def test_clt003_asking_nvda_gets_not_held_not_a_fabricated_analysis(self):
        from app.agents.securities_analysis import SecuritiesAnalysisAgent

        answer = SecuritiesAnalysisAgent().answer(
            client_id="CLT-003",
            query="Perform a technical analysis of my NVDA position").lower()
        assert "not hold" in answer or "don't hold" in answer or "does not hold" in answer
