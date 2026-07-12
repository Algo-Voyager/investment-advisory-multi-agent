"""Phase 2 tests — routing strategies, supervisor loop guard, market tools, full graph.

Tiers as in test_phase1: offline (always run), network (yfinance), llm (real key).
"""

import os
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.graph.router import END_ROUTE, AgentSpec, KeywordRoutingStrategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_env = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
HAS_REAL_KEY = "GOOGLE_API_KEY=" in _env and "your-gemini-key" not in _env

SPECS = [
    AgentSpec("portfolio", "client holdings, values, allocations, performance"),
    AgentSpec("market_research", "market conditions, news, sector performance"),
]


def _network_available() -> bool:
    try:
        from app.tools.portfolio_tools import _current_price

        return _current_price("VTI") > 0
    except Exception:
        return False


# ---------------------------------------------------------------- offline
class TestKeywordRouting:
    strategy = KeywordRoutingStrategy()

    def _state(self, query: str, ran: list[str] | None = None) -> dict:
        from langchain_core.messages import HumanMessage

        return {
            "messages": [HumanMessage(content=query)],
            "tool_results": {name: ["..."] for name in (ran or [])},
        }

    def test_portfolio_query_routes_to_portfolio(self):
        assert self.strategy.route(self._state("What stocks do I own?"), SPECS) == "portfolio"

    def test_market_query_routes_to_market_research(self):
        state = self._state("What is happening in the semiconductor sector today?")
        assert self.strategy.route(state, SPECS) == "market_research"

    def test_mixed_query_goes_portfolio_first_then_market_then_end(self):
        query = "How is my NVDA position doing and what's the news?"
        assert self.strategy.route(self._state(query), SPECS) == "portfolio"
        assert self.strategy.route(self._state(query, ran=["portfolio"]), SPECS) == "market_research"
        both = self._state(query, ran=["portfolio", "market_research"])
        assert self.strategy.route(both, SPECS) == END_ROUTE


class TestSupervisorLoopGuard:
    def test_max_hops_forces_end(self):
        from langgraph.graph import END

        from app.agents.supervisor import MAX_HOPS, SupervisorAgent

        sup = SupervisorAgent(SPECS, strategy=KeywordRoutingStrategy())
        command = sup.run({"messages": [], "hops": MAX_HOPS, "tool_results": {}})
        assert command.goto == END


# ---------------------------------------------------------------- network
@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestMarketToolsLive:
    def test_market_snapshot_has_price_and_range(self):
        from app.tools.market_tools import get_market_snapshot

        snap = get_market_snapshot("NVDA")
        assert snap["price"] > 0
        assert snap["low_52w"] <= snap["price"] <= snap["high_52w"] * 1.05

    def test_sector_performance_covers_major_sectors(self):
        from app.tools.market_tools import get_sector_performance

        perf = get_sector_performance()
        names = {s["sector"] for s in perf["sectors"]}
        assert {"Technology", "Financials", "Energy"} <= names

    def test_news_is_list_or_honest_empty(self):
        from app.tools.market_tools import get_recent_news

        result = get_recent_news("NVDA", days=7)
        assert "news" in result
        if not result["news"]:
            assert "No news available" in result["message"]

    def test_economic_indicators_stub_never_invents_numbers(self):
        from app.tools.market_tools import get_economic_indicators

        result = get_economic_indicators()
        assert result["status"] == "not_available"


# ---------------------------------------------------------------- llm (live graph)
@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestSupervisorGraphLive:
    def _run(self, client_id: str, query: str):
        from langchain_core.messages import HumanMessage

        from app.graph.builder import GraphBuilder

        graph = (GraphBuilder().with_supervisor()
                 .with_portfolio_agent().with_market_research_agent().build())
        visited = []
        state = {"messages": [HumanMessage(content=query)],
                 "client_id": client_id, "session_id": "test-p2"}
        for chunk in graph.stream(state, stream_mode="updates"):
            visited.extend(chunk.keys())
        return visited

    def test_mixed_query_visits_both_agents_in_sequence(self):
        visited = self._run("CLT-002", "How is my NVDA position doing and what's the news on it?")
        agents_run = [n for n in visited if n != "supervisor"]
        assert "portfolio" in agents_run
        assert "market_research" in agents_run
        assert agents_run.index("portfolio") < agents_run.index("market_research")
