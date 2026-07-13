"""Phase 11 tests — circuit breaker, fallback, chain integration, global node
error handling. The `TestAcceptance` class reproduces the spec's exact check:
point a secondary adapter at a bad URL, run 5 queries, circuit opens on attempt
3, subsequent calls fall through to yfinance immediately, user never sees an error.
"""

import os
import time
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.errors.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState  # noqa: E402
from app.errors.exceptions import ToolError  # noqa: E402
from app.errors.fallback import Fallback  # noqa: E402
from app.integrations.base import MarketDataAdapter  # noqa: E402
from app.integrations.chain import MarketDataChain  # noqa: E402


def _network_available() -> bool:
    try:
        from app.tools.portfolio_tools import _current_price

        return _current_price("VTI") > 0
    except Exception:
        return False


# ---------------------------------------------------------------- circuit breaker
class TestCircuitBreaker:
    def test_stays_closed_under_the_threshold(self):
        cb = CircuitBreaker("x", threshold=3, cooldown=60)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_consecutive_failures(self):
        cb = CircuitBreaker("x", threshold=3, cooldown=60)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_fails_fast_without_calling_fn(self):
        cb = CircuitBreaker("x", threshold=1, cooldown=60)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            return "ok"

        with pytest.raises(CircuitOpenError):
            cb.call(fn)
        assert calls["n"] == 0  # fn was NEVER invoked — instant fail

    def test_half_open_after_cooldown_and_closes_on_success(self):
        cb = CircuitBreaker("x", threshold=1, cooldown=0.05)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.call(lambda: "recovered") == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_the_circuit(self):
        cb = CircuitBreaker("x", threshold=1, cooldown=0.05)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))
        assert cb.state == CircuitState.OPEN

    def test_success_resets_the_failure_counter(self):
        cb = CircuitBreaker("x", threshold=3, cooldown=60)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.call(lambda: "ok") == "ok"  # 2 failures then a success
        with pytest.raises(RuntimeError):  # counter reset — needs 3 MORE to open
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------- fallback
class TestFallback:
    def test_uses_primary_when_it_succeeds(self):
        result = Fallback(lambda: "primary", lambda: "secondary").run()
        assert result == "primary"

    def test_falls_through_to_secondary_on_any_primary_exception(self):
        def broken():
            raise ValueError("nope")

        result = Fallback(broken, lambda: "secondary").run()
        assert result == "secondary"

    def test_passes_args_through_to_whichever_runs(self):
        result = Fallback(lambda x: 1 / 0, lambda x: x * 2).run(21)
        assert result == 42


# ---------------------------------------------------------------- chain + circuit breaker integration
class _FlakyAdapter(MarketDataAdapter):
    """Always fails — simulates a dead secondary adapter (bad URL)."""

    name = "flaky_secondary"

    def __init__(self):
        self.call_count = 0

    def available(self):
        return True

    def get_quote(self, ticker):
        self.call_count += 1
        raise ToolError(f"[{self.name}] connection refused (simulated dead URL)")

    def get_price_history(self, ticker, period="6mo"):
        raise ToolError("n/a")

    def get_fundamentals(self, ticker):
        raise ToolError("n/a")

    def get_news(self, ticker, limit=10):
        raise ToolError("n/a")


class _AlwaysWorksAdapter(MarketDataAdapter):
    name = "reliable_yfinance_standin"

    def available(self):
        return True

    def get_quote(self, ticker):
        return {"ticker": ticker, "price": 123.45, "source": self.name}

    def get_price_history(self, ticker, period="6mo"):
        raise ToolError("n/a")

    def get_fundamentals(self, ticker):
        raise ToolError("n/a")

    def get_news(self, ticker, limit=10):
        raise ToolError("n/a")


class TestAcceptance:
    """The spec's exact scenario: dead secondary adapter, 5 queries, circuit opens
    on attempt 3, then fails fast so the fallback engages immediately — and the
    caller (chain.get_quote) NEVER sees an error across all 5 calls."""

    def test_circuit_opens_on_third_failure_then_falls_through_instantly(self):
        from app.errors.circuit_breaker import circuit_breakers

        flaky = _FlakyAdapter()
        reliable = _AlwaysWorksAdapter()
        # Fresh breaker per test — the registry is a process-wide singleton.
        circuit_breakers._breakers.pop(flaky.name, None)
        chain = MarketDataChain([flaky, reliable])
        breaker = circuit_breakers.get(flaky.name, threshold=3, cooldown=60)
        circuit_breakers._breakers[flaky.name] = breaker

        for i in range(1, 6):
            quote = chain.get_quote("NVDA")  # user-facing call — must NEVER raise
            assert quote["price"] == 123.45  # always served by the reliable fallback
            if i < 3:
                assert breaker.state.value == "closed"
            else:
                assert breaker.state.value == "open"

        # The dead adapter was genuinely tried 3 times (proving failures), then
        # skipped instantly for calls 4 and 5 (no more network attempts).
        assert flaky.call_count == 3


# ---------------------------------------------------------------- global node error handling
class TestGlobalNodeErrorHandling:
    def test_dict_node_exception_routes_to_safe_exit(self):
        from app.errors.node_wrapper import wrap_dict_node

        def broken(state):
            raise RuntimeError("simulated agent crash")

        wrapped = wrap_dict_node(broken, "portfolio", success_goto="supervisor")
        cmd = wrapped({"client_id": "CLT-001", "session_id": "s"})
        assert cmd.goto == "safe_exit"
        assert "blocked" in cmd.update
        assert "portfolio" in cmd.update["blocked"]

    def test_dict_node_success_routes_normally(self):
        from app.errors.node_wrapper import wrap_dict_node

        wrapped = wrap_dict_node(lambda s: {"messages": []}, "portfolio",
                                 success_goto="supervisor")
        cmd = wrapped({"client_id": "CLT-001"})
        assert cmd.goto == "supervisor"

    def test_command_node_exception_routes_to_safe_exit(self):
        from app.errors.node_wrapper import wrap_command_node

        def broken(state):
            raise RuntimeError("simulated supervisor crash")

        wrapped = wrap_command_node(broken, "supervisor")
        cmd = wrapped({"client_id": "CLT-001", "session_id": "s"})
        assert cmd.goto == "safe_exit"

    def test_no_raw_exception_ever_escapes_the_wrapper(self):
        from app.errors.node_wrapper import wrap_dict_node

        wrapped = wrap_dict_node(lambda s: 1 / 0, "risk", success_goto="supervisor")
        cmd = wrapped({"client_id": "CLT-001"})  # must NOT raise ZeroDivisionError
        assert cmd.goto == "safe_exit"


class TestBuilderWithErrorHandling:
    def test_full_graph_still_compiles_with_wrapping_enabled(self):
        from app.graph.builder import GraphBuilder

        graph = GraphBuilder().with_all().build()
        nodes = set(graph.get_graph().nodes)
        assert {"safe_exit", "portfolio", "supervisor"} <= nodes

    def test_agent_failure_reaches_safe_exit_end_to_end(self, monkeypatch):
        """A portfolio agent that always crashes must still produce a safe,
        non-crashing final answer through the whole compiled graph."""
        from langchain_core.messages import HumanMessage

        from app.agents.portfolio import PortfolioAgent
        from app.graph.builder import GraphBuilder

        def crash(self, state):
            raise RuntimeError("simulated LLM outage")

        monkeypatch.setattr(PortfolioAgent, "_invoke", crash)

        # force the keyword router straight to portfolio, no LLM needed
        from app.graph.router import KeywordRoutingStrategy

        graph = (GraphBuilder().with_supervisor(strategy=KeywordRoutingStrategy())
                .with_portfolio_agent().with_guardrails().build())
        result = graph.invoke({"messages": [HumanMessage(content="What do I own?")],
                               "client_id": "CLT-001", "session_id": "s"})
        answer = result["messages"][-1].content
        assert "couldn't complete" in answer.lower() or "problem" in answer.lower()


@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestChainStillWorksLive:
    def test_real_quote_still_served_with_circuit_breakers_active(self):
        from app.integrations.chain import market_data

        quote = market_data.get_quote("NVDA")
        assert quote["price"] > 0
