"""Phase 3 tests — decorators, registry, adapters, fallback chain.

Offline tests use fake adapters/functions (no network, no keys); the network
tier exercises the real chain end-to-end.
"""

import json
import os
import time
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.errors.exceptions import RateLimitError, ToolError  # noqa: E402
from app.integrations.base import MarketDataAdapter  # noqa: E402
from app.integrations.chain import MarketDataChain  # noqa: E402
from app.tools.decorators import _CACHE_DIR, cached, rate_limited, retry  # noqa: E402
from app.tools.registry import ToolRegistry, tool_registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _network_available() -> bool:
    try:
        from app.integrations.yfinance_adapter import YFinanceAdapter

        return YFinanceAdapter().get_quote("VTI")["price"] > 0
    except Exception:
        return False


# ---------------------------------------------------------------- decorators
class TestCached:
    def test_second_call_is_served_from_cache(self):
        calls = {"n": 0}

        @cached(ttl_seconds=60)
        def expensive(x: int) -> dict:
            calls["n"] += 1
            return {"x": x, "computed_at": calls["n"]}

        first = expensive(42)
        second = expensive(42)
        assert first == second
        assert calls["n"] == 1  # the function body ran ONCE
        # cleanup the entry we just wrote
        for f in _CACHE_DIR.glob("expensive-*.json"):
            f.unlink()

    def test_different_args_get_different_entries(self):
        calls = {"n": 0}

        @cached(ttl_seconds=60)
        def fetch(sym: str) -> dict:
            calls["n"] += 1
            return {"sym": sym}

        fetch("AAA")
        fetch("BBB")
        assert calls["n"] == 2
        for f in _CACHE_DIR.glob("fetch-*.json"):
            f.unlink()

    def test_expired_entry_is_refetched(self):
        calls = {"n": 0}

        @cached(ttl_seconds=0)  # everything expires instantly
        def volatile() -> dict:
            calls["n"] += 1
            return {"n": calls["n"]}

        volatile()
        volatile()
        assert calls["n"] == 2
        for f in _CACHE_DIR.glob("volatile-*.json"):
            f.unlink()


class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)  # don't actually wait
        attempts = {"n": 0}

        @retry(max_attempts=3)
        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimitError("try again")
            return "ok"

        assert flaky() == "ok"
        assert attempts["n"] == 3

    def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)

        @retry(max_attempts=2)
        def always_limited():
            raise RateLimitError("nope")

        with pytest.raises(RateLimitError):
            always_limited()

    def test_non_retryable_errors_pass_through_immediately(self):
        attempts = {"n": 0}

        @retry(max_attempts=3)
        def broken():
            attempts["n"] += 1
            raise ValueError("a bug, not a transient failure")

        with pytest.raises(ValueError):
            broken()
        assert attempts["n"] == 1  # no retries for non-listed exceptions


class TestRateLimited:
    def test_waits_when_window_is_full(self, monkeypatch):
        waited = {"s": 0.0}
        monkeypatch.setattr(time, "sleep", lambda s: waited.__setitem__("s", waited["s"] + s))

        @rate_limited(calls_per_minute=2)
        def ping() -> str:
            return "pong"

        ping(); ping()          # fills the 2-call window
        ping()                  # third must wait
        assert waited["s"] > 0


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_registry_is_a_singleton(self):
        assert ToolRegistry() is ToolRegistry()
        assert ToolRegistry() is tool_registry

    def test_production_tools_are_registered_per_agent(self):
        import app.tools.market_tools  # noqa: F401 — trigger registration
        import app.tools.portfolio_tools  # noqa: F401

        table = tool_registry.table()
        agents = {row["agent"] for row in table}
        assert {"portfolio", "market_research"} <= agents
        portfolio_tools = [r["tool"] for r in table if r["agent"] == "portfolio"]
        assert "get_ytd_returns" in portfolio_tools
        market_tools = [r["tool"] for r in table if r["agent"] == "market_research"]
        assert len(tool_registry.tools_for("portfolio")) == 8
        # market_research always has its 4 own tools; it may ALSO have Phase 5's
        # search_filings/search_news_archive if rag_tools has been imported by
        # another test module in this process (import order isn't guaranteed) —
        # so assert the base 4 are present rather than an exact total count.
        base_market_tools = {"get_market_snapshot", "get_recent_news",
                             "get_sector_performance", "get_economic_indicators"}
        assert base_market_tools <= set(market_tools)
        assert len(tool_registry.tools_for("market_research")) >= 4

    def test_registered_functions_stay_directly_callable(self):
        from app.tools.market_tools import get_economic_indicators

        assert get_economic_indicators()["status"] == "not_available"


# ---------------------------------------------------------------- chain (fakes)
class _FakeAdapter(MarketDataAdapter):
    def __init__(self, name, quote=None, fail_with=None, is_available=True):
        self.name = name
        self._quote = quote
        self._fail = fail_with
        self._available = is_available
        self.calls = 0

    def available(self):
        return self._available

    def get_quote(self, ticker):
        self.calls += 1
        if self._fail:
            raise self._fail
        return {"ticker": ticker, "price": self._quote, "source": self.name}

    def get_price_history(self, ticker, period="6mo"):
        raise ToolError("n/a")

    def get_fundamentals(self, ticker):
        raise ToolError("n/a")

    def get_news(self, ticker, limit=10):
        raise ToolError("n/a")


class TestChainOfResponsibility:
    def test_unavailable_adapter_is_skipped_without_being_called(self):
        unkeyed = _FakeAdapter("unkeyed", is_available=False)
        backup = _FakeAdapter("backup", quote=101.0)
        chain = MarketDataChain([unkeyed, backup])
        assert chain.get_quote("NVDA")["source"] == "backup"
        assert unkeyed.calls == 0

    def test_failing_adapter_falls_through_to_next(self):
        primary = _FakeAdapter("primary", fail_with=RateLimitError("429"))
        backup = _FakeAdapter("backup", quote=99.5)
        chain = MarketDataChain([primary, backup])
        result = chain.get_quote("NVDA")
        assert result["price"] == 99.5
        assert primary.calls == 1  # it was tried first

    def test_all_failing_raises_one_clean_toolerror(self):
        chain = MarketDataChain([
            _FakeAdapter("a", fail_with=ToolError("down")),
            _FakeAdapter("b", fail_with=RateLimitError("429")),
        ])
        with pytest.raises(ToolError, match="All market-data sources failed"):
            chain.get_quote("NVDA")


# ---------------------------------------------------------------- network
@pytest.mark.skipif(not _network_available(), reason="market data feed unreachable")
class TestChainLive:
    def test_quote_served_by_some_adapter(self):
        from app.integrations.chain import market_data

        quote = market_data.get_quote("NVDA")
        assert quote["price"] > 0
        assert quote["source"] in {"finnhub", "alpha_vantage", "yfinance"}

    def test_bad_ticker_gives_clean_toolerror_not_stacktrace(self):
        from app.integrations.chain import market_data

        with pytest.raises(ToolError):
            market_data.get_quote("NOT_A_REAL_TICKER_XYZ123")

    def test_cached_market_snapshot_second_call_is_instant(self):
        from app.tools.market_tools import get_market_snapshot

        t0 = time.perf_counter()
        first = get_market_snapshot("AAPL")
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        second = get_market_snapshot("AAPL")
        t_second = time.perf_counter() - t0

        assert first["price"] == second["price"]
        assert t_second < max(t_first / 5, 0.15)  # cache hit ≫ faster than network
