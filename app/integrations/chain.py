"""MarketDataChain — Chain of Responsibility across adapters.

For each request, walk the ordered adapter list:
  skip if unavailable (no API key) → try → on failure/rate-limit, log and try
  the next → if everyone fails, raise ONE clean ToolError.

Order: keyed vendors first (better data/limits when present), yfinance last
because it's always available. With zero optional keys configured, the chain
degrades to pure yfinance and everything still works.
"""

from app.errors.exceptions import CoPilotError, ToolError
from app.integrations.alpha_vantage_adapter import AlphaVantageAdapter
from app.integrations.base import MarketDataAdapter
from app.integrations.finnhub_adapter import FinnhubAdapter
from app.integrations.news_adapter import NewsAdapter
from app.integrations.yfinance_adapter import YFinanceAdapter
from app.logging import get_logger

log = get_logger(__name__)


class MarketDataChain:
    def __init__(self, adapters: list[MarketDataAdapter] | None = None):
        self._adapters = adapters or [
            FinnhubAdapter(),        # keyed — best rate limits when configured
            AlphaVantageAdapter(),   # keyed — 25/day, quotes/fundamentals backup
            YFinanceAdapter(),       # keyless — the reliable last resort
        ]
        self._news = NewsAdapter()   # news has its own quality-ordered mini-chain

    def _try_each(self, method: str, *args, **kwargs):
        errors = []
        for adapter in self._adapters:
            if not adapter.available():
                continue
            try:
                result = getattr(adapter, method)(*args, **kwargs)
                log.info("chain_served", method=method, source=adapter.name)
                return result
            except (CoPilotError, Exception) as exc:  # noqa: BLE001 — chain must survive anything
                errors.append(f"{adapter.name}: {str(exc)[:80]}")
                log.warning("chain_adapter_failed", method=method,
                            adapter=adapter.name, error=str(exc)[:100])
        raise ToolError(f"All market-data sources failed for {method}{args}: {'; '.join(errors)}")

    def get_quote(self, ticker: str) -> dict:
        return self._try_each("get_quote", ticker)

    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        return self._try_each("get_price_history", ticker, period)

    def get_fundamentals(self, ticker: str) -> dict:
        return self._try_each("get_fundamentals", ticker)

    def get_news(self, ticker: str, limit: int = 10) -> dict:
        return self._news.get_news(ticker, limit)

    def sources(self) -> list[dict]:
        """Who's in the chain and who's currently usable (for the notebook/table)."""
        return [{"adapter": a.name, "available": a.available()} for a in self._adapters]


# Shared instance — tools import this; repositories.py re-exports it as the
# market-data access point (Repository-style naming for the data layer).
market_data = MarketDataChain()
