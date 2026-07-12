"""News adapter — a focused mini-chain for headlines only.

News quality differs wildly by vendor: Finnhub (keyed) is reliable, yfinance is
flaky-but-free. This adapter presents ONE news interface and internally prefers
the best available source, falling through on failure or empty results.
"""

from app.errors.exceptions import CoPilotError
from app.integrations.base import MarketDataAdapter
from app.integrations.finnhub_adapter import FinnhubAdapter
from app.integrations.yfinance_adapter import YFinanceAdapter
from app.logging import get_logger

log = get_logger(__name__)


class NewsAdapter(MarketDataAdapter):
    name = "news_aggregator"

    def __init__(self, sources: list[MarketDataAdapter] | None = None):
        self._sources = sources or [FinnhubAdapter(), YFinanceAdapter()]

    def get_news(self, ticker: str, limit: int = 10) -> dict:
        for source in self._sources:
            if not source.available():
                continue
            try:
                result = source.get_news(ticker, limit)
                if result["news"]:  # empty feed → give the next source a chance
                    return result
                log.info("news_source_empty", source=source.name, ticker=ticker)
            except CoPilotError as exc:
                log.warning("news_source_failed", source=source.name, error=str(exc)[:100])
        return {"ticker": ticker, "news": [],
                "message": f"No news available from any source for {ticker}."}

    # This adapter only aggregates news; other data types belong to the main chain.
    def get_quote(self, ticker: str) -> dict:
        raise NotImplementedError("NewsAdapter handles news only")

    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        raise NotImplementedError("NewsAdapter handles news only")

    def get_fundamentals(self, ticker: str) -> dict:
        raise NotImplementedError("NewsAdapter handles news only")
