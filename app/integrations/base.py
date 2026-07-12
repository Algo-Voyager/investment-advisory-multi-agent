"""MarketDataAdapter — the Adapter pattern.

Every external market-data API (yfinance, Alpha Vantage, Finnhub, …) is wrapped
behind this one interface. Callers never learn which vendor answered; adding
Polygon later is one new file. Adapters translate each vendor's quirks —
including how it FAILS (429s, silent "Note" fields) — into our exceptions.

`available()` is the fallback chain's skip signal: an adapter whose API key is
missing reports itself unavailable instead of crashing at call time.

All methods return plain JSON-safe dicts so `@cached` can persist them.
"""

from abc import ABC, abstractmethod


class MarketDataAdapter(ABC):
    name: str  # short id used in logs, e.g. "yfinance"

    def available(self) -> bool:
        """Can this adapter be used right now? (False when its API key is absent.)"""
        return True

    @abstractmethod
    def get_quote(self, ticker: str) -> dict:
        """Latest price: {'ticker', 'price', 'source'}."""

    @abstractmethod
    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        """Daily closes: {'ticker', 'dates': [...], 'closes': [...], 'source'}."""

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict:
        """Company basics: {'ticker', 'market_cap', 'pe_ratio', 'sector', ...,'source'}."""

    @abstractmethod
    def get_news(self, ticker: str, limit: int = 10) -> dict:
        """Headlines: {'ticker', 'news': [{'title','publisher','published','url'}], 'source'}."""
