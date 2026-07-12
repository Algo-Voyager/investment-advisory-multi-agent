"""Finnhub adapter (optional — needs FINNHUB_API_KEY; free tier: 60 req/min).

Main value: a far more reliable NEWS feed than yfinance's. A proper 429 status
is translated into our RateLimitError.
"""

from datetime import date, timedelta

import requests

from app.config import settings
from app.errors.exceptions import RateLimitError, ToolError
from app.integrations.base import MarketDataAdapter

BASE_URL = "https://finnhub.io/api/v1"


class FinnhubAdapter(MarketDataAdapter):
    name = "finnhub"

    def available(self) -> bool:
        return bool(settings.FINNHUB_API_KEY)

    def _get(self, path: str, params: dict) -> dict | list:
        params["token"] = settings.FINNHUB_API_KEY
        resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=15)
        if resp.status_code == 429:
            raise RateLimitError(f"[{self.name}] rate limited (429)")
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, ticker: str) -> dict:
        data = self._get("quote", {"symbol": ticker})
        price = data.get("c")  # current price
        if not price:
            raise ToolError(f"[{self.name}] no quote for '{ticker}'")
        return {"ticker": ticker, "price": round(float(price), 4), "source": self.name}

    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        # Candle data is paywalled on Finnhub's free tier — be honest and let the
        # chain fall through to yfinance instead of failing cryptically.
        raise ToolError(f"[{self.name}] price history requires a paid plan")

    def get_fundamentals(self, ticker: str) -> dict:
        data = self._get("stock/metric", {"symbol": ticker, "metric": "all"})
        metric = data.get("metric") or {}
        if not metric:
            raise ToolError(f"[{self.name}] no fundamentals for '{ticker}'")
        cap = metric.get("marketCapitalization")
        return {
            "ticker": ticker,
            "market_cap": cap * 1e6 if cap else None,  # Finnhub reports in $ millions
            "pe_ratio": metric.get("peTTM"),
            "sector": None,  # not in this endpoint
            "dividend_yield": metric.get("dividendYieldIndicatedAnnual"),
            "source": self.name,
        }

    def get_news(self, ticker: str, limit: int = 10) -> dict:
        frm = (date.today() - timedelta(days=7)).isoformat()
        data = self._get("company-news", {"symbol": ticker, "from": frm,
                                          "to": date.today().isoformat()})
        items = [{
            "title": n.get("headline"),
            "publisher": n.get("source", "unknown"),
            "published": date.fromtimestamp(n["datetime"]).isoformat() if n.get("datetime") else "unknown",
            "url": n.get("url", ""),
        } for n in (data or []) if n.get("headline")]
        return {"ticker": ticker, "news": items[:limit], "source": self.name}
