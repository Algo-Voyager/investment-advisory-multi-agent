"""Alpha Vantage adapter (optional — needs ALPHA_VANTAGE_API_KEY; free tier: 25 req/day).

Vendor quirk this adapter absorbs: when rate-limited, Alpha Vantage returns
HTTP 200 with a "Note"/"Information" field instead of a 429 — we translate that
silent failure into our RateLimitError so retry/fallback logic can react.
"""

import requests

from app.config import settings
from app.errors.exceptions import RateLimitError, ToolError
from app.integrations.base import MarketDataAdapter

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageAdapter(MarketDataAdapter):
    name = "alpha_vantage"

    def available(self) -> bool:
        return bool(settings.ALPHA_VANTAGE_API_KEY)

    def _get(self, params: dict) -> dict:
        params["apikey"] = settings.ALPHA_VANTAGE_API_KEY
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # THE quirk: 200 OK but rate-limited, announced only inside the JSON body.
        if "Note" in data or "Information" in data:
            raise RateLimitError(f"[{self.name}] rate limited: "
                                 f"{(data.get('Note') or data.get('Information'))[:100]}")
        if "Error Message" in data:
            raise ToolError(f"[{self.name}] {data['Error Message'][:120]}")
        return data

    def get_quote(self, ticker: str) -> dict:
        data = self._get({"function": "GLOBAL_QUOTE", "symbol": ticker})
        quote = data.get("Global Quote") or {}
        price = quote.get("05. price")
        if not price:
            raise ToolError(f"[{self.name}] no quote for '{ticker}'")
        return {"ticker": ticker, "price": round(float(price), 4), "source": self.name}

    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        # Free tier serves ~100 trading days ("compact"). Longer/calendar-anchored
        # periods (1y, ytd) would silently return a WRONG baseline — refuse honestly
        # and let the chain fall through to yfinance instead.
        if period not in ("compact", "100d", "3mo"):
            raise ToolError(f"[{self.name}] period '{period}' unsupported on free tier")
        data = self._get({"function": "TIME_SERIES_DAILY", "symbol": ticker,
                          "outputsize": "compact"})  # ~100 trading days
        series = data.get("Time Series (Daily)")
        if not series:
            raise ToolError(f"[{self.name}] no history for '{ticker}'")
        dates = sorted(series)
        return {
            "ticker": ticker,
            "dates": dates,
            "closes": [round(float(series[d]["4. close"]), 4) for d in dates],
            "volumes": [int(series[d]["5. volume"]) for d in dates],
            "source": self.name,
        }

    def get_fundamentals(self, ticker: str) -> dict:
        data = self._get({"function": "OVERVIEW", "symbol": ticker})
        if not data.get("Symbol"):
            raise ToolError(f"[{self.name}] no fundamentals for '{ticker}'")
        def _num(key):
            value = data.get(key)
            return float(value) if value not in (None, "None", "-", "") else None
        return {
            "ticker": ticker,
            "market_cap": _num("MarketCapitalization"),
            "pe_ratio": _num("PERatio"),
            "sector": data.get("Sector"),
            "dividend_yield": _num("DividendYield"),
            "source": self.name,
        }

    def get_news(self, ticker: str, limit: int = 10) -> dict:
        raise ToolError(f"[{self.name}] news not supported on this adapter")
