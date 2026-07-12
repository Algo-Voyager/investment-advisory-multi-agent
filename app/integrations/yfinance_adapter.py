"""yfinance adapter — the always-available baseline (no API key).

Quirks handled here so nobody else has to know them:
- `fast_info` values need ATTRIBUTE access (`.last_price`); dict-style `.get()`
  returns None in current yfinance versions.
- `.news` has shipped two shapes (flat and nested under 'content').
- Anything missing/empty becomes a ToolError, never a silent None.
"""

from datetime import datetime

import yfinance as yf

from app.errors.exceptions import ToolError
from app.integrations.base import MarketDataAdapter


class YFinanceAdapter(MarketDataAdapter):
    name = "yfinance"

    def get_quote(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        try:
            # fast_info computes lazily and raises raw KeyErrors (e.g.
            # 'exchangeTimezoneName') for unknown tickers — contain the vendor mess here.
            price = getattr(t.fast_info, "last_price", None)
        except Exception:
            price = None
        if not price:
            hist = t.history(period="5d")
            if hist.empty:
                raise ToolError(f"[{self.name}] no price for '{ticker}' — unknown or delisted symbol")
            price = float(hist["Close"].iloc[-1])
        return {"ticker": ticker, "price": round(float(price), 4), "source": self.name}

    def get_price_history(self, ticker: str, period: str = "6mo") -> dict:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            raise ToolError(f"[{self.name}] no history for '{ticker}' ({period})")
        return {
            "ticker": ticker,
            "dates": [d.date().isoformat() for d in hist.index],
            "closes": [round(float(c), 4) for c in hist["Close"]],
            "highs": [round(float(h), 4) for h in hist["High"]],   # ATR needs OHLC
            "lows": [round(float(low), 4) for low in hist["Low"]],
            "volumes": [int(v) for v in hist["Volume"]],
            "source": self.name,
        }

    def get_fundamentals(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        cap = getattr(t.fast_info, "market_cap", None)
        info = {}
        if not cap:
            try:
                info = t.info or {}
                cap = info.get("marketCap")
            except Exception as exc:
                raise ToolError(f"[{self.name}] fundamentals failed for '{ticker}': {exc}") from exc
        return {
            "ticker": ticker,
            "market_cap": cap,
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "dividend_yield": info.get("dividendYield"),
            "source": self.name,
        }

    def get_news(self, ticker: str, limit: int = 10) -> dict:
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as exc:
            raise ToolError(f"[{self.name}] news failed for '{ticker}': {exc}") from exc
        items = []
        for entry in raw[: limit * 2]:
            content = entry.get("content", entry)  # both shapes yfinance has shipped
            title = content.get("title")
            if not title:
                continue
            provider = content.get("provider", {})
            published = entry.get("providerPublishTime")
            if published:
                published = datetime.fromtimestamp(published).date().isoformat()
            else:
                published = (content.get("pubDate") or "")[:10] or "unknown"
            items.append({
                "title": title,
                "publisher": provider.get("displayName") or entry.get("publisher", "unknown"),
                "published": published,
                "url": (content.get("canonicalUrl") or {}).get("url") or entry.get("link", ""),
            })
        return {"ticker": ticker, "news": items[:limit], "source": self.name}
