"""Market research tools — Phase 3 shape: adapter chain + cached + retried + registered.

Honesty rules unchanged: an empty news feed says so explicitly; the economic
indicators stub refuses to invent numbers.
"""

from datetime import datetime, timedelta

import yfinance as yf

from app.config import settings
from app.errors.exceptions import ToolError
from app.integrations.chain import market_data
from app.logging import get_logger
from app.tools.decorators import cached, retry
from app.tools.registry import tool_registry

log = get_logger(__name__)

# S&P sector ETF proxies — a keyless, reliable way to measure sector performance.
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}


@tool_registry.register(agent="market_research")
@cached(ttl_seconds=settings.CACHE_TTL_QUOTES)
@retry(max_attempts=3)
def get_market_snapshot(symbol: str) -> dict:
    """Current price, 1-day / 1-month change, 52-week range, and volume for a symbol."""
    hist = market_data.get_price_history(symbol, "1y")
    closes, dates = hist["closes"], hist["dates"]
    if len(closes) < 2:
        raise ToolError(f"No market data available for symbol '{symbol}'")
    last, prev = closes[-1], closes[-2]
    month_ago = closes[-22] if len(closes) >= 22 else closes[0]
    return {
        "symbol": symbol,
        "price": round(last, 2),
        "change_1d_pct": round((last / prev - 1) * 100, 2),
        "change_1mo_pct": round((last / month_ago - 1) * 100, 2),
        "high_52w": round(max(closes), 2),
        "low_52w": round(min(closes), 2),
        "volume": hist.get("volumes", [None])[-1],
        "as_of": dates[-1],
        "source": hist["source"],
    }


@tool_registry.register(agent="market_research")
@cached(ttl_seconds=settings.CACHE_TTL_NEWS)
@retry(max_attempts=3)
def get_recent_news(symbol: str, days: int = 7) -> dict:
    """Recent news headlines for a symbol. Says so honestly when the feed has nothing."""
    result = market_data.get_news(symbol, limit=10)  # aggregator: finnhub → yfinance
    cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
    fresh = [n for n in result["news"] if n["published"] == "unknown" or n["published"] >= cutoff]
    if not fresh:
        return {"symbol": symbol, "news": [],
                "message": f"No news available from any source for {symbol} "
                           f"in the last {days} days."}
    return {"symbol": symbol, "news": fresh, "source": result.get("source")}


@tool_registry.register(agent="market_research")
@cached(ttl_seconds=settings.CACHE_TTL_NEWS)
@retry(max_attempts=3)
def get_sector_performance() -> dict:
    """1-day and 5-day performance of the 11 S&P sectors (via sector ETF proxies)."""
    # Batch download stays on yfinance directly: one request for 11 tickers —
    # the per-ticker adapter interface would cost 11 round-trips instead.
    try:
        data = yf.download(list(SECTOR_ETFS), period="5d", progress=False, auto_adjust=True)["Close"]
    except Exception as exc:
        raise ToolError(f"Sector performance download failed: {exc}") from exc
    if data.empty:
        raise ToolError("Sector performance download returned no data")
    sectors = []
    for etf, sector in SECTOR_ETFS.items():
        if etf not in data.columns:
            continue
        series = data[etf].dropna()
        if len(series) < 2:
            continue
        sectors.append({
            "sector": sector,
            "proxy_etf": etf,
            "change_1d_pct": round(float(series.iloc[-1] / series.iloc[-2] - 1) * 100, 2),
            "change_5d_pct": round(float(series.iloc[-1] / series.iloc[0] - 1) * 100, 2),
        })
    sectors.sort(key=lambda s: s["change_1d_pct"], reverse=True)
    return {"as_of": str(data.index[-1].date()), "sectors": sectors}


@tool_registry.register(agent="market_research")
def get_economic_indicators() -> dict:
    """Macro indicators (rates, inflation, unemployment). STUB — not yet integrated."""
    # TODO(phase3+): integrate FRED via pandas_datareader or the fredapi package.
    # Deliberately returns no numbers — inventing macro data would be a hallucination.
    return {
        "status": "not_available",
        "message": "Economic indicator feed (FRED) is not integrated yet. "
                   "Do not estimate or invent macro figures.",
    }


# Agent-facing toolbox — discovered from the registry.
MARKET_TOOLS = tool_registry.tools_for("market_research")
