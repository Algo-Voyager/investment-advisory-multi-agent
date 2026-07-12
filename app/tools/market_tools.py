"""Market research tools — yfinance-backed, no API key required.

Honesty rules baked in: yfinance's news feed is flaky, so an empty feed returns
an explicit "no news available" instead of letting the LLM improvise; economic
indicators are a clearly-marked stub until a real FRED integration (Phase 3+)
— we never invent numbers.
"""

from datetime import datetime, timedelta

import yfinance as yf
from langchain_core.tools import tool

from app.errors.exceptions import ToolError
from app.logging import get_logger

log = get_logger(__name__)

# S&P sector ETF proxies — a keyless, reliable way to measure sector performance.
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}


def get_market_snapshot(symbol: str) -> dict:
    """Current price, 1-day / 1-month change, 52-week range, and volume for a symbol."""
    hist = yf.Ticker(symbol).history(period="1y")
    if hist.empty or len(hist) < 2:
        raise ToolError(f"No market data available for symbol '{symbol}'")
    close = hist["Close"]
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    month_ago = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    return {
        "symbol": symbol,
        "price": round(last, 2),
        "change_1d_pct": round((last / prev - 1) * 100, 2),
        "change_1mo_pct": round((last / month_ago - 1) * 100, 2),
        "high_52w": round(float(close.max()), 2),
        "low_52w": round(float(close.min()), 2),
        "volume": int(hist["Volume"].iloc[-1]),
        "as_of": hist.index[-1].date().isoformat(),
    }


def get_recent_news(symbol: str, days: int = 7) -> dict:
    """Recent news headlines for a symbol. Says so honestly when the feed has nothing."""
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception as exc:
        raise ToolError(f"News feed failed for '{symbol}': {exc}") from exc

    cutoff = datetime.now() - timedelta(days=days)
    items = []
    for entry in raw:
        # yfinance has shipped two shapes: flat dicts and nested {'content': {...}}.
        content = entry.get("content", entry)
        title = content.get("title")
        if not title:
            continue
        published = _parse_news_date(entry, content)
        if published and published < cutoff:
            continue
        provider = content.get("provider", {})
        items.append({
            "title": title,
            "publisher": provider.get("displayName") or entry.get("publisher", "unknown"),
            "published": published.date().isoformat() if published else "unknown",
            "url": (content.get("canonicalUrl") or {}).get("url") or entry.get("link", ""),
        })
    if not items:
        # Flaky feed or a quiet week — say so; do NOT let the LLM fabricate headlines.
        return {"symbol": symbol, "news": [],
                "message": f"No news available from the feed for {symbol} in the last {days} days."}
    return {"symbol": symbol, "news": items[:10]}


def get_sector_performance() -> dict:
    """1-day and 5-day performance of the 11 S&P sectors (via sector ETF proxies)."""
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


def get_economic_indicators() -> dict:
    """Macro indicators (rates, inflation, unemployment). STUB — not yet integrated."""
    # TODO(phase3+): integrate FRED via pandas_datareader or the fredapi package.
    # Deliberately returns no numbers — inventing macro data would be a hallucination.
    return {
        "status": "not_available",
        "message": "Economic indicator feed (FRED) is not integrated yet. "
                   "Do not estimate or invent macro figures.",
    }


def _parse_news_date(entry: dict, content: dict) -> datetime | None:
    ts = entry.get("providerPublishTime")
    if ts:
        return datetime.fromtimestamp(ts)
    pub = content.get("pubDate")
    if pub:
        try:
            return datetime.fromisoformat(pub.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


MARKET_TOOLS = [
    tool(get_market_snapshot),
    tool(get_recent_news),
    tool(get_sector_performance),
    tool(get_economic_indicators),
]
