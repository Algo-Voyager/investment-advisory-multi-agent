"""Portfolio tools — the Portfolio agent's only way to touch data.

Layout: each capability is a plain, directly-callable Python function (easy to
test and to use from notebooks), and `PORTFOLIO_TOOLS` at the bottom wraps them
as LangChain tools for the agent.

Money rules (docs/data_assumptions.md): cash is valued at quantity × 1 (its
`quantity` IS the dollar balance); performance/YTD are undefined for cash;
market-cap buckets apply to individual stocks only (a fund has no single
company market cap).
"""

from datetime import datetime

import yfinance as yf
from langchain_core.tools import tool

from app.data.repositories import portfolio_repo
from app.errors.exceptions import ToolError
from app.logging import get_logger

log = get_logger(__name__)

# Per-process quote cache so one question doesn't hit yfinance N times for the
# same symbol. Proper TTL-file caching arrives with the Phase 3 decorators.
_price_cache: dict[str, float] = {}


def _current_price(symbol: str) -> float:
    """Latest price for a symbol via yfinance (fast_info, history fallback)."""
    if symbol in _price_cache:
        return _price_cache[symbol]
    try:
        ticker = yf.Ticker(symbol)
        # NOTE: attribute access — fast_info.get("last_price") returns None in current yfinance.
        price = getattr(ticker.fast_info, "last_price", None)
        if not price:
            hist = ticker.history(period="5d")
            if hist.empty:
                raise ValueError("no price history")
            price = float(hist["Close"].iloc[-1])
        _price_cache[symbol] = float(price)
        return _price_cache[symbol]
    except Exception as exc:
        raise ToolError(f"Could not fetch a price for symbol '{symbol}': {exc}") from exc


def get_holdings(client_id: str) -> list[dict]:
    """List every holding for a client: symbol, security_name, asset_class, quantity, purchase_price, sector."""
    portfolio = portfolio_repo.get(client_id)
    return [
        {
            "symbol": h.symbol,
            "security_name": h.security_name,
            "asset_class": h.asset_class,
            "quantity": h.quantity,
            "purchase_price": h.purchase_price,
            "sector": h.sector,
        }
        for h in portfolio.holdings
    ]


def get_portfolio_value(client_id: str) -> dict:
    """Total portfolio market value, marked to market. Cash is valued at quantity × 1."""
    portfolio = portfolio_repo.get(client_id)
    positions, total, failed = [], 0.0, []
    for h in portfolio.holdings:
        if h.is_cash:
            value = h.quantity  # dollar balance
        else:
            try:
                value = h.quantity * _current_price(h.symbol)
            except ToolError:
                failed.append(h.symbol)
                continue
        total += value
        positions.append({"symbol": h.symbol, "market_value": round(value, 2)})
    result = {"client_id": client_id, "total_value": round(total, 2), "positions": positions}
    if failed:
        result["price_unavailable_for"] = failed
    return result


def get_position(client_id: str, symbol: str) -> dict:
    """One position's detail: quantity, cost basis, current price, and market value."""
    portfolio = portfolio_repo.get(client_id)
    h = portfolio.position(symbol)
    if h is None:
        return {
            "held": False,
            "message": f"Client {client_id} does not hold '{symbol}'. "
            f"They hold: {', '.join(portfolio.symbols)}",
        }
    detail = {
        "held": True,
        "symbol": h.symbol,
        "security_name": h.security_name,
        "asset_class": h.asset_class,
        "quantity": h.quantity,
        "cost_basis": round(h.cost_basis, 2),
    }
    if h.is_cash:
        detail["market_value"] = h.quantity
        detail["note"] = "Cash position — valued at its dollar balance."
    else:
        price = _current_price(h.symbol)
        detail["current_price"] = round(price, 2)
        detail["market_value"] = round(h.quantity * price, 2)
    return detail


def get_position_performance(client_id: str, symbol: str) -> dict:
    """Total return of one position since purchase (current price vs purchase price). N/A for cash."""
    portfolio = portfolio_repo.get(client_id)
    h = portfolio.position(symbol)
    if h is None:
        return {
            "held": False,
            "message": f"Client {client_id} does not hold '{symbol}'.",
        }
    if h.is_cash:
        return {"held": True, "symbol": h.symbol, "message": "N/A — cash position (return = 0)."}
    price = _current_price(h.symbol)
    gain = (price - h.purchase_price) * h.quantity
    return {
        "held": True,
        "symbol": h.symbol,
        "purchase_date": h.purchase_date.date().isoformat(),
        "purchase_price": h.purchase_price,
        "current_price": round(price, 2),
        "return_pct": round((price / h.purchase_price - 1) * 100, 2),
        "absolute_gain": round(gain, 2),
    }


def get_ytd_returns(client_id: str) -> dict:
    """Year-to-date return per holding (vs the first trading day of this year). Skips cash."""
    portfolio = portfolio_repo.get(client_id)
    returns, failed = [], []
    for h in portfolio.holdings:
        if h.is_cash:
            continue
        try:
            hist = yf.Ticker(h.symbol).history(period="ytd")
            if hist.empty:
                raise ValueError("no YTD history")
            first, last = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
            returns.append({"symbol": h.symbol, "ytd_return_pct": round((last / first - 1) * 100, 2)})
        except Exception:
            failed.append(h.symbol)
    returns.sort(key=lambda r: r["ytd_return_pct"], reverse=True)
    result = {"client_id": client_id, "as_of": datetime.now().date().isoformat(), "ytd_returns": returns}
    if returns:
        result["best"] = returns[0]
        result["worst"] = returns[-1]
    if failed:
        result["data_unavailable_for"] = failed
    return result


def _allocation(client_id: str, key_fn) -> dict:
    """Shared allocation math: weight = position market value / total (cash at balance)."""
    portfolio = portfolio_repo.get(client_id)
    buckets: dict[str, float] = {}
    total = 0.0
    for h in portfolio.holdings:
        try:
            value = h.quantity if h.is_cash else h.quantity * _current_price(h.symbol)
        except ToolError:
            value = h.cost_basis  # price feed down → fall back to cost basis, still useful
        buckets[key_fn(h)] = buckets.get(key_fn(h), 0.0) + value
        total += value
    return {
        "client_id": client_id,
        "total_value": round(total, 2),
        "allocation_pct": {k: round(v / total * 100, 2) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
    }


def get_allocation_by_sector(client_id: str) -> dict:
    """Portfolio allocation percentages grouped by sector."""
    return _allocation(client_id, lambda h: h.sector)


def get_allocation_by_asset_class(client_id: str) -> dict:
    """Allocation by asset class — distinguishes Individual Stock vs the ETF types vs Cash Equivalent."""
    return _allocation(client_id, lambda h: h.asset_class)


def get_allocation_by_market_cap(client_id: str) -> dict:
    """Market-cap buckets (mega/large/mid/small) for INDIVIDUAL STOCKS only.

    ETFs and cash have no single-company market cap — their share of the portfolio
    is reported separately as 'not_classified_pct' instead of being silently mislabelled.
    """
    portfolio = portfolio_repo.get(client_id)
    buckets: dict[str, float] = {}
    unclassified = 0.0
    total = 0.0
    for h in portfolio.holdings:
        try:
            value = h.quantity if h.is_cash else h.quantity * _current_price(h.symbol)
        except ToolError:
            value = h.cost_basis
        total += value
        if not h.is_individual_stock:
            unclassified += value
            continue
        try:
            t = yf.Ticker(h.symbol)
            cap = getattr(t.fast_info, "market_cap", None) or t.info.get("marketCap")
        except Exception:
            cap = None
        if not cap:
            unclassified += value
            continue
        if cap >= 200e9:
            bucket = "mega_cap"
        elif cap >= 10e9:
            bucket = "large_cap"
        elif cap >= 2e9:
            bucket = "mid_cap"
        else:
            bucket = "small_cap"
        buckets[bucket] = buckets.get(bucket, 0.0) + value
    return {
        "client_id": client_id,
        "market_cap_allocation_pct": {k: round(v / total * 100, 2) for k, v in buckets.items()},
        "not_classified_pct": round(unclassified / total * 100, 2),
        "note": "ETFs and cash are not classified — a fund has no single-company market cap.",
    }


# --- LangChain tool wrappers (what the agent binds to) ---
PORTFOLIO_TOOLS = [
    tool(get_holdings),
    tool(get_portfolio_value),
    tool(get_position),
    tool(get_position_performance),
    tool(get_ytd_returns),
    tool(get_allocation_by_sector),
    tool(get_allocation_by_asset_class),
    tool(get_allocation_by_market_cap),
]
