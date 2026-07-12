"""Portfolio tools — the Portfolio agent's only way to touch data.

Phase 3 shape: every network call goes through the adapter CHAIN (vendor-agnostic,
falls back automatically), is CACHED to disk, RETRIED on transient failures, and
each tool REGISTERS itself with the ToolRegistry.

Money rules (docs/data_assumptions.md): cash is valued at quantity × 1 (its
`quantity` IS the dollar balance); performance/YTD are undefined for cash;
market-cap buckets apply to individual stocks only.
"""

from datetime import datetime

from app.config import settings
from app.data.repositories import portfolio_repo
from app.errors.exceptions import ToolError
from app.integrations.chain import market_data
from app.logging import get_logger
from app.tools.decorators import cached, retry
from app.tools.registry import tool_registry

log = get_logger(__name__)


# --- shared, resilient data helpers (not tools themselves) -------------------
@cached(ttl_seconds=settings.CACHE_TTL_QUOTES)
@retry(max_attempts=3)
def _quote(symbol: str) -> dict:
    return market_data.get_quote(symbol)


def _current_price(symbol: str) -> float:
    return float(_quote(symbol)["price"])


@cached(ttl_seconds=settings.CACHE_TTL_QUOTES)
@retry(max_attempts=3)
def _history(symbol: str, period: str) -> dict:
    return market_data.get_price_history(symbol, period)


@cached(ttl_seconds=settings.CACHE_TTL_FUNDAMENTALS)
@retry(max_attempts=3)
def _fundamentals(symbol: str) -> dict:
    return market_data.get_fundamentals(symbol)


# --- the tools ----------------------------------------------------------------
@tool_registry.register(agent="portfolio")
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


@tool_registry.register(agent="portfolio")
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


@tool_registry.register(agent="portfolio")
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


@tool_registry.register(agent="portfolio")
def get_position_performance(client_id: str, symbol: str) -> dict:
    """Total return of one position since purchase (current price vs purchase price). N/A for cash."""
    portfolio = portfolio_repo.get(client_id)
    h = portfolio.position(symbol)
    if h is None:
        return {"held": False, "message": f"Client {client_id} does not hold '{symbol}'."}
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


@tool_registry.register(agent="portfolio")
def get_ytd_returns(client_id: str) -> dict:
    """Year-to-date return per holding (vs the first trading day of this year). Skips cash."""
    portfolio = portfolio_repo.get(client_id)
    returns, failed = [], []
    for h in portfolio.holdings:
        if h.is_cash:
            continue
        try:
            hist = _history(h.symbol, "ytd")
            closes = hist["closes"]
            if len(closes) < 2:
                raise ToolError("insufficient YTD history")
            returns.append({"symbol": h.symbol,
                            "ytd_return_pct": round((closes[-1] / closes[0] - 1) * 100, 2)})
        except ToolError:
            failed.append(h.symbol)
    returns.sort(key=lambda r: r["ytd_return_pct"], reverse=True)
    result = {"client_id": client_id, "as_of": datetime.now().date().isoformat(),
              "ytd_returns": returns}
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
        "allocation_pct": {k: round(v / total * 100, 2)
                           for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
    }


@tool_registry.register(agent="portfolio")
def get_allocation_by_sector(client_id: str) -> dict:
    """Portfolio allocation percentages grouped by sector."""
    return _allocation(client_id, lambda h: h.sector)


@tool_registry.register(agent="portfolio")
def get_allocation_by_asset_class(client_id: str) -> dict:
    """Allocation by asset class — distinguishes Individual Stock vs the ETF types vs Cash Equivalent."""
    return _allocation(client_id, lambda h: h.asset_class)


@tool_registry.register(agent="portfolio")
def get_allocation_by_market_cap(client_id: str) -> dict:
    """Market-cap buckets (mega/large/mid/small) for INDIVIDUAL STOCKS only.

    ETFs and cash have no single-company market cap — their share is reported
    separately as 'not_classified_pct' instead of being silently mislabelled.
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
            cap = _fundamentals(h.symbol).get("market_cap")
        except ToolError:
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


# Agent-facing toolbox — discovered from the registry (Registry pattern), so this
# list maintains itself as tools register above.
PORTFOLIO_TOOLS = tool_registry.tools_for("portfolio")
