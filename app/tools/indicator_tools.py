"""Indicator tools — the Securities Analysis agent's toolbox.

Rules encoded here (not left to the LLM's discretion):
- Cash Equivalents are rejected up front: "N/A — cash position". Cash has no chart.
- ETFs ARE valid — analysed like any security, with NO look-through into holdings.
- All numbers come from the indicators layer; the LLM only explains them.
"""

import pandas as pd

from app.config import settings
from app.data.repositories import portfolio_repo
from app.errors.exceptions import ToolError
from app.indicators import indicator_factory
from app.integrations.chain import market_data
from app.logging import get_logger
from app.tools.decorators import cached, retry
from app.tools.registry import tool_registry

log = get_logger(__name__)


@cached(ttl_seconds=settings.CACHE_TTL_QUOTES)
@retry(max_attempts=3)
def _ohlc_history(symbol: str, period: str) -> dict:
    return market_data.get_price_history(symbol, period)


def _frame(hist: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "close": hist["closes"],
        "high": hist.get("highs", hist["closes"]),   # degrade gracefully if a source lacks OHLC
        "low": hist.get("lows", hist["closes"]),
    }, index=pd.to_datetime(hist["dates"]))
    return df


def _interpret(results: dict) -> str:
    """A compact, deterministic reading assembled from the computed values."""
    notes = []
    for r in results.values():
        ind = r["indicator"]
        if "unavailable" in r:
            notes.append(f"{ind}: unavailable ({r['unavailable'][:60]})")
            continue
        if ind.startswith("rsi"):
            notes.append(f"RSI {r['value']} → {r['zone']}")
        elif ind.startswith(("sma", "ema")):
            notes.append(f"price {r['position']} {ind} ({r['price_vs_ma_pct']:+}%)")
        elif ind == "macd":
            note = f"MACD momentum {r['momentum']}"
            if r["recent_crossover"] != "none":
                note += f" ({r['recent_crossover'].replace('_', ' ')})"
            notes.append(note)
        elif ind.startswith("bollinger"):
            notes.append(f"%B {r['percent_b']} within bands")
        elif ind.startswith("atr"):
            notes.append(f"ATR {r['atr_pct_of_price']}% of price (volatility)")
    return "; ".join(notes)


@tool_registry.register(agent="securities_analysis")
def technical_analysis(symbol: str, indicators: list[str], period: str = "6mo") -> dict:
    """Compute technical indicators (rsi, sma_N, ema_N, macd, bollinger, atr) for a symbol
    and summarize overbought/oversold/trend. Cash positions are rejected (no chart);
    ETFs are analysed like any security, without look-through."""
    if symbol.upper() == "CASH":
        return {"symbol": symbol, "status": "not_applicable",
                "message": "N/A — cash position: cash has no price series to analyse."}
    hist = _ohlc_history(symbol, period)
    frame = _frame(hist)
    close = frame["close"]
    if len(close) < 30:
        raise ToolError(f"Only {len(close)} data points for '{symbol}' ({period}) — "
                        f"not enough for meaningful indicators")

    results = {}
    for name in indicators:
        indicator = indicator_factory.get(name)          # Factory: string → Strategy
        data = frame if name.lower().startswith("atr") else close
        try:
            results[name] = indicator.compute(data)
        except ToolError as exc:
            # One infeasible indicator (SMA200 on 6mo data) must not kill the whole
            # analysis — report it per-indicator so the agent can say so or retry
            # with a longer period.
            results[name] = {"indicator": name, "unavailable": str(exc)}

    return {
        "symbol": symbol,
        "period": period,
        "series_length": len(close),
        "as_of": hist["dates"][-1],
        "current_price": round(float(close.iloc[-1]), 2),
        "indicators": results,
        "summary": _interpret(results),
        "source": hist["source"],
    }


@tool_registry.register(agent="securities_analysis")
def compare_indicators(symbol: str, indicator: str = "sma", windows: list[int] | None = None) -> dict:
    """Compare one indicator across windows (e.g. SMA 20 vs 50 vs 200) and flag crossovers."""
    windows = windows or [20, 50, 200]
    if symbol.upper() == "CASH":
        return {"symbol": symbol, "status": "not_applicable",
                "message": "N/A — cash position: cash has no price series to analyse."}
    hist = _ohlc_history(symbol, "1y")  # long window so SMA200 has data
    close = _frame(hist)["close"]

    values = {}
    for w in windows:
        ind = indicator_factory.get(f"{indicator}_{w}")
        values[f"{indicator}_{w}"] = ind.compute(close)["value"]

    ordered = sorted(windows)
    crossings = []
    for fast, slow in zip(ordered, ordered[1:]):
        fast_v, slow_v = values[f"{indicator}_{fast}"], values[f"{indicator}_{slow}"]
        state = "bullish (fast above slow)" if fast_v > slow_v else "bearish (fast below slow)"
        crossings.append({"pair": f"{indicator}{fast} vs {indicator}{slow}", "state": state})

    return {
        "symbol": symbol,
        "as_of": hist["dates"][-1],
        "current_price": round(float(close.iloc[-1]), 2),
        "values": values,
        "crossovers": crossings,
        "source": hist["source"],
    }


@tool_registry.register(agent="securities_analysis")
def check_holding(client_id: str, symbol: str) -> dict:
    """Check whether the client actually holds a symbol (call FIRST when asked about 'my X position')."""
    portfolio = portfolio_repo.get(client_id)
    h = portfolio.position(symbol)
    if h is None:
        return {"held": False,
                "message": f"Client {client_id} does not hold '{symbol}'. "
                f"They hold: {', '.join(portfolio.symbols)}"}
    return {"held": True, "symbol": h.symbol, "asset_class": h.asset_class,
            "is_cash": h.is_cash, "quantity": h.quantity}


INDICATOR_TOOLS = tool_registry.tools_for("securities_analysis")
