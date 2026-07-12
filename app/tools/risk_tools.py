"""Risk tools — the Risk Assessment agent's toolbox.

Numbers-first design: every metric is deterministic pandas math on real price
history; the LLM contextualizes ("beta 1.4 → ~40% more volatile than the market"),
it never computes.

Modelling assumption (stated, not hidden): the lookback series applies TODAY'S
portfolio composition across the whole window — purchase dates are ignored for
risk statistics. Cash contributes its balance at zero volatility.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app.data.repositories import portfolio_repo, profile_repo
from app.errors.exceptions import ToolError
from app.logging import get_logger
from app.tools.portfolio_tools import _current_price, _history
from app.tools.registry import tool_registry

log = get_logger(__name__)

TRADING_DAYS = 252


# ------------------------------------------------------------------ shared math
def _price_series(symbol: str, period: str = "1y") -> pd.Series:
    hist = _history(symbol, period)
    return pd.Series(hist["closes"], index=pd.to_datetime(hist["dates"]))


def _portfolio_value_series(client_id: str, period: str = "1y") -> tuple[pd.Series, float]:
    """(daily portfolio value series, cash balance). Composition held constant."""
    portfolio = portfolio_repo.get(client_id)
    columns, cash_total = {}, 0.0
    for h in portfolio.holdings:
        if h.is_cash:
            cash_total += h.quantity
            continue
        columns[h.symbol] = _price_series(h.symbol, period) * h.quantity
    if not columns:  # an all-cash book
        raise ToolError(f"{client_id} holds only cash — volatility/beta are zero by definition")
    values = pd.DataFrame(columns).dropna().sum(axis=1) + cash_total
    return values, cash_total


def _position_weights(client_id: str) -> list[dict]:
    """[{symbol, weight, asset_class, sector, is_stock}] using market values (cash at balance)."""
    portfolio = portfolio_repo.get(client_id)
    rows, total = [], 0.0
    for h in portfolio.holdings:
        try:
            value = h.quantity if h.is_cash else h.quantity * _current_price(h.symbol)
        except ToolError:
            value = h.cost_basis  # feed down → cost basis still gives usable weights
        rows.append({"symbol": h.symbol, "value": value, "asset_class": h.asset_class,
                     "sector": h.sector, "is_stock": h.is_individual_stock,
                     "is_cash": h.is_cash})
        total += value
    for row in rows:
        row["weight"] = row["value"] / total
    return rows


def _bucket(asset_class: str, is_cash: bool) -> str:
    """Collapse the 16+ asset_class strings into equity / bond / cash / international."""
    if is_cash:
        return "cash"
    if "Bond" in asset_class:
        return "bond"
    if "International" in asset_class or "Emerging" in asset_class:
        return "international"
    return "equity"


# Per-holding risk weights (1 safest … 9 riskiest) for the portfolio risk score.
def _holding_risk_weight(h) -> float:
    if h.is_cash:
        return 1.0
    ac = h.asset_class
    if "Bond" in ac:
        return 2.0
    if h.is_etf:
        if any(k in ac for k in ("Growth", "Technology", "Innovation", "Clean",
                                 "Small", "Emerging")):
            return 7.0  # concentrated/thematic funds
        if "Dividend" in ac:
            return 4.0
        return 5.0      # broad/large/international index funds
    return 8.0          # individual stocks carry idiosyncratic risk


# ------------------------------------------------------------------ VaR — Strategy pattern
class VaRStrategy(ABC):
    name: str

    @abstractmethod
    def compute(self, daily_returns: pd.Series, portfolio_value: float,
                confidence: float) -> dict: ...


class HistoricalVaR(VaRStrategy):
    """Empirical quantile of actually-observed daily returns — no distribution assumed."""

    name = "historical"

    def compute(self, daily_returns, portfolio_value, confidence):
        var_pct = -float(np.quantile(daily_returns, 1 - confidence)) * 100
        return {"method": self.name, "confidence": confidence,
                "var_1d_pct": round(var_pct, 2),
                "var_1d_dollars": round(portfolio_value * var_pct / 100, 2)}


class ParametricVaR(VaRStrategy):
    """Normal-distribution VaR: z(confidence) × σ(daily). Fast, assumes normality."""

    name = "parametric"
    _Z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}

    def compute(self, daily_returns, portfolio_value, confidence):
        z = self._Z.get(round(confidence, 2))
        if z is None:
            raise ToolError(f"Parametric VaR supports confidence in {sorted(self._Z)}")
        var_pct = z * float(daily_returns.std()) * 100
        return {"method": self.name, "confidence": confidence,
                "var_1d_pct": round(var_pct, 2),
                "var_1d_dollars": round(portfolio_value * var_pct / 100, 2)}


_VAR_STRATEGIES: dict[str, VaRStrategy] = {s.name: s for s in (HistoricalVaR(), ParametricVaR())}


# ------------------------------------------------------------------ the tools
@tool_registry.register(agent="risk")
def portfolio_volatility(client_id: str, lookback_days: int = 252) -> dict:
    """Annualized volatility (stddev of daily portfolio returns). Cash is zero-volatility."""
    values, cash = _portfolio_value_series(client_id)
    returns = values.pct_change().dropna().tail(lookback_days)
    annualized = float(returns.std()) * math.sqrt(TRADING_DAYS) * 100
    return {
        "client_id": client_id,
        "annualized_volatility_pct": round(annualized, 2),
        "lookback_days": int(len(returns)),
        "cash_balance_dampening": round(cash, 2),
        "note": "cash holds value constant, damping portfolio swings",
    }


# LLMs pass human names ("S&P 500") where tickers belong — absorb that at the boundary.
_BENCHMARK_ALIASES = {
    "S&P 500": "SPY", "S&P500": "SPY", "SP500": "SPY", "S&P": "SPY", "MARKET": "SPY",
    "NASDAQ": "QQQ", "NASDAQ 100": "QQQ", "NASDAQ-100": "QQQ",
    "DOW": "DIA", "DOW JONES": "DIA",
}


@tool_registry.register(agent="risk")
def portfolio_beta(client_id: str, benchmark: str = "SPY") -> dict:
    """Portfolio beta vs a benchmark ETF ticker (default SPY = S&P 500). Cash and bond
    holdings are excluded from the beta calculation and reported separately (beta is
    an equity-risk concept)."""
    benchmark = _BENCHMARK_ALIASES.get(benchmark.strip().upper(), benchmark.strip().upper())
    portfolio = portfolio_repo.get(client_id)
    equity_cols, excluded_value, equity_value = {}, 0.0, 0.0
    for h in portfolio.holdings:
        excluded = h.is_cash or "Bond" in h.asset_class
        try:
            value = h.quantity if h.is_cash else h.quantity * _current_price(h.symbol)
        except ToolError:
            value = h.cost_basis
        if excluded:
            excluded_value += value
            continue
        equity_value += value
        equity_cols[h.symbol] = _price_series(h.symbol) * h.quantity
    if not equity_cols:
        return {"client_id": client_id, "beta": 0.0,
                "note": "no equity holdings — beta is not meaningful for this book"}
    equity_series = pd.DataFrame(equity_cols).dropna().sum(axis=1)
    bench = _price_series(benchmark)
    joined = pd.concat([equity_series.pct_change(), bench.pct_change()],
                       axis=1, keys=["port", "bench"]).dropna()
    beta = float(joined["port"].cov(joined["bench"]) / joined["bench"].var())
    total = equity_value + excluded_value
    return {
        "client_id": client_id,
        "benchmark": benchmark,
        "beta_equity_sleeve": round(beta, 2),
        "equity_share_pct": round(equity_value / total * 100, 2),
        "excluded_cash_bonds_pct": round(excluded_value / total * 100, 2),
        "note": "beta covers the equity sleeve only; cash/bonds reported separately",
    }


@tool_registry.register(agent="risk")
def value_at_risk(client_id: str, confidence: float = 0.95,
                  method: str = "historical") -> dict:
    """1-day Value at Risk. method='historical' (empirical quantile) or
    'parametric' (normal approximation) — two Strategy implementations."""
    strategy = _VAR_STRATEGIES.get(method)
    if strategy is None:
        raise ToolError(f"Unknown VaR method '{method}'. Use: {sorted(_VAR_STRATEGIES)}")
    values, _ = _portfolio_value_series(client_id)
    returns = values.pct_change().dropna()
    result = strategy.compute(returns, float(values.iloc[-1]), confidence)
    result["client_id"] = client_id
    result["interpretation"] = (f"On the worst {round((1 - confidence) * 100)}% of days, "
                                f"expect to lose at least {result['var_1d_pct']}% "
                                f"(≈${result['var_1d_dollars']:,.0f}) in one day.")
    return result


@tool_registry.register(agent="risk")
def concentration_metrics(client_id: str) -> dict:
    """Concentration: top-5 weight, Herfindahl index, sector AND asset-class
    concentration, plus single-issuer concentration (largest individual stock)."""
    rows = _position_weights(client_id)
    weights = sorted((r["weight"] for r in rows), reverse=True)
    top5 = sum(weights[:5]) * 100
    hhi = sum(w * w for w in weights)  # 0..1; >0.25 ≈ highly concentrated

    sectors: dict[str, float] = {}
    buckets: dict[str, float] = {}
    for r in rows:
        sectors[r["sector"]] = sectors.get(r["sector"], 0) + r["weight"]
        b = _bucket(r["asset_class"], r["is_cash"])
        buckets[b] = buckets.get(b, 0) + r["weight"]
    top_sector = max(sectors.items(), key=lambda kv: kv[1])

    stocks = [r for r in rows if r["is_stock"]]
    single_issuer = max(stocks, key=lambda r: r["weight"]) if stocks else None
    return {
        "client_id": client_id,
        "top_5_weight_pct": round(top5, 2),
        "herfindahl_index": round(hhi, 4),
        "hhi_reading": "highly concentrated" if hhi > 0.25
                       else "moderately concentrated" if hhi > 0.15 else "diversified",
        "largest_sector": {"sector": top_sector[0], "weight_pct": round(top_sector[1] * 100, 2)},
        "asset_class_split_pct": {k: round(v * 100, 2) for k, v in
                                  sorted(buckets.items(), key=lambda kv: -kv[1])},
        "single_issuer_concentration": (
            {"symbol": single_issuer["symbol"], "weight_pct": round(single_issuer["weight"] * 100, 2)}
            if single_issuer else {"note": "no individual stocks in this book"}),
    }


@tool_registry.register(agent="risk")
def risk_tolerance_check(client_id: str) -> dict:
    """Compare the portfolio's computed risk score (1-10, from the asset mix) against
    the client's profiled tolerance. A 40% cash+bond book scores far lower than
    100% individual stocks even with identical equity picks."""
    profile = profile_repo.get(client_id)
    if profile is None:
        return {"client_id": client_id, "mismatch": None,
                "message": "No client profile on file — seed data/profiles first."}

    portfolio = portfolio_repo.get(client_id)
    rows = _position_weights(client_id)
    weight_by_symbol = {r["symbol"]: r["weight"] for r in rows}
    score = sum(_holding_risk_weight(h) * weight_by_symbol.get(h.symbol, 0)
                for h in portfolio.holdings)
    portfolio_score = round(score, 1)
    gap = portfolio_score - profile.risk_score
    mismatch = abs(gap) >= 3

    direction = ("portfolio is RISKIER than the client's tolerance" if gap > 0
                 else "portfolio is more CONSERVATIVE than the client could accept")
    return {
        "client_id": client_id,
        "profile": {"risk_tolerance": profile.risk_tolerance,
                    "risk_score": profile.risk_score,
                    "time_horizon_years": profile.time_horizon_years,
                    "income_needs": profile.income_needs,
                    "note": "profile is synthetic — not provided in source data"},
        "portfolio_risk_score": portfolio_score,
        "gap": round(gap, 1),
        "mismatch": mismatch,
        "rationale": (f"Portfolio risk {portfolio_score}/10 vs tolerance "
                      f"{profile.risk_score}/10 → {direction}." if mismatch else
                      f"Portfolio risk {portfolio_score}/10 is within tolerance "
                      f"of {profile.risk_score}/10."),
    }


@tool_registry.register(agent="risk")
def regulatory_flags(client_id: str) -> dict:
    """ILLUSTRATIVE compliance stub: wash-sale windows and position-limit checks.
    Not legal advice — a real system would integrate a compliance engine."""
    portfolio = portfolio_repo.get(client_id)
    rows = _position_weights(client_id)
    flags = []
    for r in rows:
        if not r["is_cash"] and r["weight"] > 0.25:
            flags.append({"type": "position_limit", "symbol": r["symbol"],
                          "detail": f"{r['symbol']} is {round(r['weight'] * 100, 1)}% of the "
                                    f"portfolio (illustrative 25% single-position limit)"})
    recent = [h.symbol for h in portfolio.holdings
              if not h.is_cash and (pd.Timestamp.now() - pd.Timestamp(h.purchase_date)).days <= 30]
    if recent:
        flags.append({"type": "wash_sale_window", "symbols": recent,
                      "detail": "bought within 30 days — selling at a loss could trigger "
                                "wash-sale rules (illustrative)"})
    return {"client_id": client_id, "flags": flags,
            "disclaimer": "Illustrative checks only — not compliance or legal advice."}


RISK_TOOLS = tool_registry.tools_for("risk")
