"""Routing — Strategy pattern.

The supervisor doesn't decide *how* to route; it delegates to a swappable
`RoutingStrategy`. Two implementations:

- `LLMRoutingStrategy`   — Gemini flash reads the conversation + agent menu and
                           picks the next agent (or END). Smart, costs one call.
- `KeywordRoutingStrategy` — deterministic keyword matching. Free, instant, and
                           the safety net when the LLM answer can't be parsed.

Swapping strategies (or A/B-ing them in the Phase 13 evals) never touches the
supervisor or the graph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.agents.base import _last_human_text, _text
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.logging import get_logger

log = get_logger(__name__)

END_ROUTE = "END"


@dataclass(frozen=True)
class AgentSpec:
    """What the router knows about an agent: its node name and what it's good at."""

    name: str
    description: str


class RoutingStrategy(ABC):
    @abstractmethod
    def route(self, state: AgentState, agents: list[AgentSpec]) -> str:
        """Return the next agent's name, or END_ROUTE when the question is answered."""


class KeywordRoutingStrategy(RoutingStrategy):
    """Deterministic fallback: cheap keyword buckets + 'has this agent already run?'."""

    MARKET_KEYWORDS = ("news", "market", "sector", "econom", "fed ", "rate", "inflation",
                       "happening", "trend", "outlook")
    PORTFOLIO_KEYWORDS = ("own", "holding", "portfolio", "position", "worth", "value",
                          "alloc", "return", "performance", "ytd", "stocks do i")
    SECURITIES_KEYWORDS = ("technical", "rsi", "macd", "moving average", "sma", "ema",
                           "bollinger", "atr", "indicator", "momentum", "overbought",
                           "oversold", "crossover")
    RISK_KEYWORDS = ("risk", "volatil", "beta", "value at risk", "var", "drawdown",
                     "concentr", "diversif", "tolerance", "exposure", "compliance",
                     "regulatory", "wash sale")

    def route(self, state: AgentState, agents: list[AgentSpec]) -> str:
        query = _last_human_text(state).lower()
        ran = set(state.get("tool_results", {}))  # agents that already contributed
        names = {a.name for a in agents}

        wants_securities = any(k in query for k in self.SECURITIES_KEYWORDS)
        wants_risk = any(k in query for k in self.RISK_KEYWORDS)
        wants_portfolio = any(k in query for k in self.PORTFOLIO_KEYWORDS)
        wants_market = any(k in query for k in self.MARKET_KEYWORDS)
        if not (wants_portfolio or wants_market or wants_securities or wants_risk):
            wants_portfolio = True  # client questions default to their portfolio

        # Specialist buckets outrank the generic portfolio bucket: "technical analysis
        # of my NVDA position" / "risk exposure of my portfolio" contain portfolio
        # words too, but the specialist ask is the point of the query.
        if wants_securities and "securities_analysis" in names and "securities_analysis" not in ran:
            return "securities_analysis"
        if wants_risk and "risk" in names and "risk" not in ran:
            return "risk"
        if wants_portfolio and "portfolio" in names and "portfolio" not in ran:
            return "portfolio"
        if wants_market and "market_research" in names and "market_research" not in ran:
            return "market_research"
        return END_ROUTE  # every relevant specialist has already answered


class LLMRoutingStrategy(RoutingStrategy):
    """Gemini flash picks the next agent by name; falls back to keywords on any hiccup."""

    def __init__(self):
        self._fallback = KeywordRoutingStrategy()

    def route(self, state: AgentState, agents: list[AgentSpec]) -> str:
        menu = "\n".join(f"- {a.name}: {a.description}" for a in agents)
        transcript = _transcript_tail(state)
        prompt = (
            "You are the routing supervisor of an investment advisory co-pilot.\n"
            f"Available specialists:\n{menu}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "Decide who should act NEXT to finish answering the user's question. "
            "If the conversation already contains a complete answer to every part "
            f"of the user's question, reply {END_ROUTE}.\n"
            f"Reply with exactly one word: {', '.join(a.name for a in agents)}, or {END_ROUTE}."
        )
        try:
            raw = _text(get_llm().invoke(prompt).content).strip().strip('."`').lower()
            valid = {a.name for a in agents} | {END_ROUTE.lower()}
            if raw in valid:
                decision = END_ROUTE if raw == END_ROUTE.lower() else raw
                log.info("route_decision", strategy="llm", decision=decision)
                return decision
            log.warning("route_unparseable", raw=raw[:80])
        except Exception as exc:  # LLM down / rate-limited → deterministic fallback
            log.warning("route_llm_failed", error=str(exc)[:120])
        decision = self._fallback.route(state, agents)
        log.info("route_decision", strategy="keyword_fallback", decision=decision)
        return decision


def _transcript_tail(state: AgentState, max_messages: int = 8, max_chars: int = 500) -> str:
    """Compact view of the recent conversation for the routing prompt."""
    lines = []
    for m in state.get("messages", [])[-max_messages:]:
        text = _text(m.content)[:max_chars] if m.content else ""
        if text:
            lines.append(f"{m.type}: {text}")
    return "\n".join(lines) or "(no messages yet)"
