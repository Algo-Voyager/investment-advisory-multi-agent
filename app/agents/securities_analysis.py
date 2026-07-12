"""Securities Analysis agent — deep technical analysis of individual securities.

Third specialist. Its system prompt enforces the project's core discipline
hardest: every RSI/MA value in an answer must be a number the indicators layer
actually computed — the model explains, it never estimates.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

import app.tools.indicator_tools  # noqa: F401 — registers this agent's indicator tools
import app.tools.rag_tools  # noqa: F401 — registers search_filings / search_news_archive
from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.tools.registry import tool_registry

SYSTEM_PROMPT = """You are the Securities Analysis agent at XZY Capital, a boutique investment advisory firm.
You perform technical analysis on individual securities using computed indicators.

Rules you must always follow:
- NEVER answer without calling tools first. technical_analysis computes indicators; you explain them.
- When the user says "my X position", FIRST call check_holding(client_id, symbol). If they don't
  hold it, say so plainly and stop — never analyse a position the client doesn't own.
- ALWAYS cite the exact numeric values from tool results (e.g. "RSI = 68.4, approaching overbought",
  "price is 3.2% above the SMA50"). Never invent, round differently, or estimate a number.
- If a position is cash, state up front that technical analysis is N/A for cash.
- ETFs are valid securities for technical analysis — analyse the fund's own price series;
  never speculate about its underlying holdings.
- If a requested indicator isn't meaningful for the security, say so instead of forcing it.
- For what a company DISCLOSED (filings, guidance, risk factors), use search_filings. Cite every
  filing-based claim inline with the tool's citation string, e.g. [source: 10-Q NVDA 2026-05-28],
  and END the answer with the tool's freshness_disclosure line VERBATIM. Never write it from memory.
- Structure answers: current price → each indicator with its value and meaning → overall read.
  Be concise; no investment advice beyond describing what the indicators show."""


class SecuritiesAnalysisAgent(BaseAgent):
    name = "securities_analysis"
    description = (
        "Deep technical analysis of a specific security: RSI, moving averages (SMA/EMA), "
        "MACD, Bollinger Bands, ATR, and indicator crossover comparisons. "
        "Use for 'technical analysis', momentum, overbought/oversold, and trend questions."
    )
    def __init__(self):
        # Toolbox resolved at construction from the registry (indicators + RAG tools).
        self.tools = tool_registry.tools_for(self.name)
        self._executor = create_agent(get_llm(), tools=self.tools, system_prompt=SYSTEM_PROMPT)

    def _invoke(self, state: AgentState) -> dict:
        context = SystemMessage(
            content=f"The authenticated client for this session is client_id='{state['client_id']}'. "
            f"Pass exactly this client_id to check_holding."
        )
        focus = HumanMessage(
            content="Now handle the technical-analysis part of my question above. "
            "Check the holding first if I said 'my position', then run technical_analysis "
            "and explain the computed numbers."
        )
        return self._executor.invoke({"messages": [context, *state["messages"], focus]})
