"""Market Research agent — analyses market conditions, news, and sector performance.

Unlike the Portfolio agent it looks OUTWARD (the market), not at client holdings.
When a query mixes both ("how is my NVDA doing and what's the news?"), the
supervisor sequences the two agents; they share findings via AgentState only.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

import app.tools.market_tools  # noqa: F401 — registers this agent's market tools
import app.tools.rag_tools  # noqa: F401 — registers search_filings / search_news_archive
from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.tools.registry import tool_registry

SYSTEM_PROMPT = """You are the Market Research agent at XZY Capital, a boutique investment advisory firm.

Rules you must always follow:
- Analyse market conditions, news, and sector performance using your tools.
- NEVER answer without calling a tool first. If the question mentions news for a symbol, you MUST
  call get_recent_news(symbol) before replying — even if other agents already discussed that symbol.
- Every number and every headline in your answer must come from a tool result in this conversation.
  Never invent prices, percentages, or news.
- If the news feed returns nothing, say plainly that no recent news is available — do not improvise headlines.
- If asked about economic indicators and the tool reports the feed is unavailable, say so honestly.
- You have NO access to client holdings; if the question needs portfolio data, answer only the market
  part and note that the portfolio specialist covers the rest.
- For questions about what a company SAID or REPORTED (filings, guidance, risks), use search_filings.
  Cite every filing-based claim inline using the tool's citation string, e.g. [source: 10-Q NVDA 2026-05-28].
- When you used search_filings, END your answer with the tool's freshness_disclosure line VERBATIM,
  e.g. "(Filing data as of 2026-05-28)". Never write that line from memory.
- Be concise and factual, like a morning market brief."""


class MarketResearchAgent(BaseAgent):
    name = "market_research"
    description = (
        "Analyses current market conditions: price snapshots, recent news for a symbol, "
        "sector performance, and economic indicators. Knows nothing about the client's holdings."
    )
    def __init__(self):
        # Toolbox resolved at construction time from the registry, so it includes
        # everything registered for this agent (market tools + RAG tools).
        self.tools = tool_registry.tools_for(self.name)
        self._executor = create_agent(get_llm(), tools=self.tools, system_prompt=SYSTEM_PROMPT)

    def _invoke(self, state: AgentState) -> dict:
        context = SystemMessage(
            content=f"The authenticated client for this session is client_id='{state['client_id']}'. "
            f"You cannot access their holdings — cover the market side only."
        )
        # A fresh directive AFTER the transcript: mid-conversation, the last message is
        # often another agent saying "I can't access news" — without this nudge, small
        # models mirror that refusal instead of using their own tools (observed live).
        focus = HumanMessage(
            content="Now handle ONLY the market-research part of my question above "
            "(news, prices, sectors). Use your tools — call get_recent_news for any "
            "symbol I mentioned — and don't repeat what other specialists already covered."
        )
        return self._executor.invoke({"messages": [context, *state["messages"], focus]})
