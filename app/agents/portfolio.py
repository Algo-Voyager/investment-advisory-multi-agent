"""Portfolio Analysis agent — the first specialist.

A react-style agent (LangGraph prebuilt) bound to the portfolio tools. The LLM's
job is to pick tools and explain their results — every number in an answer must
come from a tool, never from the model's memory.
"""

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.tools.portfolio_tools import PORTFOLIO_TOOLS

SYSTEM_PROMPT = """You are the Portfolio Analysis agent at XZY Capital, a boutique investment advisory firm.

Rules you must always follow:
- Always call tools with the client_id given in the conversation; never guess or switch client ids.
- Always say "symbol" (never "ticker") to match the firm's data.
- Every number in your answer must come from a tool result in this conversation. Never invent or estimate figures.
- Portfolios here hold Individual Stocks, many kinds of ETFs, and Cash Equivalents. When asked "what do I own?",
  group holdings by asset class first (stocks vs ETFs vs cash), then list symbols within each group.
- Cash positions are valued at their dollar balance; they have no price performance.
- If the client does not hold a requested symbol, say so plainly and list what they do hold instead.
- Be concise and factual, like a portfolio report."""


class PortfolioAgent(BaseAgent):
    name = "portfolio"
    description = (
        "Answers questions about a client's own holdings: what they own, position values, "
        "allocations (sector / asset class / market cap), and performance (since purchase, YTD)."
    )
    tools = PORTFOLIO_TOOLS

    def __init__(self):
        self._executor = create_react_agent(get_llm(), tools=self.tools, prompt=SYSTEM_PROMPT)

    def _invoke(self, state: AgentState) -> dict:
        client_note = SystemMessage(
            content=f"The authenticated client for this session is client_id='{state['client_id']}'. "
            f"Pass exactly this client_id to every tool."
        )
        return self._executor.invoke({"messages": [client_note, *state["messages"]]})
