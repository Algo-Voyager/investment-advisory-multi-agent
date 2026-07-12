"""Risk Assessment agent — the conservative risk officer.

Fourth specialist. Personality is deliberate: it contextualizes every number in
plain language and ALWAYS reports the equity/bond/cash split before diving into
equity risk — a 40% cash+bond book is a different animal from 100% stocks even
with identical equity picks.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

import app.tools.risk_tools  # noqa: F401 — registers this agent's tools
from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.tools.registry import tool_registry

SYSTEM_PROMPT = """You are the Risk Assessment agent at XZY Capital — a conservative risk officer.

Rules you must always follow:
- NEVER answer without calling tools first; every number in your answer must come from a tool result.
- ALWAYS begin with the portfolio's asset-class split (equity / bond / cash / international, from
  concentration_metrics) before discussing equity risk — composition frames everything else.
- CONTEXTUALIZE every metric in plain language: "beta of 1.4 means the portfolio tends to move about
  40% more than the market"; "VaR of 2.1% means on the worst 5% of days you'd expect to lose at
  least that much in a day". Never drop a bare number.
- For tolerance questions, call risk_tolerance_check and state the mismatch flag PLAINLY when it
  fires: what the profile says, what the portfolio scores, and the direction of the gap.
- Note that client profiles are synthetic assumptions when reporting profile data.
- Frame findings as risks and observations, never as trading instructions.
- Be measured and precise — a risk officer, not a salesperson."""


class RiskAssessmentAgent(BaseAgent):
    name = "risk"
    description = (
        "Portfolio risk analysis: volatility, beta, Value-at-Risk, concentration "
        "(sector/asset-class/single-issuer), risk-tolerance alignment, and "
        "illustrative regulatory flags. Use for 'how risky', 'exposure', 'am I "
        "aligned with my tolerance', and diversification questions."
    )

    def __init__(self):
        self.tools = tool_registry.tools_for(self.name)
        self._executor = create_agent(get_llm(), tools=self.tools, system_prompt=SYSTEM_PROMPT)

    def _invoke(self, state: AgentState) -> dict:
        context = SystemMessage(
            content=f"The authenticated client for this session is client_id='{state['client_id']}'. "
            f"Pass exactly this client_id to every tool."
        )
        focus = HumanMessage(
            content="Now handle the risk-assessment part of my question above. Start from the "
            "asset-class split, then the requested metrics, with each number contextualized."
        )
        return self._executor.invoke({"messages": [context, *state["messages"], focus]})
