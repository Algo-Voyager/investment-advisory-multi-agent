"""ClarifierNode — Human-in-the-Loop via LangGraph's `interrupt()`.

Runs after the planner. A small Gemini call classifies the query for genuine
ambiguity (multiple holdings could match "tech", an unclear ticker, an undefined
"recent"). If ambiguous, `interrupt(...)` PAUSES the graph — LangGraph persists
the state via the checkpointer (Phase 7) and returns control to the caller. The
caller (CLI/UI) shows the question, gets an answer, and resumes with
`Command(resume=answer)`; execution continues exactly where it paused.

This is the **State pattern**: "waiting for human input" is a first-class graph
state, not a special case bolted on. The resume value plays the role of a
**Callback** into the paused computation.

`settings.ENABLE_CLARIFICATION=False` (set by the eval harness, Phase 13) skips
this node entirely so batch evaluation never blocks on a human.
"""

import json

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from app.agents.base import _last_human_text, _text
from app.config import settings
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.logging import bind_context, get_logger

log = get_logger(__name__)

_PROMPT = """You are the clarification checker for an investment advisory co-pilot.
Decide whether the user's question is genuinely AMBIGUOUS given their portfolio context —
e.g. "my tech holdings" could mean mega-cap (AAPL, MSFT) or high-beta/growth names (SNOW, CRM),
a ticker/company name is unclear, or a time period like "recent" is undefined.
Do NOT flag a question as ambiguous just because it needs multiple tools or specialists —
that is normal and handled elsewhere.

Reply ONLY as JSON: {{"needs_clarification": bool, "question": str, "options": [str, ...]}}
If not ambiguous, needs_clarification must be false (question/options can be empty).

Client's holdings: {holdings}
User's question: {query}"""


class ClarifierNode:
    name = "clarifier"

    def __init__(self, next_node: str):
        self._next = next_node

    def run(self, state: AgentState):
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        from langgraph.types import Command  # local import avoids a cycle with builder

        if not settings.ENABLE_CLARIFICATION:
            return Command(goto=self._next)

        # Already resumed once for this turn? Don't re-ask.
        if state.get("clarification_answer"):
            return Command(goto=self._next)

        query = _last_human_text(state)
        ambiguity = self._detect(state["client_id"], query)

        if not ambiguity.get("needs_clarification"):
            return Command(goto=self._next)

        log.info("clarification_needed", question=ambiguity["question"],
                 options=ambiguity.get("options", []))
        # PAUSE — the checkpointer persists state here; execution resumes below
        # with `answer` bound to whatever Command(resume=...) supplies.
        answer = interrupt({"question": ambiguity["question"],
                            "options": ambiguity.get("options", [])})
        log.info("clarification_resumed", answer=str(answer)[:120])
        return Command(goto=self._next, update={
            "clarification_answer": str(answer),
            "messages": [HumanMessage(content=f"(Clarification) {answer}")],
        })

    @staticmethod
    def _detect(client_id: str, query: str) -> dict:
        from app.data.repositories import portfolio_repo

        try:
            holdings = ", ".join(portfolio_repo.get(client_id).symbols)
        except Exception:
            holdings = "(unavailable)"
        try:
            raw = _text(get_llm().invoke(
                _PROMPT.format(holdings=holdings, query=query)).content).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            if isinstance(result, dict) and "needs_clarification" in result:
                return result
        except Exception as exc:  # noqa: BLE001 — never block a turn on a judge failure
            log.warning("clarifier_detect_failed", error=str(exc)[:120])
        return {"needs_clarification": False}
