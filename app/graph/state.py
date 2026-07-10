"""AgentState is the shared blackboard — this IS our inter-agent communication
protocol; agents read/write it, they never call each other directly. The
supervisor (Phase 2) is the only component that decides who runs next, and every
piece of context an agent produces for another agent travels through this state.
"""

from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    client_id: str    # raw "CLT-XXX" — set from the authenticated selection, NEVER from chat input
    session_id: str
    route: Optional[str]              # supervisor's routing decision (Phase 2)
    plan: Optional[list[str]]         # planner's sub-goals (Phase 8)
    retrieved_context: list[str]      # RAG chunks (Phase 5)
    tool_results: dict                # raw tool outputs, keyed by agent — guardrails audit these (Phase 9)
    needs_clarification: Optional[str]  # clarifier's pending question (Phase 10)
