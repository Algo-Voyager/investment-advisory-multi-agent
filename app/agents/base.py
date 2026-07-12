"""BaseAgent — Template Method pattern.

`run()` defines the invariant shape of every agent's execution:

    run(state) → _preprocess() → _invoke() → _postprocess()

Subclasses override ONLY `_invoke()` and declare `name` / `description` / `tools`.
Pre/post handle the boring-but-important parts uniformly: log context binding,
timing, and persisting tool outputs into `state.tool_results` so later phases
(guardrails, synthesizer) can audit exactly what the tools really said.

Note for react-style agents: `create_agent(...)` (langchain.agents) returns a compiled graph.
The Template Method wraps the *invocation* of that executor — it does not
replace the react loop.
"""

import time
from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.state import AgentState
from app.logging import bind_context, get_logger

log = get_logger(__name__)


class BaseAgent(ABC):
    name: str
    description: str
    tools: list = []

    def run(self, state: AgentState) -> dict:
        """The template — same skeleton for every agent."""
        self._preprocess(state)
        started = time.perf_counter()
        result = self._invoke(state)  # ← the only step subclasses implement
        return self._postprocess(state, result, elapsed=time.perf_counter() - started)

    # -- template steps -------------------------------------------------
    def _preprocess(self, state: AgentState) -> None:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"), agent=self.name)
        log.info("agent_start", query=_last_human_text(state))

    @abstractmethod
    def _invoke(self, state: AgentState) -> dict:
        """Do the actual work; return {'messages': [...]} in create_agent shape."""

    def _postprocess(self, state: AgentState, result: dict, elapsed: float) -> dict:
        new_messages = result.get("messages", [])[len(state.get("messages", [])):]
        # Persist raw tool outputs — guardrails (Phase 9) cross-check answers against these.
        tool_outputs = [m.content for m in new_messages if isinstance(m, ToolMessage)]
        tool_results = dict(state.get("tool_results", {}))
        if tool_outputs:
            tool_results[self.name] = tool_outputs
        log.info("agent_done", seconds=round(elapsed, 2), new_messages=len(new_messages),
                 tool_calls=len(tool_outputs))
        return {"messages": new_messages, "tool_results": tool_results}

    # -- convenience ----------------------------------------------------
    def answer(self, client_id: str, query: str, session_id: str = "adhoc") -> str:
        """One-shot helper for notebooks/tests: ask this agent a question directly."""
        state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "client_id": client_id,
            "session_id": session_id,
        }
        update = self.run(state)
        for message in reversed(update["messages"]):
            if isinstance(message, AIMessage) and message.content:
                return _text(message.content)
        return "(no answer produced)"


def _last_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _text(message.content)
    return ""


def _text(content) -> str:
    """Message content can be a string or a list of content blocks — normalize to text."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
