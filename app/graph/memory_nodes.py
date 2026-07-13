"""Memory nodes — the graph's read-on-entry / write-on-exit plumbing.

MemoryReadNode  (START → here → supervisor): fetches the client's recent
decisions from long-term memory and injects a compact "prior context" system
message, so a new session starts warm ("what did we discuss last time?").

MemoryWriteNode (supervisor → here → END): persists this interaction (query,
final answer, agents used) so FUTURE sessions can recall it.

Neither node reasons — no LLM calls. They are plumbing, not agents.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.base import _text
from app.graph.state import AgentState
from app.logging import get_logger
from app.memory.store import MemoryStore, get_memory_store

log = get_logger(__name__)


class MemoryReadNode:
    name = "memory_read"

    def __init__(self, store: MemoryStore | None = None):
        self._store = store or get_memory_store()

    def run(self, state: AgentState) -> dict:
        # Reset per-TURN scratch state: in a persisted thread, hops/visited/route
        # survive from the previous user turn and would wrongly block agents now.
        # (messages deliberately persist — that's the conversation.)
        update: dict = {"hops": 0, "visited": [], "route": None}

        decisions = self._store.get_recent_decisions(state["client_id"], limit=5)
        # exclude the current session — its own turns are already in the thread
        prior = [d for d in decisions if d["session_id"] != state.get("session_id")]
        if prior and not _already_briefed(state):
            lines = [f"- [{d['ts'][:10]}] asked: \"{d['query'][:120]}\" → advised: "
                     f"{d['answer'][:180]}" for d in prior]
            update["messages"] = [SystemMessage(content=(
                "Prior context from this client's earlier sessions (long-term memory):\n"
                + "\n".join(lines)
                + "\nUse this to answer questions about previous discussions."))]
            log.info("memory_read", client_id=state["client_id"], prior_decisions=len(prior))
        return update


def _already_briefed(state: AgentState) -> bool:
    """Don't inject the prior-context note twice into the same persisted thread."""
    return any(isinstance(m, SystemMessage)
               and str(m.content).startswith("Prior context from this client's")
               for m in state.get("messages", []))


class MemoryWriteNode:
    name = "memory_write"

    def __init__(self, store: MemoryStore | None = None):
        self._store = store or get_memory_store()

    def run(self, state: AgentState) -> dict:
        # Best-effort: this is the terminal step, right after the user's answer was
        # already computed. A save failure (disk full, locked db) must never turn
        # into an uncaught exception that erases an otherwise-successful answer.
        try:
            query = next((_text(m.content) for m in state.get("messages", [])
                         if isinstance(m, HumanMessage)), "")
            answer = next((_text(m.content) for m in reversed(state.get("messages", []))
                          if isinstance(m, AIMessage) and m.content), "")
            if query and answer:
                self._store.save_decision(
                    client_id=state["client_id"], session_id=state.get("session_id", "unknown"),
                    query=query, answer=answer, agents_used=state.get("visited", []))
        except Exception as exc:  # noqa: BLE001
            log.error("tool_error", tool=self.name, error_type=type(exc).__name__,
                      client_id=state.get("client_id"), session_id=state.get("session_id"),
                      error=str(exc)[:200])
        return {}
