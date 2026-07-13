"""Global per-node exception handling — Interceptor/Middleware pattern.

Any node in the graph can throw (an LLM call errors, a tool raises, a bug).
Without this wrapper, that exception propagates straight out of `graph.stream()`
and the CLI/UI shows a raw Python traceback — exactly what the brief forbids
("no stack trace ever reaches the user").

Mechanic: LangGraph does NOT let a node override a static `add_edge` with a
returned `Command` — both fire and collide (verified empirically). So every
wrapped node must route ONLY via `Command`, never a static edge:

    wrap_dict_node(agent.run, "portfolio", success_goto="supervisor")
        → normal:    Command(goto="supervisor", update=agent.run's dict)
        → exception: Command(goto="safe_exit",  update={"blocked": ..., ...})

    wrap_command_node(supervisor.run, "supervisor")
        → normal:    whatever Command supervisor.run already returns
        → exception: Command(goto="safe_exit", update={...})

Every catch emits ONE structured event —
`{event: "tool_error", tool, error_type, client_id, session_id}` — that Phase 13's
eval report aggregates as a failure-mode table.

CRITICAL: `interrupt()` (Phase 10) pauses a node by RAISING `GraphInterrupt` — that
is LangGraph's own control-flow signal, not an error, and it MUST propagate
untouched so the engine can catch it and pause the graph. A blanket
`except Exception` would swallow it and misreport a clean pause as a crash. Every
wrapper here re-raises `GraphBubbleUp` (GraphInterrupt's base class) before its
generic handler.
"""

from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from app.logging import bind_context, get_logger

log = get_logger(__name__)


def _log_and_build_error_update(node_name: str, exc: Exception, state) -> dict:
    bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"))
    log.error("tool_error", tool=node_name, error_type=type(exc).__name__,
             client_id=state.get("client_id"), session_id=state.get("session_id"),
             error=str(exc)[:200])
    message = (f"I couldn't complete that because the {node_name.replace('_', ' ')} "
              f"step ran into a problem. Please try rephrasing your question.")
    return {"blocked": message,
            "messages": [AIMessage(content=message, name="safe_exit")]}


def wrap_dict_node(fn, node_name: str, success_goto: str, error_goto: str = "safe_exit"):
    """For nodes that normally return a plain state-update dict. Converts them to
    Command-only routing so an exception can redirect to safe_exit."""

    def wrapped(state):
        try:
            update = fn(state) or {}
            return Command(goto=success_goto, update=update)
        except GraphBubbleUp:
            raise  # interrupt()'s pause signal — must reach LangGraph's engine untouched
        except Exception as exc:  # noqa: BLE001 — must catch everything else, this IS the safety net
            return Command(goto=error_goto, update=_log_and_build_error_update(node_name, exc, state))

    wrapped.__name__ = f"safe_{node_name}"
    return wrapped


def wrap_command_node(fn, node_name: str, error_goto: str = "safe_exit"):
    """For nodes that already return a Command (supervisor, reflector, input_guard,
    clarifier). Catches exceptions raised BEFORE they produce that Command."""

    def wrapped(state):
        try:
            return fn(state)
        except GraphBubbleUp:
            raise  # interrupt()'s pause signal — must reach LangGraph's engine untouched
        except Exception as exc:  # noqa: BLE001
            return Command(goto=error_goto, update=_log_and_build_error_update(node_name, exc, state))

    wrapped.__name__ = f"safe_{node_name}"
    return wrapped
