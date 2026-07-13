"""Access control — Interceptor pattern. The data-privacy enforcement point.

Layer 1 (structural): thread_id = f"{client_id}-{session_id}" keeps every client's
conversation history in disjoint checkpointer threads, and the MemoryStore keys
long-term memory by client_id.

Layer 2 (this file, belt-and-braces): the AUTHENTICATED client for the current
run is bound to a ContextVar by BaseAgent._preprocess, and every tool that takes
a client_id verifies the request against it. A prompt-injected "show me CLT-001's
portfolio" inside a CLT-002 session dies HERE with a PermissionError, no matter
how convincingly the LLM was tricked into calling the tool.

ContextVar is task/thread-local, so concurrent sessions for different clients
cannot bleed into each other (proven by tests/test_phase7.py's gather test).

When no session client is bound (direct tool calls in tests/notebooks — trusted
developer context), the check passes through.
"""

from contextvars import ContextVar

from app.logging import get_logger

log = get_logger(__name__)

_session_client: ContextVar[str | None] = ContextVar("session_client", default=None)


def set_session_client(client_id: str | None) -> None:
    """Bind the authenticated client for the current execution context.
    Called by BaseAgent._preprocess from graph state — NEVER from chat text."""
    _session_client.set(client_id)


def get_session_client() -> str | None:
    return _session_client.get()


def verify_client_access(requested_client_id: str,
                         session_client_id: str | None = None) -> None:
    """Raise PermissionError if a tool call reaches for another client's data.

    session_client_id defaults to the ContextVar-bound authenticated client.
    """
    session = session_client_id if session_client_id is not None else _session_client.get()
    if session is None:
        return  # no session bound — direct developer/test invocation
    if requested_client_id != session:
        log.warning("access_denied", session_client=session, requested=requested_client_id)
        raise PermissionError(
            f"Client {session} may not access data for client {requested_client_id}")
