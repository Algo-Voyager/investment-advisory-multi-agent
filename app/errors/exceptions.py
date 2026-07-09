"""Custom exception hierarchy.

Everything raised on purpose inside the co-pilot derives from `CoPilotError`, so
graph-level handlers (Phase 11's SafeExitNode) can distinguish "our" failures from
genuine bugs and never leak a stack trace to the user.
"""


class CoPilotError(Exception):
    """Base class for all co-pilot errors."""


class ToolError(CoPilotError):
    """A tool failed in a way the agent should explain honestly (bad ticker, API down...)."""


class RetrievalError(CoPilotError):
    """Knowledge-base retrieval failed (Chroma unavailable, empty collection...)."""


class HallucinationError(CoPilotError):
    """A guardrail determined the answer is not grounded in tool results / retrieved context."""


class ClarificationNeeded(CoPilotError):
    """The query is ambiguous; the graph should interrupt and ask the user."""

    def __init__(self, question: str, options: list[str] | None = None):
        super().__init__(question)
        self.question = question
        self.options = options or []


class RateLimitError(ToolError):
    """An upstream API (market data or Gemini) returned a rate-limit response."""
