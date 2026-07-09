"""Structured logging via structlog.

- Pretty, colored console output when attached to a TTY (dev).
- JSON lines otherwise (prod / piped / eval harness), so guardrail decisions and
  tool errors are machine-parseable for the Phase 13 evaluation report.
- `bind_context(client_id=..., session_id=..., agent=...)` attaches request context
  to every subsequent log line via contextvars (works across async boundaries).

Usage:
    from app.logging import get_logger, bind_context
    log = get_logger(__name__)
    bind_context(client_id="CLT-001", session_id="sess-1", agent="portfolio")
    log.info("tool_call", tool="get_holdings")
"""

import logging
import sys

import structlog

from app.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent structlog + stdlib logging configuration."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,   # picks up client_id/session_id/agent
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()            # pretty in dev (TTY)
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()   # JSON in prod / pipes
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "xzy-copilot"):
    configure_logging()
    return structlog.get_logger(name)


def bind_context(**kwargs) -> None:
    """Bind request-scoped context (client_id, session_id, agent) to all log lines."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in kwargs.items() if v is not None})


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
