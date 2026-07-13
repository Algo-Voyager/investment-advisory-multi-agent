"""Evaluation harness — runs a dataset through the real compiled graph.

Records, per item: latency, agents_used, tools_called, final_answer,
guardrail_events, and whether the run was blocked/errored. Clarification is
force-disabled (`ENABLE_CLARIFICATION=False`) so a batch run never blocks
waiting on a human — exactly what Phase 10's config toggle exists for.

Optional LangSmith tracing (Observer pattern): if `LANGSMITH_API_KEY` is set,
each run is traced automatically via LangChain's global tracer — no extra code
needed here beyond setting the env vars, which `_maybe_enable_langsmith` does.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import _text
from app.config import settings
from app.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "evals" / "datasets"


@dataclass
class EvalResult:
    item_id: str
    client_id: str
    query: str
    final_answer: str
    agents_used: list = field(default_factory=list)
    tools_called: int = 0
    guardrail_events: list = field(default_factory=list)
    blocked: bool = False
    latency_s: float = 0.0
    error: str | None = None
    retrieved_context: list = field(default_factory=list)


def _maybe_enable_langsmith() -> bool:
    if not settings.LANGSMITH_API_KEY:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", "xzy-copilot-evals")
    return True


def load_dataset(name: str) -> list[dict]:
    """name: 'simple_queries', 'analytical_queries', 'rag_queries', 'adversarial_queries'."""
    path = DATASETS_DIR / f"{name}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_item(graph, item: dict, session_id: str = "eval", run_id: str | None = None) -> EvalResult:
    """Invoke the graph for one dataset item and collect everything the metrics need.

    `run_id` scopes the thread_id to THIS harness invocation. Without it, repeated
    `make eval` runs would reuse the same thread_id per item, and Phase 7's
    persisted memory/checkpoint state would leak between runs — making scores
    non-reproducible (observed directly: identical items routed and answered
    differently across two consecutive `make eval` runs before this fix).
    """
    started = time.perf_counter()
    result = EvalResult(item_id=item["id"], client_id=item["client_id"], query=item["query"],
                        final_answer="")
    try:
        state = {
            "messages": [HumanMessage(content=item["query"])],
            "client_id": item["client_id"],
            "session_id": session_id,
        }
        run_id = run_id or "adhoc"
        config = {"configurable": {"thread_id": f"{item['client_id']}-eval-{run_id}-{item['id']}"}}
        out = graph.invoke(state, config)

        result.agents_used = out.get("visited", [])
        result.tools_called = sum(len(v) for v in (out.get("tool_results") or {}).values())
        result.guardrail_events = out.get("guardrail_events", [])
        result.blocked = bool(out.get("blocked"))
        result.retrieved_context = out.get("retrieved_context", [])

        final = out.get("final_answer")
        if not final:
            final = next((_text(m.content) for m in reversed(out.get("messages", []))
                         if isinstance(m, AIMessage) and m.content), "")
        result.final_answer = final
    except Exception as exc:  # noqa: BLE001 — a broken item must not kill the whole eval run
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        log.error("eval_item_failed", item_id=item["id"], error=result.error)
    result.latency_s = time.perf_counter() - started
    return result


def run_dataset(name: str, limit: int | None = None, session_id: str = "eval") -> list[EvalResult]:
    """Build one graph, run every item (or the first `limit`), return results.

    Clarification is disabled for the whole run — see settings.ENABLE_CLARIFICATION.
    """
    from app.graph.builder import GraphBuilder

    _maybe_enable_langsmith()
    original_flag = settings.ENABLE_CLARIFICATION
    settings.ENABLE_CLARIFICATION = False
    run_id = uuid.uuid4().hex[:8]  # unique per run_dataset() call — see run_item()'s docstring
    try:
        graph = GraphBuilder().with_all().build()
        items = load_dataset(name)
        if limit:
            items = items[:limit]
        results = []
        for item in items:
            r = run_item(graph, item, session_id=session_id, run_id=run_id)
            log.info("eval_item_done", item_id=r.item_id, latency_s=round(r.latency_s, 2),
                     agents=r.agents_used, blocked=r.blocked, error=r.error)
            results.append(r)
        return results
    finally:
        settings.ENABLE_CLARIFICATION = original_flag


def results_to_dicts(results: list[EvalResult]) -> list[dict]:
    return [asdict(r) for r in results]
