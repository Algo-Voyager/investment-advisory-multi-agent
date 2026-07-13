"""Graph nodes for the trust layer (Phase 9).

InputGuardNode — runs the input guardrail pipeline BEFORE planning. Redacts PII in
place; blocks injection/off-topic queries (→ safe_exit). Interceptor pattern.

ReflectorNode — the Producer/Critic reflection loop. After the synthesizer, it runs
the hallucination pipeline over the answer. On "revise" (within MAX_REVISIONS) it
loops back to the synthesizer with the critique; on "block" it routes to safe_exit;
otherwise the answer ships.

SafeExitNode — turns a block/uncaught failure into an honest, non-technical reply.
No stack trace ever reaches the user.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.config import settings
from app.guardrails.base import GuardrailPipeline
from app.guardrails.hallucination_detector import (
    CitationCoverageGuard,
    ConflictDisclosureGuard,
    GroundednessGuard,
    NumericConsistencyGuard,
)
from app.guardrails.input_guardrails import PIIGuard, PromptInjectionGuard, ScopeGuard
from app.graph.state import AgentState
from app.logging import bind_context, get_logger

log = get_logger(__name__)


def _record(state: AgentState, results) -> list:
    events = list(state.get("guardrail_events", []))
    events.extend({"guard": r.name, "action": r.action, "reason": r.reason} for r in results)
    return events


class InputGuardNode:
    name = "input_guard"

    def __init__(self, next_node: str):
        self._next = next_node
        # cheap + blocking first (PII redaction, injection, scope)
        self._pipeline = GuardrailPipeline([PIIGuard(), PromptInjectionGuard(), ScopeGuard()],
                                           short_circuit=False)

    def run(self, state: AgentState) -> Command:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        results = self._pipeline.run(state)
        update = {"guardrail_events": _record(state, results)}

        for r in results:
            if r.action == "block":
                msg = ("I can only help with questions about your portfolio, markets, "
                       "securities analysis, and risk. Could you rephrase your question "
                       "around those?")
                log.info("input_blocked", guard=r.name, reason=r.reason)
                return Command(goto="safe_exit", update={**update, "blocked": msg})

        # apply PII redaction to the latest human message, if any
        redaction = next((r.metadata.get("redacted_text") for r in results
                          if r.name == "pii" and r.metadata.get("redacted_text")), None)
        if redaction:
            new_messages = list(state.get("messages", []))
            for i in range(len(new_messages) - 1, -1, -1):
                if isinstance(new_messages[i], HumanMessage):
                    new_messages[i] = HumanMessage(content=redaction)
                    break
            update["messages"] = new_messages
            log.info("pii_redacted")
        return Command(goto=self._next, update=update)


class ReflectorNode:
    name = "reflector"

    def __init__(self, next_node: str):
        self._next = next_node
        # cheap (no-LLM) guards first, then LLM judges (short-circuit saves quota)
        self._pipeline = GuardrailPipeline([
            NumericConsistencyGuard(),
            ConflictDisclosureGuard(),
            CitationCoverageGuard(),
            GroundednessGuard(),
        ], short_circuit=True)

    def run(self, state: AgentState) -> Command:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        results = self._pipeline.run(state)
        update = {"guardrail_events": _record(state, results)}
        failing = next((r for r in results if r.action != "pass"), None)

        if failing is None:
            log.info("reflection_pass", revisions=state.get("revisions", 0))
            return Command(goto=self._next, update=update)

        if failing.action == "block":
            return Command(goto="safe_exit", update={
                **update, "blocked": "I couldn't produce a fully verified answer to that. "
                                     "Please try rephrasing, and I'll take another look."})

        revisions = state.get("revisions", 0)
        if revisions >= settings.MAX_REVISIONS:
            # Ship best-effort rather than loop forever — but note it happened.
            log.warning("reflection_max_revisions", reason=failing.reason[:120])
            return Command(goto=self._next, update=update)

        log.info("reflection_revise", attempt=revisions + 1, guard=failing.name,
                 reason=failing.reason[:120])
        return Command(goto="synthesizer", update={
            **update, "revisions": revisions + 1,
            "messages": [SystemMessage(content=f"REVISION REQUIRED: {failing.reason}")]})


class SafeExitNode:
    name = "safe_exit"

    def __init__(self, next_node: str):
        self._next = next_node

    def run(self, state: AgentState) -> Command:
        message = state.get("blocked") or (
            "I'm sorry — I couldn't complete that request. Please try rephrasing.")
        log.info("safe_exit")
        return Command(goto=self._next,
                       update={"final_answer": message,
                               "messages": [AIMessage(content=message, name="safe_exit")]})
