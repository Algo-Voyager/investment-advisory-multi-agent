"""Guardrail framework — Chain of Responsibility.

Each `Guardrail.check(state)` returns a `GuardrailResult` with an action:
- "pass"   → nothing wrong
- "revise" → answer is salvageable; loop back to the synthesizer with the reason
- "block"  → unsafe/off-topic; stop and return a safe reply

`GuardrailPipeline.run` executes guards IN ORDER and short-circuits on the first
non-pass result (so a cheap guard that already fired saves the expensive LLM
guards a call — important on the free Gemini tier). Order guards cheap-and-
blocking first, expensive LLM-judge last.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from app.logging import get_logger

log = get_logger(__name__)

Action = Literal["pass", "block", "revise"]


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    action: Action = "pass"
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class Guardrail(ABC):
    name: str

    @abstractmethod
    def check(self, state) -> GuardrailResult: ...


class GuardrailPipeline:
    def __init__(self, guardrails: list[Guardrail], short_circuit: bool = True):
        self._guards = guardrails
        self._short_circuit = short_circuit

    def run(self, state) -> list[GuardrailResult]:
        results: list[GuardrailResult] = []
        for guard in self._guards:
            try:
                result = guard.check(state)
            except Exception as exc:  # noqa: BLE001 — a broken guard must not break the answer
                log.warning("guardrail_errored", guard=guard.name, error=str(exc)[:120])
                result = GuardrailResult(guard.name, passed=True, action="pass",
                                         reason=f"guard errored, passing: {exc}")
            results.append(result)
            log.info("guardrail_check", guard=result.name, action=result.action,
                     passed=result.passed, reason=result.reason[:120])
            if self._short_circuit and result.action != "pass":
                break
        return results
