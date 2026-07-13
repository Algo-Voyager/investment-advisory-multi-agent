"""Input guardrails — run BEFORE the query reaches the planner/agents.

- PIIGuard: redacts SSNs / card / account numbers so they never hit the LLM or
  memory. Action stays "pass" (we sanitize, we don't block); the redacted text is
  returned in metadata for the node to apply.
- PromptInjectionGuard: blocks obvious jailbreak / instruction-override attempts.
- ScopeGuard: refuses clearly off-topic requests (poems, recipes, code) with a
  helpful redirect — conservative, so real finance questions are never blocked.
"""

import re

from app.guardrails.base import Guardrail, GuardrailResult

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_ACCOUNT = re.compile(r"\b(?:acct|account)[ #:]*\d{6,}\b", re.IGNORECASE)

_INJECTION = re.compile(
    r"ignore (all |your |the )?(previous|prior|above) (instructions|prompts?)|"
    r"disregard (the |your )?(system|previous)|"
    r"you are now|forget (your|the) (instructions|rules)|"
    r"reveal (your |the )?(system )?prompt|act as (a |an )?(dan|jailbreak)",
    re.IGNORECASE)

_OFFTOPIC = re.compile(
    r"\b(write|compose|tell) (me )?(a |an )?(poem|story|song|joke|essay|haiku|recipe)\b|"
    r"\b(recipe for|lyrics|screenplay)\b|"
    r"\bwrite (me )?(some |a )?(python|java|c\+\+|code|script|program)\b",
    re.IGNORECASE)


class PIIGuard(Guardrail):
    name = "pii"

    def check(self, state) -> GuardrailResult:
        text = _query(state)
        redacted = _ACCOUNT.sub("[REDACTED-ACCT]",
                    _CARD.sub("[REDACTED-CARD]", _SSN.sub("[REDACTED-SSN]", text)))
        if redacted != text:
            return GuardrailResult(self.name, passed=True, action="pass",
                                   reason="PII redacted before processing",
                                   metadata={"redacted_text": redacted})
        return GuardrailResult(self.name, passed=True)


class PromptInjectionGuard(Guardrail):
    name = "prompt_injection"

    def check(self, state) -> GuardrailResult:
        if _INJECTION.search(_query(state)):
            return GuardrailResult(self.name, passed=False, action="block",
                                   reason="prompt-injection / instruction-override attempt")
        return GuardrailResult(self.name, passed=True)


class ScopeGuard(Guardrail):
    name = "scope"

    def check(self, state) -> GuardrailResult:
        if _OFFTOPIC.search(_query(state)):
            return GuardrailResult(self.name, passed=False, action="block",
                                   reason="off-topic (creative/coding) request")
        return GuardrailResult(self.name, passed=True)


def _query(state) -> str:
    from app.agents.base import _last_human_text

    return _last_human_text(state) if hasattr(state, "get") else str(state)
