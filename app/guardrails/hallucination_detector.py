"""Hallucination detectors — audit the synthesized answer against the evidence.

Ordered cheap→expensive (the pipeline short-circuits on the first non-pass):
1. NumericConsistencyGuard  — no LLM. Every significant number in the answer must
   trace to a tool result (or a trivial derivation of one). Catches invented figures.
2. ConflictDisclosureGuard  — no LLM. If the evidence contains opposing signals but
   the answer doesn't acknowledge them, ask for a revision (ties to Phase 8's
   conflict-handling requirement).
3. CitationCoverageGuard    — Gemini judge. Are the answer's factual claims supported?
4. GroundednessGuard        — Gemini Chain-of-Verification against retrieved context.

LLM guards are gated by settings.ENABLE_LLM_GUARDRAILS and fail OPEN (pass on error)
— a flaky judge must never block a correct answer.
"""

import json
import re

from app.config import settings
from app.guardrails.base import Guardrail, GuardrailResult
from app.llm.factory import get_llm
from app.logging import get_logger

log = get_logger(__name__)

_NUM = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")


def _numbers(text: str) -> list[float]:
    out = []
    for token in _NUM.findall(text or ""):
        cleaned = token.replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _evidence_text(state) -> str:
    parts = [json.dumps(state.get("tool_results", {}), default=str)]
    parts.extend(state.get("retrieved_context", []) or [])
    return " ".join(parts)


class NumericConsistencyGuard(Guardrail):
    """Every significant number in the answer must appear in — or be a trivial
    derivation of — the tool evidence. Years and small counts (≤31) are exempt."""

    name = "numeric_consistency"
    _TOL = 0.01  # 1% relative tolerance

    def check(self, state) -> GuardrailResult:
        answer = state.get("final_answer") or ""
        evidence_nums = _numbers(_evidence_text(state))
        allowed = set(evidence_nums)
        # trivial derivations: beta/ratio → "% more/less" (t*100, (t-1)*100)
        for t in evidence_nums:
            if 0 < t < 5:
                allowed.update({round(t * 100, 2), round((t - 1) * 100, 2), round((1 - t) * 100, 2)})

        unmatched = []
        for n in _numbers(answer):
            if 1900 <= n <= 2100 or (n == int(n) and abs(n) <= 31):
                continue  # years, small counts/dates
            if not any(_close(n, a) for a in allowed):
                unmatched.append(n)
        if unmatched:
            return GuardrailResult(self.name, passed=False, action="revise",
                                   reason=f"answer contains numbers not found in tool "
                                          f"evidence: {unmatched}")
        return GuardrailResult(self.name, passed=True)


class ConflictDisclosureGuard(Guardrail):
    """If the evidence holds opposing signals but the answer reads one-sided, revise."""

    name = "conflict_disclosure"
    _POS = ("bullish", "outperform", "gain", "growth", "positive", "strong", "buy", "above")
    _NEG = ("bearish", "underperform", "loss", "decline", "negative", "weak", "sell",
            "below", "oversold", "mismatch", "overbought")
    _ACK = ("however", "conflict", "mixed", "on the other hand", "but ", "tension",
            "although", "despite", "whereas", "caveat", "diverg")

    def check(self, state) -> GuardrailResult:
        evidence = _evidence_text(state).lower()
        has_pos = any(w in evidence for w in self._POS)
        has_neg = any(w in evidence for w in self._NEG)
        if has_pos and has_neg:
            answer = (state.get("final_answer") or "").lower()
            if not any(a in answer for a in self._ACK):
                return GuardrailResult(self.name, passed=False, action="revise",
                                       reason="evidence has conflicting signals but the "
                                              "answer does not acknowledge the tension")
        return GuardrailResult(self.name, passed=True)


class CitationCoverageGuard(Guardrail):
    """Gemini judge: is every factual claim in the answer supported by the evidence?"""

    name = "citation_coverage"

    def check(self, state) -> GuardrailResult:
        if not settings.ENABLE_LLM_GUARDRAILS:
            return GuardrailResult(self.name, passed=True, reason="LLM guards disabled")
        answer = state.get("final_answer") or ""
        if len(answer) < 40:
            return GuardrailResult(self.name, passed=True)
        prompt = (
            "You are a fact-checker. Given EVIDENCE and an ANSWER, decide whether every "
            "factual/numeric claim in the answer is supported by the evidence. Ignore "
            "generic advice and disclaimers. Reply ONLY JSON: "
            '{"supported": true/false, "unsupported": ["..."]}.\n\n'
            f"EVIDENCE:\n{_evidence_text(state)[:3000]}\n\nANSWER:\n{answer[:2000]}")
        from app.agents.base import _text

        raw = _text(get_llm(reasoning=True).invoke(prompt).content).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        verdict = json.loads(raw)
        if not verdict.get("supported", True):
            return GuardrailResult(self.name, passed=False, action="revise",
                                   reason=f"unsupported claims: {verdict.get('unsupported', [])[:3]}")
        return GuardrailResult(self.name, passed=True)


class GroundednessGuard(Guardrail):
    """Chain-of-Verification against retrieved RAG context. Only runs when the answer
    used filings (retrieved_context present)."""

    name = "groundedness"

    def check(self, state) -> GuardrailResult:
        if not settings.ENABLE_LLM_GUARDRAILS:
            return GuardrailResult(self.name, passed=True, reason="LLM guards disabled")
        context = state.get("retrieved_context") or []
        if not context:
            return GuardrailResult(self.name, passed=True, reason="no retrieved context")
        answer = state.get("final_answer") or ""
        prompt = (
            "Chain-of-verification. From the ANSWER, derive up to 3 factual claims, then "
            "check each against the CONTEXT. Reply ONLY JSON "
            '{"grounded": true/false, "issues": ["..."]}.\n\n'
            f"CONTEXT:\n{' '.join(context)[:3000]}\n\nANSWER:\n{answer[:2000]}")
        from app.agents.base import _text

        raw = _text(get_llm(reasoning=True).invoke(prompt).content).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        verdict = json.loads(raw)
        if not verdict.get("grounded", True):
            return GuardrailResult(self.name, passed=False, action="revise",
                                   reason=f"not grounded in filings: {verdict.get('issues', [])[:3]}")
        return GuardrailResult(self.name, passed=True)


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol
