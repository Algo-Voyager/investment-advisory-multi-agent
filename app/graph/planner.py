"""PlannerNode — query decomposition + a swappable complexity classifier.

Runs after the input guardrails and before the supervisor. It embodies the
"is_complex → planner → supervisor" dispatcher: it always runs, classifies the
query, and either writes an ordered PLAN (complex) or leaves it empty (simple).

- ComplexityClassifier is a **Strategy** (Heuristic = free; LLM = a small Gemini
  call) selected by settings.COMPLEXITY_STRATEGY.
- Each plan step is a **Command** the supervisor later executes: {"agent","goal"}.
- The plan is a tiny DSL the supervisor **interprets** step by step.

Everything degrades safely: if decomposition fails to parse, the query is
treated as simple (supervisor falls back to keyword/LLM routing).
"""

import json
from abc import ABC, abstractmethod

from app.agents.base import _last_human_text, _text
from app.graph.router import (
    AgentSpec,
    KeywordRoutingStrategy,
)
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.logging import bind_context, get_logger

log = get_logger(__name__)

_REASONING_VERBS = ("should i", "should we", "recommend", "rebalance", "given",
                    "impact", "what should", "how should", "optimi", "versus",
                    " vs ", "compare", "worth it", "better to")


class ComplexityClassifier(ABC):
    @abstractmethod
    def is_complex(self, query: str) -> bool: ...


class HeuristicComplexityClassifier(ComplexityClassifier):
    """Free classifier: complex if the query spans ≥2 specialist domains OR uses a
    reasoning verb (should/recommend/rebalance/given/compare…)."""

    def is_complex(self, query: str) -> bool:
        q = query.lower()
        kw = KeywordRoutingStrategy
        domains = (kw.PORTFOLIO_KEYWORDS, kw.MARKET_KEYWORDS,
                   kw.SECURITIES_KEYWORDS, kw.RISK_KEYWORDS)
        domains_touched = sum(any(k in q for k in group) for group in domains)
        reasoning = any(v in q for v in _REASONING_VERBS)
        return domains_touched >= 2 or reasoning


class LLMComplexityClassifier(ComplexityClassifier):
    """Gemini flash returns yes/no. Falls back to the heuristic on any error."""

    def __init__(self):
        self._fallback = HeuristicComplexityClassifier()

    def is_complex(self, query: str) -> bool:
        try:
            prompt = (
                "Is this investment query COMPLEX (needs several specialists and "
                "synthesis) or SIMPLE (one lookup)? Reply exactly COMPLEX or SIMPLE.\n"
                f"Query: {query}")
            answer = _text(get_llm().invoke(prompt).content).strip().upper()
            if "COMPLEX" in answer:
                return True
            if "SIMPLE" in answer:
                return False
        except Exception as exc:  # noqa: BLE001
            log.warning("complexity_llm_failed", error=str(exc)[:100])
        return self._fallback.is_complex(query)


def _classifier(name: str) -> ComplexityClassifier:
    return LLMComplexityClassifier() if name == "llm" else HeuristicComplexityClassifier()


class PlannerNode:
    name = "planner"

    def __init__(self, agent_specs: list[AgentSpec], classifier: ComplexityClassifier | None = None):
        self._specs = agent_specs
        self._known = {a.name for a in agent_specs}
        from app.config import settings

        self._classifier = classifier or _classifier(settings.COMPLEXITY_STRATEGY)

    def run(self, state: AgentState) -> dict:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        # Reset per-turn planning/synthesis scratch (persisted threads carry it over).
        reset = {"plan": None, "plan_step": 0, "final_answer": None, "revisions": 0,
                 "route": None, "hops": 0, "visited": []}
        query = _last_human_text(state)

        if not self._classifier.is_complex(query):
            log.info("planner_simple", query=query[:80])
            return reset  # supervisor will route normally

        plan = self._decompose(query)
        if not plan:
            log.info("planner_simple_fallback", query=query[:80])
            return reset
        log.info("planner_plan", steps=[f"{s['agent']}:{s['goal'][:40]}" for s in plan])
        return {**reset, "plan": plan}

    def _decompose(self, query: str) -> list[dict]:
        menu = "\n".join(f"- {a.name}: {a.description}" for a in self._specs)
        prompt = (
            "You are the planner for an investment advisory co-pilot. Decompose the "
            "user's question into an ordered list of sub-goals, each assigned to ONE "
            "specialist. Only use these specialists:\n"
            f"{menu}\n\n"
            f"User question: {query}\n\n"
            "Return ONLY a JSON array, each item {\"agent\": <name>, \"goal\": <short instruction>}. "
            "Order matters (facts before recommendations). Do not include a synthesizer step — "
            "synthesis happens automatically after the plan. Max 4 steps."
        )
        try:
            raw = _text(get_llm().invoke(prompt).content).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            steps = json.loads(raw)
            plan = [{"agent": s["agent"], "goal": str(s["goal"])[:200]}
                    for s in steps
                    if isinstance(s, dict) and s.get("agent") in self._known]
            return plan[:4]
        except Exception as exc:  # noqa: BLE001
            log.warning("planner_decompose_failed", error=str(exc)[:120])
            return []
