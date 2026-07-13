"""Phase 8 tests — complexity classifier, planner decomposition, supervisor plan-walk."""

import os
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.graph.planner import HeuristicComplexityClassifier, PlannerNode  # noqa: E402
from app.graph.router import AgentSpec  # noqa: E402

SPECS = [
    AgentSpec("portfolio", "holdings, allocations, performance"),
    AgentSpec("market_research", "news, sectors, market conditions"),
    AgentSpec("securities_analysis", "technical indicators"),
    AgentSpec("risk", "volatility, beta, VaR, tolerance"),
]


class TestComplexityHeuristic:
    clf = HeuristicComplexityClassifier()

    @pytest.mark.parametrize("query", [
        "Should I rebalance given the recent Fed announcement and my risk tolerance?",
        "How is my NVDA doing and what's the news and is it within my risk tolerance?",
        "Compare my portfolio risk versus the market and recommend changes",
    ])
    def test_complex_queries_flagged(self, query):
        assert self.clf.is_complex(query) is True

    @pytest.mark.parametrize("query", [
        "What stocks do I own?",
        "What is my VTI position worth?",
        "What is my cash balance?",
    ])
    def test_simple_queries_not_flagged(self, query):
        assert self.clf.is_complex(query) is False


class TestPlannerNode:
    def test_simple_query_produces_no_plan_and_resets_scratch(self):
        from langchain_core.messages import HumanMessage

        node = PlannerNode(SPECS, classifier=HeuristicComplexityClassifier())
        out = node.run({"messages": [HumanMessage(content="What stocks do I own?")],
                        "plan": [{"agent": "risk", "goal": "stale"}], "hops": 2})
        assert out["plan"] is None
        assert out["hops"] == 0 and out["visited"] == []  # per-turn reset

    def test_decompose_filters_unknown_agents_and_synth_steps(self):
        # Inject a fake LLM by stubbing _decompose's model call via a fake classifier
        # that forces "complex", then monkeypatch get_llm through the module.
        import app.graph.planner as planner_mod

        class FakeResp:
            content = ('[{"agent":"market_research","goal":"summarize Fed"},'
                       '{"agent":"portfolio","goal":"get allocation"},'
                       '{"agent":"synthesizer","goal":"should be dropped"},'
                       '{"agent":"nonsense","goal":"should be dropped"},'
                       '{"agent":"risk","goal":"risk metrics"}]')

        class FakeLLM:
            def invoke(self, _):
                return FakeResp()

        orig = planner_mod.get_llm
        planner_mod.get_llm = lambda *a, **k: FakeLLM()
        try:
            node = PlannerNode(SPECS, classifier=_AlwaysComplex())
            from langchain_core.messages import HumanMessage

            out = node.run({"messages": [HumanMessage(content="rebalance given fed and risk")]})
        finally:
            planner_mod.get_llm = orig

        agents = [s["agent"] for s in out["plan"]]
        assert agents == ["market_research", "portfolio", "risk"]  # synth + nonsense dropped


class _AlwaysComplex(HeuristicComplexityClassifier):
    def is_complex(self, query: str) -> bool:
        return True


class TestSupervisorPlanWalk:
    def _sup(self):
        from app.agents.supervisor import SupervisorAgent

        return SupervisorAgent(SPECS, end_node="synthesizer")

    def test_walks_plan_steps_in_order(self):
        sup = self._sup()
        plan = [{"agent": "market_research", "goal": "fed"},
                {"agent": "portfolio", "goal": "alloc"},
                {"agent": "risk", "goal": "risk"}]
        base = {"plan": plan, "messages": []}

        c0 = sup.run({**base, "plan_step": 0, "hops": 0})
        assert c0.goto == "market_research" and c0.update["plan_step"] == 1
        c1 = sup.run({**base, "plan_step": 1, "hops": 1})
        assert c1.goto == "portfolio" and c1.update["plan_step"] == 2
        c2 = sup.run({**base, "plan_step": 2, "hops": 2})
        assert c2.goto == "risk" and c2.update["plan_step"] == 3

    def test_exhausted_plan_hands_off_to_synthesizer(self):
        sup = self._sup()
        plan = [{"agent": "portfolio", "goal": "x"}]
        cmd = sup.run({"plan": plan, "plan_step": 1, "hops": 1, "messages": []})
        assert cmd.goto == "synthesizer"

    def test_plan_step_injects_focused_subgoal(self):
        from langchain_core.messages import SystemMessage

        sup = self._sup()
        plan = [{"agent": "portfolio", "goal": "get current allocation by sector"}]
        cmd = sup.run({"plan": plan, "plan_step": 0, "hops": 0, "messages": []})
        note = cmd.update["messages"][0]
        assert isinstance(note, SystemMessage)
        assert "get current allocation by sector" in note.content


class TestSynthesizerEvidence:
    def test_builds_evidence_block_from_tool_results_and_summaries(self):
        from langchain_core.messages import AIMessage

        from app.graph.synthesizer import SynthesizerNode

        state = {"tool_results": {"risk": ['{"beta": 1.4}']},
                 "messages": [AIMessage(content="Beta is 1.4", name="risk")]}
        block = SynthesizerNode._evidence_block(state)
        assert "risk.tool0" in block and "1.4" in block
        assert "risk.summary" in block

    def test_detects_pending_revision_critique(self):
        from langchain_core.messages import SystemMessage

        from app.graph.synthesizer import SynthesizerNode

        state = {"messages": [SystemMessage(content="REVISION REQUIRED: number 72 is invented")]}
        assert "72 is invented" in SynthesizerNode._pending_critique(state)
