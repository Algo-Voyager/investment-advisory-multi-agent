"""Phase 10 tests — clarifier node, interrupt/resume mechanics, config toggle."""

import os
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

ROOT = Path(__file__).resolve().parents[1]
_env = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
HAS_REAL_KEY = "GOOGLE_API_KEY=" in _env and "your-gemini-key" not in _env


class TestClarifierConfigToggle:
    def test_disabled_skips_straight_through_without_any_llm_call(self, monkeypatch):
        from app.config import settings
        from app.graph.clarifier import ClarifierNode

        monkeypatch.setattr(settings, "ENABLE_CLARIFICATION", False)
        node = ClarifierNode(next_node="supervisor")
        cmd = node.run({"client_id": "CLT-005", "session_id": "s",
                        "messages": [], "clarification_answer": None})
        assert cmd.goto == "supervisor"
        assert cmd.update is None  # nothing was touched — proves _detect never ran

    def test_already_answered_this_turn_skips_re_asking(self, monkeypatch):
        from app.config import settings
        from app.graph.clarifier import ClarifierNode

        monkeypatch.setattr(settings, "ENABLE_CLARIFICATION", True)
        node = ClarifierNode(next_node="supervisor")
        cmd = node.run({"client_id": "CLT-005", "session_id": "s", "messages": [],
                        "clarification_answer": "mega-cap only"})
        assert cmd.goto == "supervisor"  # no interrupt() call, no re-ask


class TestClarifierDetectFailsOpen:
    def test_llm_error_defaults_to_not_ambiguous(self, monkeypatch):
        import app.graph.clarifier as clarifier_mod

        class Boom:
            def invoke(self, _):
                raise RuntimeError("quota exceeded")

        monkeypatch.setattr(clarifier_mod, "get_llm", lambda *a, **k: Boom())
        result = clarifier_mod.ClarifierNode._detect("CLT-005", "how's my tech doing?")
        assert result["needs_clarification"] is False  # fail open — never blocks a turn


class TestInterruptResumeMechanics:
    """Exercises the raw LangGraph interrupt()/Command(resume=...) contract that
    ClarifierNode relies on — offline, no LLM, deterministic."""

    def _build(self, tmp_path):
        import sqlite3

        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt

        from app.graph.state import AgentState

        def clarifying_node(state):
            answer = interrupt({"question": "Which tech — mega-cap or high-beta?",
                                "options": ["mega-cap", "high-beta"]})
            return {"messages": [AIMessage(content=f"Scoped to: {answer}")]}

        graph = StateGraph(AgentState)
        graph.add_node("ask", clarifying_node)
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False))
        return graph.compile(checkpointer=saver)

    def test_stream_pauses_with_interrupt_payload(self, tmp_path):
        app = self._build(tmp_path)
        cfg = {"configurable": {"thread_id": "t1"}}
        chunks = list(app.stream({"messages": [], "client_id": "CLT-005"}, cfg,
                                 stream_mode="updates"))
        assert any("__interrupt__" in c for c in chunks)
        payload = next(c["__interrupt__"][0].value for c in chunks if "__interrupt__" in c)
        assert payload["question"].startswith("Which tech")
        assert payload["options"] == ["mega-cap", "high-beta"]
        # state is paused, not finished
        assert app.get_state(cfg).next == ("ask",)

    def test_resume_continues_exactly_where_it_paused(self, tmp_path):
        from langgraph.types import Command

        app = self._build(tmp_path)
        cfg = {"configurable": {"thread_id": "t2"}}
        list(app.stream({"messages": [], "client_id": "CLT-005"}, cfg, stream_mode="updates"))

        result = app.invoke(Command(resume="mega-cap"), cfg)
        assert "Scoped to: mega-cap" in result["messages"][-1].content
        assert app.get_state(cfg).next == ()  # finished


# ---------------------------------------------------------------- live (Gemini)
@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestClarifierLive:
    def test_ambiguous_tech_question_is_flagged_for_clt005(self):
        from app.graph.clarifier import ClarifierNode

        result = ClarifierNode._detect("CLT-005", "How's my tech doing?")
        assert result["needs_clarification"] is True
        assert result["question"]

    def test_unambiguous_question_is_not_flagged(self):
        from app.graph.clarifier import ClarifierNode

        result = ClarifierNode._detect("CLT-002", "What is my NVDA position worth?")
        assert result["needs_clarification"] is False


@pytest.mark.skipif(not HAS_REAL_KEY, reason="needs a real GOOGLE_API_KEY in .env")
class TestGraphAcceptance:
    def test_ambiguous_query_interrupts_then_resume_yields_scoped_answer(self):
        from langchain_core.messages import HumanMessage

        from app.graph.builder import GraphBuilder

        graph = GraphBuilder().with_all().build()
        cfg = {"configurable": {"thread_id": "CLT-005-test-p10"}}
        state = {"messages": [HumanMessage(content="How's my tech doing?")],
                 "client_id": "CLT-005", "session_id": "test-p10"}

        chunks = list(graph.stream(state, cfg, stream_mode="updates"))
        interrupted = [c for c in chunks if "__interrupt__" in c]
        assert interrupted, "ambiguous tech question should trigger a clarification"

        from langgraph.types import Command

        resumed = graph.invoke(Command(resume="mega-cap only"), cfg)
        answer = next(m.content for m in reversed(resumed["messages"])
                      if getattr(m, "content", None) and m.type == "ai")
        assert answer  # a scoped answer was produced after resuming
