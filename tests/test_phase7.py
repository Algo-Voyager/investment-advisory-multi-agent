"""Phase 7 tests — memory store, checkpointer isolation, access control, concurrency.

The concurrency test proves the spec's core claim: two sessions for different
clients running at the same time cannot contaminate each other — ContextVars are
task-local and checkpointer threads are disjoint by construction.
"""

import asyncio
import os
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.guardrails.access_control import (  # noqa: E402
    set_session_client,
    verify_client_access,
)
from app.memory.store import MemoryStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(db_path=str(tmp_path / "memory.sqlite"))


@pytest.fixture(autouse=True)
def _clear_session_client():
    yield
    set_session_client(None)  # never leak a bound client into the next test


# ---------------------------------------------------------------- memory store
class TestMemoryStore:
    def test_decision_roundtrip_newest_first(self, store):
        store.save_decision("CLT-001", "s1", "What do I own?", "VTI, BND…", ["portfolio"])
        store.save_decision("CLT-001", "s2", "Risk exposure?", "Vol 12%…", ["risk"])
        decisions = store.get_recent_decisions("CLT-001")
        assert len(decisions) == 2
        assert decisions[0]["query"] == "Risk exposure?"  # newest first
        assert decisions[0]["agents_used"] == ["risk"]

    def test_memory_is_partitioned_per_client(self, store):
        store.save_decision("CLT-001", "s1", "q1", "a1", ["portfolio"])
        store.save_decision("CLT-002", "s1", "q2", "a2", ["risk"])
        assert len(store.get_recent_decisions("CLT-001")) == 1
        assert store.get_recent_decisions("CLT-002")[0]["query"] == "q2"

    def test_preferences_upsert(self, store):
        store.save_preference("CLT-001", "reporting", "concise")
        store.save_preference("CLT-001", "reporting", "detailed")  # overwrite
        assert store.get_preferences("CLT-001") == {"reporting": "detailed"}


# ---------------------------------------------------------------- memory nodes
class TestMemoryNodes:
    def test_read_node_injects_prior_context_and_resets_scratch(self, store):
        from langchain_core.messages import HumanMessage

        from app.graph.memory_nodes import MemoryReadNode

        store.save_decision("CLT-001", "old-session", "What do I own?",
                            "You hold VTI, BND, and cash.", ["portfolio"])
        update = MemoryReadNode(store=store).run(
            {"client_id": "CLT-001", "session_id": "new-session",
             "messages": [HumanMessage(content="What did we discuss last time?")],
             "hops": 3, "visited": ["portfolio"]})
        assert update["hops"] == 0 and update["visited"] == []  # per-turn reset
        assert "Prior context" in update["messages"][0].content
        assert "VTI" in update["messages"][0].content

    def test_read_node_stays_quiet_without_history(self, store):
        from app.graph.memory_nodes import MemoryReadNode

        update = MemoryReadNode(store=store).run(
            {"client_id": "CLT-004", "session_id": "s", "messages": []})
        assert "messages" not in update  # no fake "prior context"

    def test_write_node_persists_the_interaction(self, store):
        from langchain_core.messages import AIMessage, HumanMessage

        from app.graph.memory_nodes import MemoryWriteNode

        MemoryWriteNode(store=store).run(
            {"client_id": "CLT-002", "session_id": "s9",
             "visited": ["portfolio", "market_research"],
             "messages": [HumanMessage(content="NVDA news?"),
                          AIMessage(content="NVDA is up 4%; headlines: …")]})
        saved = store.get_recent_decisions("CLT-002")[0]
        assert saved["query"] == "NVDA news?"
        assert saved["agents_used"] == ["portfolio", "market_research"]


# ---------------------------------------------------------------- access control
class TestAccessControl:
    def test_cross_client_access_is_denied(self):
        set_session_client("CLT-002")
        with pytest.raises(PermissionError, match="CLT-002 may not access"):
            verify_client_access("CLT-001")

    def test_same_client_passes(self):
        set_session_client("CLT-002")
        verify_client_access("CLT-002")  # no raise

    def test_unbound_context_allows_developer_calls(self):
        set_session_client(None)
        verify_client_access("CLT-001")  # no raise

    def test_repository_chokepoint_enforces_it(self):
        from app.data.repositories import portfolio_repo

        set_session_client("CLT-002")
        with pytest.raises(PermissionError):
            portfolio_repo.get("CLT-001")   # the ACTUAL data path is guarded
        assert portfolio_repo.get("CLT-002").client_id == "CLT-002"

    def test_profile_repository_is_guarded_too(self):
        from app.data.repositories import profile_repo

        set_session_client("CLT-002")
        with pytest.raises(PermissionError):
            profile_repo.get("CLT-009")


# ---------------------------------------------------------------- concurrency (the spec's test)
class TestConcurrentSessionIsolation:
    def test_two_concurrent_client_sessions_do_not_contaminate(self):
        """Each asyncio task binds its own session client; both run at once.
        Task A (CLT-001) must read CLT-001 and be denied CLT-002 — and vice versa."""
        from app.data.repositories import portfolio_repo

        async def session(own: str, other: str) -> tuple[str, bool]:
            set_session_client(own)          # ContextVar → task-local
            await asyncio.sleep(0.01)        # force interleaving
            mine = portfolio_repo.get(own).client_id
            denied = False
            try:
                portfolio_repo.get(other)
            except PermissionError:
                denied = True
            return mine, denied

        async def both():
            return await asyncio.gather(session("CLT-001", "CLT-002"),
                                        session("CLT-002", "CLT-001"))

        (a_mine, a_denied), (b_mine, b_denied) = asyncio.run(both())
        assert a_mine == "CLT-001" and a_denied
        assert b_mine == "CLT-002" and b_denied

    def test_checkpointer_threads_are_disjoint(self, tmp_path):
        """Two thread_ids on one checkpointer hold independent histories."""
        import sqlite3

        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph

        from app.graph.state import AgentState

        def canned(state):  # no LLM — deterministic node
            return {"messages": [AIMessage(content=f"reply-to-{state['client_id']}")]}

        graph = StateGraph(AgentState)
        graph.add_node("canned", canned)
        graph.add_edge(START, "canned")
        graph.add_edge("canned", END)
        saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.sqlite"),
                                            check_same_thread=False))
        app = graph.compile(checkpointer=saver)

        cfg_a = {"configurable": {"thread_id": "CLT-001-s1"}}
        cfg_b = {"configurable": {"thread_id": "CLT-002-s1"}}
        app.invoke({"messages": [], "client_id": "CLT-001"}, cfg_a)
        app.invoke({"messages": [], "client_id": "CLT-001"}, cfg_a)  # turn 2, same thread
        app.invoke({"messages": [], "client_id": "CLT-002"}, cfg_b)

        hist_a = app.get_state(cfg_a).values["messages"]
        hist_b = app.get_state(cfg_b).values["messages"]
        assert len(hist_a) == 2 and len(hist_b) == 1  # histories grew independently
        assert all("CLT-001" in m.content for m in hist_a)
        assert all("CLT-002" in m.content for m in hist_b)
