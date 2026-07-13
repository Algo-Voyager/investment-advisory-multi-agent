"""Streamlit demo UI — Facade over `GraphBuilder.build()`.

The UI never talks to agents, tools, or memory directly — it only calls
`graph.astream_events(...)` and `graph.invoke(Command(resume=...))`. Everything
about HOW a question gets answered (routing, planning, guardrails, RAG) lives
in the graph; this file is purely presentation (MVC-lite: this is the View,
`AgentState` is the Model, the graph is the Controller).

Run: `make ui`  (streamlit run ui/streamlit_app.py)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

# Streamlit Cloud's secrets manager populates `st.secrets`, NOT the process
# environment — but `app.config.Settings` (pydantic-settings) reads from
# `os.environ`/`.env`. Bridge them before importing `app.config` (its module-level
# `settings = Settings()` runs at import time and needs GOOGLE_API_KEY present).
# Local dev has no `.streamlit/secrets.toml`, so this is a no-op there (.env covers it).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

from app.agents.base import _text  # noqa: E402
from app.config import settings  # noqa: E402
from app.data.repositories import portfolio_repo, profile_repo  # noqa: E402
from app.graph.builder import GraphBuilder  # noqa: E402
from app.memory.store import get_memory_store  # noqa: E402
from ui.async_bridge import BackgroundLoop  # noqa: E402
from ui.components.citations_panel import render_citations_panel  # noqa: E402
from ui.components.portfolio_panel import render_portfolio_panel  # noqa: E402

st.set_page_config(page_title="XZY Capital — Investment Advisory Co-Pilot", page_icon="💼",
                   layout="wide")


def render_markdown(text: str) -> None:
    """st.markdown() renders `$...$` as LaTeX math — every dollar-figure in a
    financial answer ("$209.07... $210.96") gets misread as a math-mode
    delimiter pair and garbled into italic Unicode (confirmed via a real
    browser check: "SMA 50: $209.07. The current price ($210.96)" rendered as
    mangled math). Escape literal `$` so it displays as plain currency."""
    st.markdown(text.replace("$", "\\$"))

NODE_LABELS = {
    "memory_read": "📖 Recalling prior context…",
    "input_guard": "🚦 Screening the question…",
    "planner": "🗺️ Planning…",
    "clarifier": "❓ Checking for ambiguity…",
    "supervisor": "🧭 Deciding who should answer…",
    "portfolio": "📊 Portfolio Agent consulting…",
    "market_research": "🔎 Market Research consulting…",
    "securities_analysis": "📈 Securities Analysis consulting…",
    "risk": "🛡️ Risk Assessment consulting…",
    "synthesizer": "🧩 Synthesizing the final answer…",
    "reflector": "🧠 Reflecting on the answer…",
    "safe_exit": "🛟 Wrapping up safely…",
    "memory_write": "💾 Saving this exchange…",
}


@st.cache_resource(show_spinner=False)
def get_background_loop() -> BackgroundLoop:
    return BackgroundLoop()


@st.cache_resource(show_spinner=False)
def get_graph():
    """Build the graph ONCE for the process, with an async-capable checkpointer
    (AsyncSqliteSaver) constructed on — and bound to — the persistent background
    loop, since `astream_events` needs an async checkpointer and every turn's
    streaming call is dispatched onto that SAME loop (see async_bridge.py)."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.config import settings as _settings

    bg = get_background_loop()

    async def _build_checkpointer():
        path = Path(_settings.SQLITE_CHECKPOINT_PATH)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        return saver

    checkpointer = bg.run(_build_checkpointer())
    return GraphBuilder().with_all(checkpointer=checkpointer).build()


def client_options() -> list[dict]:
    options = []
    for client_id in portfolio_repo.client_ids():
        profile = profile_repo.get(client_id)
        label = f"{client_id} — {profile.name} ({profile.risk_tolerance})" if profile else client_id
        options.append({"client_id": client_id, "label": label})
    return options


def init_session_state() -> None:
    st.session_state.setdefault("client_id", client_options()[0]["client_id"])
    st.session_state.setdefault("session_id", f"ui-{datetime.now():%Y%m%d-%H%M%S}")
    st.session_state.setdefault("messages", [])       # [{role, content}] for display
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("last_tool_results", {})
    st.session_state.setdefault("show_reasoning", False)


def new_session() -> None:
    st.session_state["session_id"] = f"ui-{datetime.now():%Y%m%d-%H%M%S}"
    st.session_state["messages"] = []
    st.session_state["pending_interrupt"] = None
    st.session_state["last_tool_results"] = {}


def switch_client(client_id: str) -> None:
    st.session_state["client_id"] = client_id
    new_session()


def load_recent_sessions(client_id: str) -> list[str]:
    decisions = get_memory_store().get_recent_decisions(client_id, limit=20)
    seen, ordered = set(), []
    for d in decisions:
        if d["session_id"] not in seen:
            seen.add(d["session_id"])
            ordered.append(d["session_id"])
    return ordered


def resume_session(client_id: str, session_id: str) -> None:
    """Switch to a prior thread and replay its checkpointed history for display."""
    st.session_state["client_id"] = client_id
    st.session_state["session_id"] = session_id
    st.session_state["pending_interrupt"] = None
    graph = get_graph()
    config = {"configurable": {"thread_id": f"{client_id}-{session_id}"}}
    snapshot = graph.get_state(config)
    messages = []
    for m in snapshot.values.get("messages", []):
        if isinstance(m, HumanMessage) and m.content:
            messages.append({"role": "user", "content": _text(m.content)})
        elif isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "safe_exit":
            messages.append({"role": "assistant", "content": _text(m.content)})
    st.session_state["messages"] = messages
    st.session_state["last_tool_results"] = snapshot.values.get("tool_results", {})


def render_sidebar() -> None:
    st.sidebar.title("💼 XZY Capital")
    st.sidebar.caption("Investment Advisory Co-Pilot")

    options = client_options()
    labels = [o["label"] for o in options]
    ids = [o["client_id"] for o in options]
    current_idx = ids.index(st.session_state["client_id"])
    chosen_label = st.sidebar.selectbox("Client", labels, index=current_idx)
    chosen_id = ids[labels.index(chosen_label)]
    if chosen_id != st.session_state["client_id"]:
        switch_client(chosen_id)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Session")
    st.sidebar.caption(f"thread: `{st.session_state['client_id']}-{st.session_state['session_id']}`")
    if st.sidebar.button("🆕 New session", use_container_width=True):
        new_session()
        st.rerun()

    recent = load_recent_sessions(st.session_state["client_id"])
    recent = [s for s in recent if s != st.session_state["session_id"]]
    if recent:
        picked = st.sidebar.selectbox("Recent sessions", ["—"] + recent)
        if picked != "—":
            resume_session(st.session_state["client_id"], picked)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    st.sidebar.caption(
        f"Worker tier: `{settings.MODEL_NAME}`  \n"
        f"Reasoning tier: `{settings.REASONING_MODEL_NAME}`  \n"
        f"Both Gemini — no other provider is used in this project."
    )
    st.session_state["show_reasoning"] = st.sidebar.toggle(
        "🧠 Show reasoning", value=st.session_state["show_reasoning"],
        help="Show the plan, routing decisions, and guardrail checks inline.")

    st.sidebar.divider()
    # `with st.sidebar:` is REQUIRED here, not just calling these from within
    # render_sidebar() — the panel components use plain `st.expander`/`st.metric`
    # internally, which render into the MAIN area unless the sidebar container
    # is the active context when they're called (found via a real browser check:
    # without this, the panels silently rendered in the main content area instead
    # of the sidebar).
    with st.sidebar:
        render_portfolio_panel(st.session_state["client_id"])
        render_citations_panel(st.session_state["last_tool_results"])


async def stream_turn(graph, run_input, config, show_reasoning: bool) -> dict:
    """Drive astream_events(version='v2') on the BACKGROUND loop/thread.

    Streamlit's `st.*` calls require the script's own execution context (a
    thread-local), and this coroutine runs on a single loop/thread SHARED
    across every user session (see async_bridge.py) — so it must never call
    `st.*` itself. It only collects plain-data "steps"; the caller (back on
    the Streamlit script's own thread) renders them.
    """
    final_answer = None
    plan_shown = False
    steps: list[dict] = []

    async for event in graph.astream_events(run_input, config, version="v2"):
        name = event.get("name", "")
        kind = event.get("event", "")

        if kind == "on_chain_start" and name in NODE_LABELS:
            steps.append({"kind": "status", "label": NODE_LABELS[name]})

        if kind == "on_chain_end" and name == "planner" and show_reasoning:
            plan = ((event.get("data") or {}).get("output") or {})
            plan = plan.get("plan") if isinstance(plan, dict) else None
            if plan and not plan_shown:
                plan_shown = True
                steps.append({"kind": "plan", "plan": plan})

        if kind == "on_tool_end":
            steps.append({"kind": "tool", "name": event.get("name", "tool"),
                         "output": str((event.get("data") or {}).get("output"))[:800]})

        if kind == "on_chain_end" and name == "reflector" and show_reasoning:
            out = (event.get("data") or {}).get("output")
            events = getattr(out, "get", lambda *_: None)("guardrail_events") if out else None
            if events:
                steps.append({"kind": "guardrails", "events": events[-6:]})

        if kind == "on_chain_end" and name in ("synthesizer", "safe_exit"):
            out = (event.get("data") or {}).get("output") or {}
            if isinstance(out, dict) and out.get("final_answer"):
                final_answer = out["final_answer"]

    # aget_state, not get_state: we're running ON the checkpointer's own
    # background loop here — AsyncSqliteSaver explicitly forbids its sync
    # methods being called from that loop's thread (only from a DIFFERENT
    # thread, which is how resume_session()'s sync get_state() call is fine).
    state = await graph.aget_state(config)
    if state.next:  # paused — an interrupt is pending
        for task in state.tasks:
            if task.interrupts:
                return {"interrupt": task.interrupts[0].value, "steps": steps}
    if not final_answer:
        for m in reversed(state.values.get("messages", [])):
            if isinstance(m, AIMessage) and m.content:
                final_answer = _text(m.content)
                break
    return {"final_answer": final_answer or "(no answer produced)",
           "tool_results": state.values.get("tool_results", {}), "steps": steps}


def run_turn_sync(graph, run_input, config) -> dict:
    # Dispatch onto the SAME persistent background loop the checkpointer is bound
    # to (see get_graph()) — a fresh asyncio.run() here would create a different
    # loop each rerun and break AsyncSqliteSaver's loop affinity. stream_turn does
    # no st.* rendering itself (see its docstring) — this call blocks the current
    # (Streamlit script) thread until the whole turn finishes, then we render.
    bg = get_background_loop()
    return bg.run(stream_turn(graph, run_input, config, st.session_state["show_reasoning"]))


def _render_steps(steps: list[dict]) -> None:
    """Render the collected steps — called on the Streamlit script's own thread."""
    for step in steps:
        if step["kind"] == "status":
            st.caption(step["label"])
        elif step["kind"] == "plan":
            with st.expander("🗺️ Plan", expanded=True):
                for i, s in enumerate(step["plan"], 1):
                    st.write(f"{i}. **{s['agent']}** — {s['goal']}")
        elif step["kind"] == "tool":
            with st.expander(f"🔧 {step['name']}", expanded=False):
                st.code(step["output"], language="json")
        elif step["kind"] == "guardrails":
            with st.expander("🧠 Guardrail checks", expanded=False):
                for ev in step["events"]:
                    st.write(f"- `{ev['guard']}` → **{ev['action']}** {ev.get('reason', '')}")


def handle_query(query: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": query})
    graph = get_graph()
    client_id, session_id = st.session_state["client_id"], st.session_state["session_id"]
    thread_id = f"{client_id}-{session_id}"
    # run_name: LangSmith's root-run display name — without it every trace shows up
    # as generic "LangGraph"; this makes the Threads/Traces list thread-identifiable.
    config = {"configurable": {"thread_id": thread_id}, "run_name": thread_id}
    run_input = {"messages": [HumanMessage(content=query)], "client_id": client_id,
                "session_id": session_id}
    _drive_graph(graph, run_input, config)


def handle_resume(answer: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": f"*(clarification)* {answer}"})
    graph = get_graph()
    client_id, session_id = st.session_state["client_id"], st.session_state["session_id"]
    thread_id = f"{client_id}-{session_id}"
    config = {"configurable": {"thread_id": thread_id}, "run_name": thread_id}
    st.session_state["pending_interrupt"] = None
    _drive_graph(graph, Command(resume=answer), config)


def _drive_graph(graph, run_input, config) -> None:
    with st.chat_message("assistant"):
        with st.spinner("Working…"):
            result = run_turn_sync(graph, run_input, config)
        _render_steps(result.get("steps", []))
        if "interrupt" in result:
            st.session_state["pending_interrupt"] = result["interrupt"]
            st.warning(f"⚠️ Clarification needed: {result['interrupt']['question']}")
        else:
            render_markdown(result["final_answer"])
            st.session_state["messages"].append({"role": "assistant", "content": result["final_answer"]})
            st.session_state["last_tool_results"] = result.get("tool_results", {})


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("Investment Advisory Co-Pilot")
    st.caption(f"Chatting as **{st.session_state['client_id']}**")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            render_markdown(msg["content"])

    if st.session_state["pending_interrupt"]:
        interrupt = st.session_state["pending_interrupt"]
        with st.chat_message("assistant"):
            st.warning(f"⚠️ {interrupt['question']}")
            options = interrupt.get("options") or []
            if options:
                cols = st.columns(len(options))
                for col, option in zip(cols, options):
                    if col.button(option, use_container_width=True):
                        handle_resume(option)
                        st.rerun()
            free_text = st.chat_input("Or type your own answer…")
            if free_text:
                handle_resume(free_text)
                st.rerun()
        return  # don't show the normal chat_input while a clarification is pending

    query = st.chat_input("Ask about your portfolio, markets, or risk…")
    if query:
        with st.chat_message("user"):
            render_markdown(query)
        handle_query(query)
        st.rerun()


if __name__ == "__main__":
    main()
