"""Streamlit demo UI — Facade over `GraphBuilder.build()`.

The UI never talks to agents, tools, or memory directly — it only calls
`graph.astream_events(...)` and `graph.invoke(Command(resume=...))`. Everything
about HOW a question gets answered (routing, planning, guardrails, RAG) lives
in the graph; this file is purely presentation (MVC-lite: this is the View,
`AgentState` is the Model, the graph is the Controller).

Run: `make ui`  (streamlit run ui/streamlit_app.py)
"""

import os
import queue
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


def _escape_dollars(text: str) -> str:
    """st.markdown() renders `$...$` as LaTeX math — every dollar-figure in a
    financial answer ("$209.07... $210.96") gets misread as a math-mode
    delimiter pair and garbled into italic Unicode (confirmed via a real
    browser check: "SMA 50: $209.07. The current price ($210.96)" rendered as
    mangled math). Escape literal `$` so it displays as plain currency."""
    return text.replace("$", "\\$")


def render_markdown(text: str) -> None:
    st.markdown(_escape_dollars(text))

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


def _extract_update(out) -> dict:
    """A node's `on_chain_end` output is either a plain state-update dict, or a
    `Command(update=...)` when Phase 11's global error wrapping is on (true for
    every node under `with_all()`) — normalize both to a plain dict so callers
    don't have to care which shape a given node happens to return."""
    if out is None:
        return {}
    if hasattr(out, "update"):  # langgraph.types.Command
        return out.update or {}
    return out if isinstance(out, dict) else {}


async def stream_turn(graph, run_input, config, show_reasoning: bool,
                      event_queue: "queue.Queue") -> dict:
    """Drive astream_events(version='v2') on the BACKGROUND loop/thread, pushing
    live progress onto `event_queue` as it happens.

    Streamlit's `st.*` calls require the script's own execution context (a
    thread-local), and this coroutine runs on a single loop/thread SHARED
    across every user session (see async_bridge.py) — so it must never call
    `st.*` itself. It only puts plain-data events on the queue; the caller
    (back on the Streamlit script's own thread) polls the queue and renders,
    so the UI updates AS the turn progresses instead of only at the end.
    """
    final_answer = None
    plan_shown = False
    pending_tool_calls: dict[str, dict] = {}  # run_id -> {"name","input"}, paired at on_tool_end

    async for event in graph.astream_events(run_input, config, version="v2"):
        name = event.get("name", "")
        kind = event.get("event", "")

        if kind == "on_chain_start" and name in NODE_LABELS:
            event_queue.put({"kind": "status", "label": NODE_LABELS[name]})
            if name == "synthesizer":
                # A reflector revision loop re-runs the synthesizer from scratch —
                # without this, a second draft's tokens would just concatenate
                # onto the end of the first draft's already-streamed text.
                event_queue.put({"kind": "answer_reset"})

        if kind == "on_chain_end" and name == "supervisor" and show_reasoning:
            update = _extract_update((event.get("data") or {}).get("output"))
            route, reason = update.get("route"), update.get("route_reason")
            if route and route != "END" and reason:
                event_queue.put({"kind": "route_reason", "agent": route, "text": reason})

        if kind == "on_chain_end" and name == "planner" and show_reasoning:
            update = _extract_update((event.get("data") or {}).get("output"))
            plan = update.get("plan")
            if plan and not plan_shown:
                plan_shown = True
                event_queue.put({"kind": "plan", "plan": plan})

        if kind == "on_tool_start":
            pending_tool_calls[event.get("run_id")] = {
                "name": name, "input": (event.get("data") or {}).get("input")}

        if kind == "on_tool_end":
            call = pending_tool_calls.pop(event.get("run_id"), {"name": name, "input": None})
            event_queue.put({"kind": "tool", "name": call["name"], "input": call["input"],
                             "output": str((event.get("data") or {}).get("output"))[:800]})

        if kind == "on_chain_end" and name == "reflector" and show_reasoning:
            update = _extract_update((event.get("data") or {}).get("output"))
            events = update.get("guardrail_events")
            if events:
                event_queue.put({"kind": "guardrails", "events": events[-6:]})

        if kind == "on_chat_model_stream" and (event.get("metadata") or {}).get(
                "langgraph_node") == "synthesizer":
            # Gemini chunk content is a list of blocks, e.g. [{"type":"text","text":"..."}] —
            # _text() (app.agents.base) already knows how to flatten that shape.
            chunk = (event.get("data") or {}).get("chunk")
            delta = _text(getattr(chunk, "content", "") or "") if chunk is not None else ""
            if delta:
                event_queue.put({"kind": "token", "text": delta})

        if kind == "on_chain_end" and name in ("synthesizer", "safe_exit"):
            update = _extract_update((event.get("data") or {}).get("output"))
            if update.get("final_answer"):
                final_answer = update["final_answer"]

    # aget_state, not get_state: we're running ON the checkpointer's own
    # background loop here — AsyncSqliteSaver explicitly forbids its sync
    # methods being called from that loop's thread (only from a DIFFERENT
    # thread, which is how resume_session()'s sync get_state() call is fine).
    state = await graph.aget_state(config)
    if state.next:  # paused — an interrupt is pending
        for task in state.tasks:
            if task.interrupts:
                return {"interrupt": task.interrupts[0].value}
    if not final_answer:
        for m in reversed(state.values.get("messages", [])):
            if isinstance(m, AIMessage) and m.content:
                final_answer = _text(m.content)
                break
    return {"final_answer": final_answer or "(no answer produced)",
           "tool_results": state.values.get("tool_results", {})}


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
    """Kick the turn off on the background loop WITHOUT blocking (bg.submit, not
    bg.run), then poll the queue stream_turn is writing to and render live: a
    `st.status` box grows with each step (routing reasoning, plan, tool calls,
    guardrail checks) while the answer streams token-by-token into its own
    placeholder below it — instead of the UI going blank until the whole turn
    finishes and dumping everything at once.
    """
    show_reasoning = st.session_state["show_reasoning"]
    with st.chat_message("assistant"):
        event_queue: "queue.Queue" = queue.Queue()
        bg = get_background_loop()
        future = bg.submit(stream_turn(graph, run_input, config, show_reasoning, event_queue))

        status_box = st.status("Working…", expanded=show_reasoning)
        answer_placeholder = st.empty()
        answer_text = ""

        while True:
            try:
                item = event_queue.get(timeout=0.05)
            except queue.Empty:
                if future.done():
                    break
                continue

            kind = item["kind"]
            if kind == "status":
                status_box.update(label=item["label"])
            elif kind == "answer_reset":
                answer_text = ""
                answer_placeholder.empty()
            elif kind == "token":
                answer_text += item["text"]
                answer_placeholder.markdown(_escape_dollars(answer_text) + "▌")
            elif kind == "route_reason":
                with status_box:
                    st.write(f"🧭 Routing to **{item['agent']}** — {item['text']}")
            elif kind == "plan":
                with status_box:
                    st.write("**🗺️ Plan:**")
                    for i, s in enumerate(item["plan"], 1):
                        st.write(f"{i}. **{s['agent']}** — {s['goal']}")
            elif kind == "tool":
                with status_box, st.expander(f"🔧 {item['name']}", expanded=False):
                    if item.get("input") is not None:
                        st.caption("Called with:")
                        st.code(str(item["input"]), language="json")
                    st.caption("Result:")
                    st.code(item["output"], language="json")
            elif kind == "guardrails":
                with status_box:
                    st.write("**🧠 Guardrail checks:**")
                    for ev in item["events"]:
                        st.write(f"- `{ev['guard']}` → **{ev['action']}** {ev.get('reason', '')}")

        result = future.result()  # propagates any exception the turn hit, same as before
        status_box.update(label="✅ Done", state="complete", expanded=show_reasoning)

        if "interrupt" in result:
            answer_placeholder.empty()
            st.session_state["pending_interrupt"] = result["interrupt"]
            st.warning(f"⚠️ Clarification needed: {result['interrupt']['question']}")
        else:
            answer_placeholder.markdown(_escape_dollars(result["final_answer"]))
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
