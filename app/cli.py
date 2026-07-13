"""Tiny CLI for chatting with the graph.

    python -m app.cli --client CLT-002 "How is my NVDA doing and what's the news?"
    python -m app.cli --client CLT-001 --session monday "What did we discuss last time?"

Streams the run so you can watch the supervisor dispatch each specialist.
The client id comes from --client — the CLI stands in for the authenticated
client selection (never parsed out of the question text). Re-using the same
--session continues that conversation; a new session starts fresh but still
recalls prior sessions via long-term memory.
"""

import argparse
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import _text
from app.graph.builder import GraphBuilder

AGENT_ICONS = {"portfolio": "📊", "market_research": "🔎", "securities_analysis": "📈",
               "risk": "🛡️", "supervisor": "🧭", "planner": "🗺️", "synthesizer": "🧩",
               "reflector": "🔬", "input_guard": "🚦", "safe_exit": "🛟",
               "memory_read": "📖", "memory_write": "💾"}


def main() -> None:
    parser = argparse.ArgumentParser(description="XZY Co-Pilot CLI")
    parser.add_argument("--client", required=True, help="client id, e.g. CLT-001")
    parser.add_argument("--session", default=None,
                        help="session name; reuse to continue a conversation "
                             "(default: a fresh timestamped session)")
    parser.add_argument("query", help="your question, in quotes")
    args = parser.parse_args()

    session_id = args.session or f"sess-{datetime.now():%Y%m%d-%H%M%S}"
    # THE data-isolation key: client prefix keeps threads disjoint per client.
    thread_id = f"{args.client}-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    graph = GraphBuilder().with_all().build()

    # The synthesizer (when it runs) produces the single final answer; otherwise we
    # fall back to showing each specialist's own reply, in the order they ran.
    specialist_answers: dict[str, str] = {}
    final_answer = None
    print(f"\n[client {args.client} | thread {thread_id}] {args.query}\n" + "─" * 60)
    for chunk in graph.stream(
        {
            "messages": [HumanMessage(content=args.query)],
            "client_id": args.client,
            "session_id": session_id,
        },
        config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            icon = AGENT_ICONS.get(node, "•")
            if node == "planner":
                plan = (update or {}).get("plan")
                if plan:
                    print(f"{icon} planner: decomposed into {len(plan)} steps")
                    for i, step in enumerate(plan, 1):
                        print(f"     {i}. {step['agent']} — {step['goal']}")
                else:
                    print(f"{icon} planner: simple query, no plan")
                continue
            if node == "supervisor":
                route = (update or {}).get("route")
                print(f"{icon} supervisor: {'done' if route == 'END' else f'→ {route}'}")
                continue
            if node in ("synthesizer", "safe_exit"):
                final_answer = (update or {}).get("final_answer") or final_answer
                print(f"{icon} {node} composed the final answer")
                continue
            if node == "reflector":
                print(f"{icon} reflector: checking answer against evidence")
                continue
            if node in ("input_guard", "memory_read", "memory_write"):
                continue
            print(f"{icon} {node} finished its part")
            for message in (update or {}).get("messages", []):
                if isinstance(message, AIMessage) and message.content and _text(message.content).strip():
                    specialist_answers[node] = _text(message.content)

    print("─" * 60)
    if final_answer:
        print(final_answer)
    elif specialist_answers:
        for node, answer in specialist_answers.items():
            print(f"\n[{AGENT_ICONS.get(node, '•')} {node}]\n{answer}")
    else:
        print("(no answer produced)")


if __name__ == "__main__":
    main()
