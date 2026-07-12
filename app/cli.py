"""Tiny CLI for chatting with the graph.

    python -m app.cli --client CLT-002 "How is my NVDA doing and what's the news?"

Streams the run so you can watch the supervisor dispatch each specialist.
The client id comes from --client — the CLI stands in for the authenticated
client selection (never parsed out of the question text).
"""

import argparse

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import _text
from app.graph.builder import GraphBuilder

AGENT_ICONS = {"portfolio": "📊", "market_research": "🔎", "supervisor": "🧭"}


def main() -> None:
    parser = argparse.ArgumentParser(description="XZY Co-Pilot CLI")
    parser.add_argument("--client", required=True, help="client id, e.g. CLT-001")
    parser.add_argument("query", help="your question, in quotes")
    args = parser.parse_args()

    graph = (GraphBuilder()
             .with_supervisor()
             .with_portfolio_agent()
             .with_market_research_agent()
             .build())

    # Until the Phase 8 synthesizer exists, the honest output is each specialist's
    # own answer, in the order they ran — not just whichever message came last.
    answers: dict[str, str] = {}
    print(f"\n[client {args.client}] {args.query}\n" + "─" * 60)
    for chunk in graph.stream(
        {
            "messages": [HumanMessage(content=args.query)],
            "client_id": args.client,
            "session_id": "cli",
        },
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            icon = AGENT_ICONS.get(node, "•")
            if node == "supervisor":
                route = (update or {}).get("route")
                label = "done" if route == "END" else f"routing to → {route}"
                print(f"{icon} supervisor: {label}")
                continue
            print(f"{icon} {node} agent finished its part")
            for message in (update or {}).get("messages", []):
                if isinstance(message, AIMessage) and message.content and _text(message.content).strip():
                    answers[node] = _text(message.content)  # keep each agent's final say

    print("─" * 60)
    if not answers:
        print("(no answer produced)")
    for node, answer in answers.items():
        print(f"\n[{AGENT_ICONS.get(node, '•')} {node}]\n{answer}")


if __name__ == "__main__":
    main()
