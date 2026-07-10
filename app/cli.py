"""Tiny CLI for chatting with the graph.

    python -m app.cli --client CLT-001 "what do I own?"

The client id comes from this flag — the CLI stands in for the authenticated
client selection (never parsed out of the question text).
"""

import argparse

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import _text
from app.graph.builder import GraphBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="XZY Co-Pilot CLI")
    parser.add_argument("--client", required=True, help="client id, e.g. CLT-001")
    parser.add_argument("query", help="your question, in quotes")
    args = parser.parse_args()

    graph = GraphBuilder().with_portfolio_agent().build()
    print(f"[client {args.client}] → portfolio agent\n")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=args.query)],
            "client_id": args.client,
            "session_id": "cli",
        }
    )
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            print(_text(message.content))
            break


if __name__ == "__main__":
    main()
