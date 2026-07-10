"""GraphBuilder — Builder pattern.

The graph is assembled progressively across phases:

    Phase 1:  GraphBuilder().with_portfolio_agent().build()      # single node
    Phase 2+: .with_supervisor().with_market_research_agent()... # routing appears

Callers (CLI, notebooks, later the Streamlit UI) only ever see `.build()` —
they never learn how nodes and edges are wired. That keeps the UI a Facade
client and lets each phase grow the graph without touching its consumers.
"""

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.portfolio import PortfolioAgent
from app.graph.state import AgentState


class GraphBuilder:
    def __init__(self):
        self._agents: list[BaseAgent] = []

    def with_portfolio_agent(self) -> "GraphBuilder":
        self._agents.append(PortfolioAgent())
        return self

    def build(self):
        if not self._agents:
            raise ValueError("GraphBuilder: add at least one agent before build()")
        graph = StateGraph(AgentState)
        # Phase 1: no routing — every query goes to the (single) portfolio agent.
        agent = self._agents[0]
        graph.add_node(agent.name, agent.run)
        graph.add_edge(START, agent.name)
        graph.add_edge(agent.name, END)
        return graph.compile()
