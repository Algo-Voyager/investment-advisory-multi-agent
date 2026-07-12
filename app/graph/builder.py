"""GraphBuilder — Builder pattern.

The graph is assembled progressively across phases:

    Phase 1:  GraphBuilder().with_portfolio_agent().build()          # single node
    Phase 2:  GraphBuilder().with_supervisor()
                            .with_portfolio_agent()
                            .with_market_research_agent().build()    # supervisor loop

Phase 2 shape:  START → supervisor → (portfolio | market_research) → supervisor → … → END
The supervisor routes via Command(goto=...); every agent returns to the
supervisor, which decides the next hop or ends the run.

Callers (CLI, notebooks, later the Streamlit UI) only ever see `.build()` —
they never learn how nodes and edges are wired.
"""

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.market_research import MarketResearchAgent
from app.agents.portfolio import PortfolioAgent
from app.agents.risk import RiskAssessmentAgent
from app.agents.securities_analysis import SecuritiesAnalysisAgent
from app.agents.supervisor import SupervisorAgent
from app.graph.router import AgentSpec, RoutingStrategy
from app.graph.state import AgentState


class GraphBuilder:
    def __init__(self):
        self._agents: list[BaseAgent] = []
        self._use_supervisor = False
        self._routing_strategy: RoutingStrategy | None = None

    def with_supervisor(self, strategy: RoutingStrategy | None = None) -> "GraphBuilder":
        self._use_supervisor = True
        self._routing_strategy = strategy
        return self

    def with_portfolio_agent(self) -> "GraphBuilder":
        self._agents.append(PortfolioAgent())
        return self

    def with_market_research_agent(self) -> "GraphBuilder":
        self._agents.append(MarketResearchAgent())
        return self

    def with_securities_analysis_agent(self) -> "GraphBuilder":
        self._agents.append(SecuritiesAnalysisAgent())
        return self

    def with_risk_agent(self) -> "GraphBuilder":
        self._agents.append(RiskAssessmentAgent())
        return self

    def with_all(self) -> "GraphBuilder":
        """Everything built so far — what the CLI/UI/notebooks use."""
        return (self.with_supervisor()
                .with_portfolio_agent()
                .with_market_research_agent()
                .with_securities_analysis_agent()
                .with_risk_agent())

    def build(self):
        if not self._agents:
            raise ValueError("GraphBuilder: add at least one agent before build()")
        graph = StateGraph(AgentState)

        if not self._use_supervisor:
            # Phase 1 shape — single agent, no routing.
            agent = self._agents[0]
            graph.add_node(agent.name, agent.run)
            graph.add_edge(START, agent.name)
            graph.add_edge(agent.name, END)
            return graph.compile()

        # Phase 2 shape — supervisor loop.
        specs = [AgentSpec(a.name, a.description) for a in self._agents]
        supervisor = SupervisorAgent(specs, strategy=self._routing_strategy)
        # destinations= lets LangGraph render the Command-based edges in Mermaid.
        graph.add_node(supervisor.name, supervisor.run,
                       destinations=tuple(a.name for a in self._agents) + (END,))
        graph.add_edge(START, supervisor.name)
        for agent in self._agents:
            graph.add_node(agent.name, agent.run)
            graph.add_edge(agent.name, supervisor.name)  # every agent reports back
        return graph.compile()
