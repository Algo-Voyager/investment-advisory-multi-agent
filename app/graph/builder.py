"""GraphBuilder — Builder pattern.

THREAD-ID SCHEME (Phase 7 — the data-isolation key):
    thread_id = f"{client_id}-{session_id}"        e.g. "CLT-001-sess-20260713-1"
The checkpointer keys ALL short-term memory by this string, so CLT-001's
conversations live under "CLT-001-*" and can never collide with CLT-002's.
`client_id` is always the raw CLT-XXX string from the AUTHENTICATED selection
(CLI flag / UI dropdown) — never constructed from chat input. Long-term memory
(MemoryStore) is keyed by client_id the same way, and the access-control
interceptor (app/guardrails/access_control.py) enforces the boundary on every
data access as belt-and-braces.

Graph shape with memory (Phase 7):
    START → memory_read → supervisor ⇄ (portfolio | market_research |
            securities_analysis | risk) → supervisor → memory_write → END

Callers (CLI, notebooks, the Streamlit UI) only ever see `.build()` — they never
learn how nodes and edges are wired.
"""

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.market_research import MarketResearchAgent
from app.agents.portfolio import PortfolioAgent
from app.agents.risk import RiskAssessmentAgent
from app.agents.securities_analysis import SecuritiesAnalysisAgent
from app.agents.supervisor import SupervisorAgent
from app.graph.memory_nodes import MemoryReadNode, MemoryWriteNode
from app.graph.router import AgentSpec, RoutingStrategy
from app.graph.state import AgentState


class GraphBuilder:
    def __init__(self):
        self._agents: list[BaseAgent] = []
        self._use_supervisor = False
        self._use_memory = False
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

    def with_memory(self) -> "GraphBuilder":
        """Persistent memory: checkpointer (short-term, per thread_id) +
        MemoryRead/MemoryWrite nodes (long-term, per client). Requires a
        thread_id in the invoke config."""
        self._use_memory = True
        return self

    def with_all(self) -> "GraphBuilder":
        """Everything built so far — what the CLI/UI/notebooks use."""
        return (self.with_supervisor()
                .with_portfolio_agent()
                .with_market_research_agent()
                .with_securities_analysis_agent()
                .with_risk_agent()
                .with_memory())

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

        # Supervisor loop (Phase 2), optionally wrapped in memory nodes (Phase 7).
        specs = [AgentSpec(a.name, a.description) for a in self._agents]
        end_target = "memory_write" if self._use_memory else END
        supervisor = SupervisorAgent(specs, strategy=self._routing_strategy,
                                     end_node=end_target)
        # destinations= lets LangGraph render the Command-based edges in Mermaid.
        graph.add_node(supervisor.name, supervisor.run,
                       destinations=tuple(a.name for a in self._agents) + (end_target,))
        for agent in self._agents:
            graph.add_node(agent.name, agent.run)
            graph.add_edge(agent.name, supervisor.name)  # every agent reports back

        if self._use_memory:
            graph.add_node("memory_read", MemoryReadNode().run)
            graph.add_node("memory_write", MemoryWriteNode().run)
            graph.add_edge(START, "memory_read")
            graph.add_edge("memory_read", supervisor.name)
            graph.add_edge("memory_write", END)
            from app.memory.checkpointer import get_checkpointer

            return graph.compile(checkpointer=get_checkpointer())

        graph.add_edge(START, supervisor.name)
        return graph.compile()
