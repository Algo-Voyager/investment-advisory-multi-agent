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

Full graph shape (Phase 9):
    START → [memory_read] → [input_guard] → [planner] → supervisor
            supervisor ⇄ (portfolio | market_research | securities_analysis | risk)
            supervisor ─(done/plan-exhausted)→ synthesizer → reflector
            reflector ─(revise)→ synthesizer   (≤ MAX_REVISIONS)
            reflector ─(block)→ safe_exit
            input_guard ─(block)→ safe_exit
            (reflector pass | safe_exit) → [memory_write] → END

Each stage is optional (flags), so tests can build a lean graph; `with_all()`
turns everything on. Callers only ever see `.build()`.
"""

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.market_research import MarketResearchAgent
from app.agents.portfolio import PortfolioAgent
from app.agents.risk import RiskAssessmentAgent
from app.agents.securities_analysis import SecuritiesAnalysisAgent
from app.agents.supervisor import SupervisorAgent
from app.graph.memory_nodes import MemoryReadNode, MemoryWriteNode
from app.graph.planner import PlannerNode
from app.graph.reflector import InputGuardNode, ReflectorNode, SafeExitNode
from app.graph.router import AgentSpec, RoutingStrategy
from app.graph.state import AgentState
from app.graph.synthesizer import SynthesizerNode


class GraphBuilder:
    def __init__(self):
        self._agents: list[BaseAgent] = []
        self._use_supervisor = False
        self._use_memory = False
        self._use_planning = False
        self._use_guardrails = False
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

    def with_planning(self) -> "GraphBuilder":
        """Planner (complexity classifier + decomposition) + Synthesizer (Phase 8)."""
        self._use_planning = True
        return self

    def with_guardrails(self) -> "GraphBuilder":
        """Input guardrails + reflection/hallucination loop + safe exit (Phase 9)."""
        self._use_guardrails = True
        return self

    def with_all(self) -> "GraphBuilder":
        """Everything built so far — what the CLI/UI/notebooks use."""
        return (self.with_supervisor()
                .with_portfolio_agent()
                .with_market_research_agent()
                .with_securities_analysis_agent()
                .with_risk_agent()
                .with_memory()
                .with_planning()
                .with_guardrails())

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

        # ---- Supervisor loop (Phase 2), composed with the optional stages ----
        specs = [AgentSpec(a.name, a.description) for a in self._agents]

        # terminal = where a finished/blocked answer lands last
        terminal = "memory_write" if self._use_memory else END
        # supervisor hands off to the synthesizer when planning is on, else to terminal
        supervisor_end = "synthesizer" if self._use_planning else terminal
        supervisor = SupervisorAgent(specs, strategy=self._routing_strategy,
                                     end_node=supervisor_end)
        graph.add_node(supervisor.name, supervisor.run,
                       destinations=tuple(a.name for a in self._agents) + (supervisor_end,))
        for agent in self._agents:
            graph.add_node(agent.name, agent.run)
            graph.add_edge(agent.name, supervisor.name)  # every agent reports back

        # synthesizer + reflector (Phase 8/9)
        if self._use_planning:
            graph.add_node("synthesizer", SynthesizerNode().run)
            synth_next = "reflector" if self._use_guardrails else terminal
            if self._use_guardrails:
                graph.add_node("reflector", ReflectorNode(next_node=terminal).run,
                               destinations=("synthesizer", "safe_exit", terminal))
                graph.add_edge("synthesizer", "reflector")
            else:
                graph.add_edge("synthesizer", synth_next)

        # safe exit (Phase 9) — reachable from input_guard and reflector
        if self._use_guardrails:
            graph.add_node("safe_exit", SafeExitNode(next_node=terminal).run,
                           destinations=(terminal,))

        # ---- entry chain: START → [memory_read] → [input_guard] → [planner] → supervisor
        first_after_entry = supervisor.name
        if self._use_planning:
            planner = PlannerNode(specs)
            graph.add_node("planner", planner.run)
            graph.add_edge("planner", supervisor.name)
            first_after_entry = "planner"
        if self._use_guardrails:
            guard = InputGuardNode(next_node=first_after_entry)
            graph.add_node("input_guard", guard.run,
                           destinations=(first_after_entry, "safe_exit"))
            first_after_entry = "input_guard"

        if self._use_memory:
            graph.add_node("memory_read", MemoryReadNode().run)
            graph.add_node("memory_write", MemoryWriteNode().run)
            graph.add_edge(START, "memory_read")
            graph.add_edge("memory_read", first_after_entry)
            graph.add_edge("memory_write", END)
            from app.memory.checkpointer import get_checkpointer

            return graph.compile(checkpointer=get_checkpointer())

        graph.add_edge(START, first_after_entry)
        return graph.compile()
