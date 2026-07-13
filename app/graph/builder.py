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

Full graph shape (Phase 10):
    START → [memory_read] → [input_guard] → [planner] → [clarifier] → supervisor
            supervisor ⇄ (portfolio | market_research | securities_analysis | risk)
            supervisor ─(done/plan-exhausted)→ synthesizer → reflector
            reflector ─(revise)→ synthesizer   (≤ MAX_REVISIONS)
            reflector ─(block)→ safe_exit
            input_guard ─(block)→ safe_exit
            (reflector pass | safe_exit) → [memory_write] → END

`clarifier` may PAUSE the graph mid-run via `interrupt()` (Phase 10) — this needs a
checkpointer to persist the pause, so enabling clarification implies compiling
with one even if `with_memory()` wasn't requested.

GLOBAL ERROR HANDLING (Phase 11): whenever guardrails are on (so `safe_exit`
exists), every node is wrapped via `app.errors.node_wrapper` so an uncaught
exception anywhere becomes `Command(goto="safe_exit", ...)` instead of a raw
traceback reaching `graph.stream()`. LangGraph does not let a returned `Command`
override a STATIC `add_edge` (both would fire and collide) — so wrapped nodes
route purely via `Command`, with no static edge alongside them.

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
from app.errors.node_wrapper import wrap_command_node, wrap_dict_node
from app.graph.clarifier import ClarifierNode
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
        self._use_clarification = False
        self._routing_strategy: RoutingStrategy | None = None
        self._checkpointer = None  # override — see with_memory()

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

    def with_memory(self, checkpointer=None) -> "GraphBuilder":
        """Persistent memory: checkpointer (short-term, per thread_id) +
        MemoryRead/MemoryWrite nodes (long-term, per client). Requires a
        thread_id in the invoke config.

        `checkpointer`: override the default sync `SqliteSaver` (used by the
        CLI/notebooks/tests). The Streamlit UI passes an `AsyncSqliteSaver`
        instead — `astream_events` requires an async-capable checkpointer;
        the default sync one raises NotImplementedError under it.
        """
        self._use_memory = True
        if checkpointer is not None:
            self._checkpointer = checkpointer
        return self

    def with_planning(self) -> "GraphBuilder":
        """Planner (complexity classifier + decomposition) + Synthesizer (Phase 8)."""
        self._use_planning = True
        return self

    def with_guardrails(self) -> "GraphBuilder":
        """Input guardrails + reflection/hallucination loop + safe exit (Phase 9)."""
        self._use_guardrails = True
        return self

    def with_clarification(self) -> "GraphBuilder":
        """Human-in-the-loop clarification via interrupt() (Phase 10). Runs after
        the planner. Implies a checkpointer even without with_memory()."""
        self._use_clarification = True
        return self

    def with_all(self, checkpointer=None) -> "GraphBuilder":
        """Everything built so far — what the CLI/UI/notebooks use.

        `checkpointer`: forwarded to with_memory() — pass an AsyncSqliteSaver
        here when building a graph that will be driven via astream_events()
        (e.g. the Streamlit UI).
        """
        return (self.with_supervisor()
                .with_portfolio_agent()
                .with_market_research_agent()
                .with_securities_analysis_agent()
                .with_risk_agent()
                .with_memory(checkpointer=checkpointer)
                .with_planning()
                .with_guardrails()
                .with_clarification())

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
        safe = self._use_guardrails  # global error wrapping is only meaningful once safe_exit exists

        # terminal = where a finished/blocked answer lands last
        terminal = "memory_write" if self._use_memory else END
        # supervisor hands off to the synthesizer when planning is on, else to terminal
        supervisor_end = "synthesizer" if self._use_planning else terminal
        supervisor = SupervisorAgent(specs, strategy=self._routing_strategy,
                                     end_node=supervisor_end)
        sup_destinations = tuple(a.name for a in self._agents) + (supervisor_end,)
        graph.add_node(supervisor.name,
                       wrap_command_node(supervisor.run, supervisor.name) if safe else supervisor.run,
                       destinations=sup_destinations + (("safe_exit",) if safe else ()))
        for agent in self._agents:
            if safe:
                graph.add_node(agent.name, wrap_dict_node(agent.run, agent.name, supervisor.name),
                               destinations=(supervisor.name, "safe_exit"))
            else:
                graph.add_node(agent.name, agent.run)
                graph.add_edge(agent.name, supervisor.name)  # every agent reports back

        # synthesizer + reflector (Phase 8/9)
        if self._use_planning:
            synth = SynthesizerNode()
            synth_next = "reflector" if self._use_guardrails else terminal
            if safe:
                graph.add_node("synthesizer", wrap_dict_node(synth.run, "synthesizer", synth_next),
                               destinations=(synth_next, "safe_exit"))
            else:
                graph.add_node("synthesizer", synth.run)
                graph.add_edge("synthesizer", synth_next)
            if self._use_guardrails:
                reflector = ReflectorNode(next_node=terminal)
                graph.add_node("reflector",
                               wrap_command_node(reflector.run, "reflector") if safe else reflector.run,
                               destinations=("synthesizer", "safe_exit", terminal))

        # safe exit (Phase 9) — reachable from input_guard, reflector, and (Phase 11) any wrapped node
        if self._use_guardrails:
            graph.add_node("safe_exit", SafeExitNode(next_node=terminal).run,
                           destinations=(terminal,))

        # ---- entry chain: START → [memory_read] → [input_guard] → [planner] →
        #                   [clarifier] → supervisor
        first_after_entry = supervisor.name
        if self._use_clarification:
            clarifier = ClarifierNode(next_node=first_after_entry)
            if safe:
                graph.add_node("clarifier", wrap_command_node(clarifier.run, "clarifier"),
                               destinations=(first_after_entry, "safe_exit"))
            else:
                graph.add_node("clarifier", clarifier.run, destinations=(first_after_entry,))
            first_after_entry = "clarifier"
        if self._use_planning:
            planner = PlannerNode(specs)
            if safe:
                graph.add_node("planner", wrap_dict_node(planner.run, "planner", first_after_entry),
                               destinations=(first_after_entry, "safe_exit"))
            else:
                graph.add_node("planner", planner.run)
                graph.add_edge("planner", first_after_entry)
            first_after_entry = "planner"
        if self._use_guardrails:
            guard = InputGuardNode(next_node=first_after_entry)
            graph.add_node("input_guard",
                           wrap_command_node(guard.run, "input_guard") if safe else guard.run,
                           destinations=(first_after_entry, "safe_exit"))
            first_after_entry = "input_guard"

        # A checkpointer is required whenever the clarifier can interrupt() —
        # not only when with_memory() was explicitly requested.
        needs_checkpointer = self._use_memory or self._use_clarification

        if self._use_memory:
            mem_read = MemoryReadNode()
            # memory_write handles its own errors internally (see memory_nodes.py) —
            # it's the terminal step; a save failure must never erase a good answer.
            graph.add_node("memory_write", MemoryWriteNode().run)
            if safe:
                graph.add_node("memory_read", wrap_dict_node(mem_read.run, "memory_read",
                                                              first_after_entry),
                               destinations=(first_after_entry, "safe_exit"))
            else:
                graph.add_node("memory_read", mem_read.run)
                graph.add_edge("memory_read", first_after_entry)
            graph.add_edge(START, "memory_read")
            graph.add_edge("memory_write", END)
        else:
            graph.add_edge(START, first_after_entry)

        if needs_checkpointer:
            checkpointer = self._checkpointer
            if checkpointer is None:
                from app.memory.checkpointer import get_checkpointer

                checkpointer = get_checkpointer()
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()
