"""SupervisorAgent — the multi-agent orchestrator (LangGraph supervisor pattern).

The supervisor is the ONLY component that decides who runs next; agents
communicate solely through `AgentState` and never call each other. Each turn the
supervisor consults its RoutingStrategy and returns `Command(goto=<agent>)` —
or `Command(goto=END)` when the conversation already answers the user.

A hop counter guards against ping-pong loops: after MAX_HOPS agent turns the
supervisor ends the run regardless, so a confused router can never spin forever.
"""

from langgraph.graph import END
from langgraph.types import Command

from app.graph.router import END_ROUTE, AgentSpec, LLMRoutingStrategy, RoutingStrategy
from app.graph.state import AgentState
from app.logging import bind_context, get_logger

log = get_logger(__name__)

MAX_HOPS = 4  # safety valve — no realistic query needs more agent turns than this


class SupervisorAgent:
    name = "supervisor"

    def __init__(self, agent_specs: list[AgentSpec], strategy: RoutingStrategy | None = None):
        self._specs = agent_specs
        self._strategy = strategy or LLMRoutingStrategy()

    def run(self, state: AgentState) -> Command:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        hops = state.get("hops", 0)
        if hops >= MAX_HOPS:
            log.warning("supervisor_max_hops", hops=hops)
            return Command(goto=END, update={"route": END_ROUTE})

        visited = state.get("visited", [])
        decision = self._strategy.route(state, self._specs)
        if decision in visited:
            # Each specialist gets ONE turn per run (multi-step plans arrive with the
            # Phase 8 planner). A router that re-picks a finished agent is looping.
            log.info("supervisor_dedupe", repeated=decision)
            decision = END_ROUTE
        if decision == END_ROUTE:
            log.info("supervisor_end", hops=hops)
            return Command(goto=END, update={"route": END_ROUTE})

        log.info("supervisor_dispatch", next_agent=decision, hop=hops + 1)
        return Command(goto=decision,
                       update={"route": decision, "hops": hops + 1,
                               "visited": visited + [decision]})
