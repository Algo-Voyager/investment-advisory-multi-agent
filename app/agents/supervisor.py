"""SupervisorAgent — the multi-agent orchestrator (LangGraph supervisor pattern).

The supervisor is the ONLY component that decides who runs next; agents
communicate solely through `AgentState` and never call each other.

Two modes:
- PLAN mode (Phase 8): if `state.plan` exists, the supervisor INTERPRETS it —
  dispatching plan[plan_step] each turn and advancing the index. When the plan is
  exhausted it hands off to the synthesizer.
- SIMPLE mode: no plan → it consults its RoutingStrategy each turn (one turn per
  specialist), then hands off.

Three independent guards prevent runaway loops: plan bounds (plan mode), the
`visited` dedup (simple mode), and MAX_HOPS (both).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from app.graph.router import (
    END_ROUTE,
    AgentSpec,
    KeywordRoutingStrategy,
    LLMRoutingStrategy,
    RoutingStrategy,
)
from app.graph.state import AgentState
from app.logging import bind_context, get_logger

log = get_logger(__name__)

MAX_HOPS = 4  # safety valve — no realistic query needs more agent turns than this


class SupervisorAgent:
    name = "supervisor"

    def __init__(self, agent_specs: list[AgentSpec], strategy: RoutingStrategy | None = None,
                 end_node: str = END):
        self._specs = agent_specs
        self._strategy = strategy or LLMRoutingStrategy()
        self._end = end_node  # END, or "memory_write" when the memory layer is on (Phase 7)

    def run(self, state: AgentState) -> Command:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        hops = state.get("hops", 0)

        # -- PLAN mode: interpret the planner's step list ----------------------
        plan = state.get("plan")
        if plan:
            step = state.get("plan_step", 0)
            if step >= len(plan) or hops >= len(plan) + 1:
                log.info("supervisor_plan_done", steps=len(plan))
                return Command(goto=self._end, update={"route": END_ROUTE})
            agent = plan[step]["agent"]
            goal = plan[step]["goal"]
            log.info("supervisor_plan_step", step=step + 1, of=len(plan), agent=agent,
                     goal=goal[:60])
            # Feed the step's sub-goal to the agent as a focused instruction.
            return Command(goto=agent, update={
                "route": agent, "route_reason": goal, "hops": hops + 1, "plan_step": step + 1,
                "visited": state.get("visited", []) + [agent],
                "messages": [SystemMessage(
                    content=f"PLAN STEP {step + 1}/{len(plan)} — focus only on: {goal}")],
            })

        # -- SIMPLE mode: strategy routing, one turn per specialist ------------
        if hops >= MAX_HOPS:
            log.warning("supervisor_max_hops", hops=hops)
            return Command(goto=self._end, update={"route": END_ROUTE})

        visited = state.get("visited", [])
        decision = self._strategy.route(state, self._specs)
        reason = getattr(self._strategy, "last_reason", None)
        if decision in visited:
            # Each specialist gets ONE turn per run (multi-step plans arrive with the
            # Phase 8 planner). A router that re-picks a finished agent is looping.
            log.info("supervisor_dedupe", repeated=decision)
            decision = END_ROUTE
        if decision == END_ROUTE and not visited:
            # Never end a turn before ANY agent has spoken. This fires when the LLM
            # router thinks the question is already answered — usually because
            # memory_read injected prior context that reads like an answer. But no
            # agent has produced a reply THIS turn, so fall back to a deterministic
            # pick (defaults to portfolio) and let it compose one.
            fallback_strategy = KeywordRoutingStrategy()
            fallback = fallback_strategy.route(state, self._specs)
            decision = fallback if fallback != END_ROUTE else self._specs[0].name
            reason = fallback_strategy.last_reason or "router ended before any agent ran"
            log.info("supervisor_forced_dispatch", agent=decision,
                     reason="router ended before any agent ran")
        if decision == END_ROUTE:
            log.info("supervisor_end", hops=hops)
            return Command(goto=self._end, update={"route": END_ROUTE})

        log.info("supervisor_dispatch", next_agent=decision, hop=hops + 1)
        return Command(goto=decision,
                       update={"route": decision, "route_reason": reason, "hops": hops + 1,
                               "visited": visited + [decision]})
