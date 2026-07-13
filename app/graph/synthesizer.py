"""SynthesizerNode — composes the final answer from all specialists' work.

Uses the reasoning-tier model (gemini-3.5-flash) with a Chain-of-Thought prompt:
think step by step over the collected evidence, cite it, and — critically — when
tool results CONFLICT (news bullish but momentum bearish; two sources disagree;
a risk mismatch alongside a strong return), surface the conflict explicitly and
explain how it was weighed. Never silently pick one side. (The brief's live demo
requires handling "conflicting information".)

On a reflector revision, a "REVISION REQUIRED: …" system message is present; the
synthesizer must fix exactly that critique.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.base import _last_human_text, _text
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.logging import bind_context, get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are the Synthesizer at XZY Capital. You compose the FINAL answer to the
client from the specialists' findings. You do not call tools or invent data.

Method (reason step by step, then answer):
1. Restate what the client actually asked.
2. Lay out the relevant evidence from the specialists below, grounding every claim in it.
3. If any evidence CONFLICTS (e.g. bullish news vs bearish momentum, a risk/tolerance
   mismatch alongside strong returns, two disagreeing figures), STATE THE CONFLICT PLAINLY
   and explain how you weigh it — never hide it or silently pick one side.
4. Give a clear, well-structured final answer. Every number must come from the evidence.
   This is analysis and observation, not a trading instruction; include a brief disclaimer
   when giving recommendations.

Be concise and precise. Do not fabricate figures that are not in the evidence."""


class SynthesizerNode:
    name = "synthesizer"

    def run(self, state: AgentState) -> dict:
        bind_context(client_id=state.get("client_id"), session_id=state.get("session_id"),
                     agent=self.name)
        question = _last_human_text(state)
        evidence = self._evidence_block(state)
        critique = self._pending_critique(state)

        parts = [f"Client question:\n{question}\n", f"Specialist evidence:\n{evidence}"]
        if state.get("plan"):
            steps = "; ".join(f"{s['agent']}→{s['goal']}" for s in state["plan"])
            parts.append(f"\nThe plan that was executed: {steps}")
        if critique:
            parts.append(f"\nREVISION REQUIRED — your previous answer had this problem, fix it:\n{critique}")

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(parts))]
        answer = _text(get_llm(reasoning=True).invoke(messages).content)
        log.info("synthesized", chars=len(answer), revision=state.get("revisions", 0),
                 had_critique=bool(critique))
        return {"final_answer": answer, "messages": [AIMessage(content=answer, name="synthesizer")]}

    @staticmethod
    def _evidence_block(state: AgentState) -> str:
        """Raw tool outputs (numbers) + each specialist's narrative, tagged by source id."""
        lines = []
        for agent, outputs in (state.get("tool_results") or {}).items():
            for i, out in enumerate(outputs):
                lines.append(f"[{agent}.tool{i}] {_text(out)[:700]}")
        # specialist narratives (their composed messages), excluding the synthesizer's own
        for m in state.get("messages", []):
            if isinstance(m, AIMessage) and m.content and getattr(m, "name", None) != "synthesizer":
                lines.append(f"[{getattr(m, 'name', 'agent')}.summary] {_text(m.content)[:500]}")
        return "\n".join(lines) if lines else "(no specialist evidence was collected)"

    @staticmethod
    def _pending_critique(state: AgentState) -> str | None:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, SystemMessage) and str(m.content).startswith("REVISION REQUIRED:"):
                return str(m.content)[len("REVISION REQUIRED:"):].strip()
        return None
