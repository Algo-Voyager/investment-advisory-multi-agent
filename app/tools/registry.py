"""ToolRegistry — Registry pattern (singleton).

Tools declare themselves where they live:

    @tool_registry.register(agent="portfolio")
    def get_holdings(client_id: str) -> list[dict]: ...

and are discovered centrally:

    tool_registry.tools_for("portfolio")   # → LangChain tools for that agent
    tool_registry.table()                  # → the who-has-what overview

This kills the "hand-maintained tool list" problem: an agent's toolbox is
whatever registered for it — adding a tool is one decorator, no other edits.
"""

from langchain_core.tools import tool as make_langchain_tool

from app.logging import get_logger

log = get_logger(__name__)


class ToolRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:  # classic singleton — one registry per process
            cls._instance = super().__new__(cls)
            cls._instance._by_agent = {}
            cls._instance._lc_cache = {}
        return cls._instance

    def register(self, agent: str, name: str | None = None):
        """Decorator: attach a plain function to an agent's toolbox."""

        def decorator(fn):
            fn._tool_name = name or fn.__name__
            self._by_agent.setdefault(agent, [])
            # idempotent on module re-import (notebooks re-run imports freely)
            if all(f._tool_name != fn._tool_name for f in self._by_agent[agent]):
                self._by_agent[agent].append(fn)
                self._lc_cache.pop(agent, None)
            return fn  # unchanged — still directly callable in tests/notebooks

        return decorator

    def tools_for(self, agent: str) -> list:
        """The agent's toolbox as LangChain tools (wrapped once, cached)."""
        if agent not in self._lc_cache:
            self._lc_cache[agent] = [make_langchain_tool(f) for f in self._by_agent.get(agent, [])]
        return self._lc_cache[agent]

    def table(self) -> list[dict]:
        """[{'agent', 'tool', 'description'}] — for docs, notebooks, and the demo."""
        rows = []
        for agent in sorted(self._by_agent):
            for fn in self._by_agent[agent]:
                doc = (fn.__doc__ or "").strip().splitlines()[0]
                rows.append({"agent": agent, "tool": fn._tool_name, "description": doc})
        return rows


tool_registry = ToolRegistry()
