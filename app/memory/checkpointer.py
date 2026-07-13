"""Checkpointer factory — Factory pattern.

Short-term (per-session) memory: LangGraph checkpoints every graph step keyed by
`thread_id`, which is what makes conversations resumable and `interrupt()`
(Phase 10) possible. Dev backend is SQLite; production swaps this factory's
return value for `AsyncPostgresSaver` — callers never change.
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    path = Path(settings.SQLITE_CHECKPOINT_PATH)
    if not path.is_absolute():
        path = _ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the graph may hop threads (streaming, asyncio bridge);
    # SqliteSaver serializes access internally.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
