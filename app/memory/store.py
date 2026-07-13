"""MemoryStore — Facade pattern for LONG-TERM memory.

Domain vocabulary (save_decision / get_recent_decisions / save_preference) over a
tiny SQLite backend — so past advice survives process restarts (a CLI run is one
process; "what did we discuss last time?" needs cross-process recall, which rules
out LangGraph's InMemoryStore for dev). Production swaps the backend for a
LangGraph persistent Store / Postgres — the facade methods stay identical.

Every row is keyed by client_id: long-term memory is per-client by construction,
which is half of the data-privacy story (the other half is access_control).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.logging import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    agents_used TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_client ON decisions(client_id, id DESC);
CREATE TABLE IF NOT EXISTS preferences (
    client_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (client_id, key)
);
"""


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        path = Path(db_path) if db_path else Path(settings.SQLITE_CHECKPOINT_PATH).with_name("memory.sqlite")
        if not path.is_absolute():
            path = _ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    # ---- decisions (historical advice) ------------------------------------
    def save_decision(self, client_id: str, session_id: str, query: str,
                      answer: str, agents_used: list[str],
                      timestamp: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO decisions (client_id, session_id, ts, query, answer, agents_used) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (client_id, session_id, timestamp or datetime.now().isoformat(timespec="seconds"),
             query[:500], answer[:1000], json.dumps(agents_used)))
        self._conn.commit()
        log.info("decision_saved", client_id=client_id, session_id=session_id)

    def get_recent_decisions(self, client_id: str, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT session_id, ts, query, answer, agents_used FROM decisions "
            "WHERE client_id = ? ORDER BY id DESC LIMIT ?", (client_id, limit)).fetchall()
        return [{"session_id": r[0], "ts": r[1], "query": r[2], "answer": r[3],
                 "agents_used": json.loads(r[4])} for r in rows]

    # ---- preferences --------------------------------------------------------
    def save_preference(self, client_id: str, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO preferences (client_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(client_id, key) DO UPDATE SET value = excluded.value",
            (client_id, key, value))
        self._conn.commit()

    def get_preferences(self, client_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT key, value FROM preferences WHERE client_id = ?", (client_id,)).fetchall()
        return dict(rows)


_default_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store
