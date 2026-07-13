"""A single persistent background event loop for the whole Streamlit process.

Streamlit reruns the script synchronously on every interaction; a naive
`asyncio.run(...)` per rerun creates a FRESH event loop each time. That breaks
`AsyncSqliteSaver` (Phase 14's checkpointer for `astream_events`), which binds
its internal lock and loop reference to whatever loop was running at
construction — reusing it from a different loop on the next rerun raises
loop-affinity errors.

Fix: start one loop in a dedicated background thread, once, and dispatch every
async call (checkpointer construction AND each turn's `astream_events` drive)
onto that SAME loop via `run_coroutine_threadsafe`. This is the standard
pattern for embedding asyncio work inside a sync framework.
"""

import asyncio
import threading


class BackgroundLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        """Run `coro` on the background loop from any (sync) calling thread."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()
