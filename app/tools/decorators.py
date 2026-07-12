"""Cross-cutting resilience decorators — Decorator pattern.

Three concerns every network-touching tool needs, applied without polluting the
tool's own logic:

    @cached(ttl_seconds=60)      # don't re-fetch what we just fetched
    @retry(max_attempts=3)       # survive transient failures with backoff
    @rate_limited(30)            # never hammer a free-tier API

Stack order matters — outermost first:
    cached → retry → rate_limited → fn
A cache hit must cost nothing: no retry machinery, no rate-limit token, no network.

The cache is file-backed under `.cache/` so it survives process restarts —
crucial for notebooks (fresh kernel) and the live demo (pre-warm it).
"""

import functools
import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path

import requests

from app.config import settings
from app.errors.exceptions import RateLimitError
from app.logging import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"  # anchored to project root


def cached(ttl_seconds: int):
    """File-backed cache. Key = function name + JSON-serialized arguments."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key_raw = json.dumps({"fn": f"{fn.__module__}.{fn.__qualname__}",
                                  "args": args, "kwargs": kwargs},
                                 sort_keys=True, default=str)
            key = hashlib.sha1(key_raw.encode()).hexdigest()
            path = _CACHE_DIR / f"{fn.__name__}-{key[:16]}.json"

            if path.exists():
                try:
                    entry = json.loads(path.read_text())
                    age = time.time() - entry["at"]
                    if age < ttl_seconds:
                        log.info("cache_hit", fn=fn.__name__, age_s=round(age, 1))
                        return entry["value"]
                except (json.JSONDecodeError, KeyError):
                    path.unlink(missing_ok=True)  # corrupt entry — drop it

            value = fn(*args, **kwargs)
            try:
                _CACHE_DIR.mkdir(exist_ok=True)
                path.write_text(json.dumps({"at": time.time(), "value": value}, default=str))
            except TypeError:
                log.warning("cache_skip_unserializable", fn=fn.__name__)
            return value

        return wrapper

    return decorator


def retry(max_attempts: int = 3, backoff: str = "exponential",
          on: tuple = (RateLimitError, requests.HTTPError, requests.ConnectionError)):
    """Retry transient failures. Exponential backoff: 1s, 2s, 4s… (capped at 8s)."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except on as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(2 ** (attempt - 1), 8) if backoff == "exponential" else 1
                    log.warning("retrying", fn=fn.__name__, attempt=attempt,
                                delay_s=delay, error=str(exc)[:120])
                    time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator


def rate_limited(calls_per_minute: int):
    """Token bucket per function: if the window is full, wait instead of getting banned."""

    def decorator(fn):
        calls: deque = deque()
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.monotonic()
                while calls and now - calls[0] > 60:
                    calls.popleft()  # forget calls older than the window
                if len(calls) >= calls_per_minute:
                    wait = 60 - (now - calls[0]) + 0.05
                    log.info("rate_limit_wait", fn=fn.__name__, seconds=round(wait, 1))
                    time.sleep(wait)
                calls.append(time.monotonic())
            return fn(*args, **kwargs)

        return wrapper

    return decorator
