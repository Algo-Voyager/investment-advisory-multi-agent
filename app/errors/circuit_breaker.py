"""CircuitBreaker — per-adapter, three-state (closed/open/half-open).

Phase 3's @retry survives a BLIP (one bad request); it does NOT protect against
an adapter being down for the whole demo — that would mean 3 slow, doomed
retries on every single call. The circuit breaker adds a fast-fail memory:

    closed    → calls go through normally; N consecutive failures → OPEN
    open      → calls fail INSTANTLY (no network attempt) until the cooldown elapses
    half_open → cooldown elapsed; the next call is a trial. Success → closed.
                Failure → open again, cooldown restarts.

One breaker per adapter (keyed by name), shared across calls via the registry
below — so a dead Alpha Vantage key doesn't get re-discovered as dead on every
single tool call once the circuit has already tripped.
"""

import threading
import time
from enum import Enum

from app.errors.exceptions import ToolError
from app.logging import get_logger

log = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ToolError):
    """Raised instantly when a breaker is open — no network attempt was made."""


class CircuitBreaker:
    def __init__(self, name: str, threshold: int = 3, cooldown: int = 60):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if time.monotonic() - self._opened_at >= self.cooldown:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def call(self, fn, *args, **kwargs):
        with self._lock:
            state = self._state_locked()
            if state == CircuitState.OPEN:
                log.warning("circuit_fail_fast", adapter=self.name, failures=self._failures)
                raise CircuitOpenError(
                    f"[{self.name}] circuit open after {self._failures} consecutive "
                    f"failures — failing fast for {self._remaining_cooldown():.0f}s more")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure(state)
            raise
        else:
            self._record_success(state)
            return result

    def _record_failure(self, state_before: CircuitState) -> None:
        with self._lock:
            self._failures += 1
            if state_before == CircuitState.HALF_OPEN or self._failures >= self.threshold:
                self._opened_at = time.monotonic()
                log.warning("circuit_opened", adapter=self.name, failures=self._failures)

    def _record_success(self, state_before: CircuitState) -> None:
        with self._lock:
            if state_before == CircuitState.HALF_OPEN:
                log.info("circuit_closed", adapter=self.name)
            self._failures = 0
            self._opened_at = None

    def _remaining_cooldown(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.cooldown - (time.monotonic() - self._opened_at))

    def reset(self) -> None:
        """Test/ops hook: force back to closed."""
        with self._lock:
            self._failures = 0
            self._opened_at = None


class CircuitBreakerRegistry:
    """One breaker per adapter name, created on first use (Registry-style singleton)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._breakers: dict[str, CircuitBreaker] = {}
        return cls._instance

    def get(self, name: str, threshold: int | None = None, cooldown: int | None = None) -> CircuitBreaker:
        from app.config import settings

        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name, threshold or settings.CIRCUIT_BREAKER_THRESHOLD,
                cooldown or settings.CIRCUIT_BREAKER_COOLDOWN)
        return self._breakers[name]

    def all_states(self) -> dict[str, str]:
        return {name: cb.state.value for name, cb in self._breakers.items()}


circuit_breakers = CircuitBreakerRegistry()
