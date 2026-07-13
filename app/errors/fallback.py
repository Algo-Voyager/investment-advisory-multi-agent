"""Fallback — primary → secondary under failure.

A small, general-purpose helper distinct from the adapter CHAIN (Phase 3):
`MarketDataChain` tries N adapters for the SAME kind of call. `Fallback` is for
one-off primary/secondary pairs elsewhere in the codebase — e.g. the Retriever
falling back from Chroma to a keyword search when the vector store is down.
"""

from typing import Callable, TypeVar

from app.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class Fallback:
    def __init__(self, primary: Callable[..., T], secondary: Callable[..., T],
                 name: str = "fallback"):
        self._primary = primary
        self._secondary = secondary
        self._name = name

    def run(self, *args, **kwargs) -> T:
        try:
            return self._primary(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — deliberately broad: ANY primary failure falls through
            log.warning("fallback_engaged", name=self._name, error=str(exc)[:150])
            return self._secondary(*args, **kwargs)
