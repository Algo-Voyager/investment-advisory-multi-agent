"""Indicator — Strategy pattern.

Every technical indicator implements one interface: `.compute(data, **params)`.
Adding Stochastic RSI later is one new file; nothing else changes. The LLM's job
is to EXPLAIN these numbers, never to compute them — that division is the
backbone of hallucination control (Phase 9 audits answers against tool output).

Implementation note: indicators are hand-rolled with pandas (not pandas-ta) —
the library lags Python releases and hand-rolled math is unit-verifiable, which
Phase 4's acceptance explicitly requires.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    name: str

    @abstractmethod
    def compute(self, data: pd.Series | pd.DataFrame, **params) -> dict:
        """Compute the indicator.

        `data` is a close-price Series for most indicators; ATR needs a DataFrame
        with high/low/close columns. Returns a JSON-safe dict of latest values —
        ready for caching and for handing to the LLM.
        """
