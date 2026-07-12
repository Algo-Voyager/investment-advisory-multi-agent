"""MACD — Moving Average Convergence/Divergence (12/26 EMA, 9-period signal).

MACD line = EMA12 − EMA26; signal = EMA9 of the MACD line; histogram = the gap.
Line above signal → bullish momentum; a recent sign flip of the histogram marks
a crossover.
"""

import pandas as pd

from app.indicators.base import Indicator


class MACD(Indicator):
    name = "macd"

    def compute(self, data: pd.Series, fast: int = 12, slow: int = 26,
                signal: int = 9, **_) -> dict:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        crossover = "none"
        if len(histogram) >= 2:
            prev, last = float(histogram.iloc[-2]), float(histogram.iloc[-1])
            if prev <= 0 < last:
                crossover = "bullish_crossover"   # line just crossed above signal
            elif prev >= 0 > last:
                crossover = "bearish_crossover"
        return {
            "indicator": "macd",
            "macd_line": round(float(macd_line.iloc[-1]), 4),
            "signal_line": round(float(signal_line.iloc[-1]), 4),
            "histogram": round(float(histogram.iloc[-1]), 4),
            "momentum": "bullish" if histogram.iloc[-1] > 0 else "bearish",
            "recent_crossover": crossover,
        }
