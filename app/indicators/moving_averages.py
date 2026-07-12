"""Moving averages — SMA and EMA with a configurable window."""

import math

import pandas as pd

from app.errors.exceptions import ToolError
from app.indicators.base import Indicator


class MovingAverage(Indicator):
    def __init__(self, window: int = 20, kind: str = "sma"):
        self.window = window
        self.kind = kind
        self.name = f"{kind}_{window}"

    def series(self, close: pd.Series) -> pd.Series:
        if self.kind == "ema":
            return close.ewm(span=self.window, adjust=False).mean()
        return close.rolling(self.window).mean()

    def compute(self, data: pd.Series, **_) -> dict:
        ma = self.series(data)
        value = float(ma.iloc[-1])
        # NaN means the window exceeds the data (e.g. SMA200 on a 6mo series).
        # NaN is invalid JSON and would poison the whole LLM tool payload — refuse honestly.
        if math.isnan(value):
            raise ToolError(f"{self.name} needs ≥{self.window} data points, "
                            f"got {len(data)} — request a longer period")
        price = float(data.iloc[-1])
        return {
            "indicator": self.name,
            "value": round(value, 2),
            "price": round(price, 2),
            "price_vs_ma_pct": round((price / value - 1) * 100, 2),
            "position": "above" if price > value else "below",  # above MA = bullish lean
        }
