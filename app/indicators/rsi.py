"""RSI — Relative Strength Index (Wilder's smoothing, default 14 periods).

RSI = 100 − 100 / (1 + RS), RS = avg gain / avg loss, where the averages use
Wilder's smoothing — an EMA with alpha = 1/window (`ewm(alpha=1/w, adjust=False)`).
Classic reading: >70 overbought, <30 oversold.
"""

import pandas as pd

from app.indicators.base import Indicator


class RSI(Indicator):
    name = "rsi"

    def series(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(100.0)  # all-gain series has loss=0 → RS=inf → RSI 100

    def compute(self, data: pd.Series, window: int = 14, **_) -> dict:
        rsi = self.series(data, window)
        value = round(float(rsi.iloc[-1]), 2)
        if value >= 70:
            zone = "overbought"
        elif value <= 30:
            zone = "oversold"
        else:
            zone = "neutral"
        return {"indicator": f"rsi_{window}", "value": value, "zone": zone}
