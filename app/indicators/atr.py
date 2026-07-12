"""ATR — Average True Range (Wilder's smoothing, default 14). Volatility gauge.

True Range = max(high−low, |high−prev close|, |low−prev close|). Needs OHLC data,
so this indicator takes a DataFrame with high/low/close columns.
"""

import pandas as pd

from app.errors.exceptions import ToolError
from app.indicators.base import Indicator


class ATR(Indicator):
    name = "atr"

    def compute(self, data: pd.DataFrame, window: int = 14, **_) -> dict:
        if not isinstance(data, pd.DataFrame) or not {"high", "low", "close"} <= set(data.columns):
            raise ToolError("ATR needs OHLC data (high/low/close columns)")
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / window, adjust=False).mean()
        value = float(atr.iloc[-1])
        price = float(data["close"].iloc[-1])
        return {
            "indicator": f"atr_{window}",
            "value": round(value, 2),
            "atr_pct_of_price": round(value / price * 100, 2),  # normalized volatility
        }
