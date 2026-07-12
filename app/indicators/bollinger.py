"""Bollinger Bands — SMA(20) ± 2 standard deviations.

%B tells where price sits inside the bands (0 = at lower band, 1 = at upper);
bandwidth measures volatility squeeze/expansion.
"""

import pandas as pd

from app.indicators.base import Indicator


class BollingerBands(Indicator):
    name = "bollinger"

    def compute(self, data: pd.Series, window: int = 20, num_std: float = 2.0, **_) -> dict:
        mid = data.rolling(window).mean()
        std = data.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        price = float(data.iloc[-1])
        up, lo, mi = float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
        pct_b = (price - lo) / (up - lo) if up != lo else 0.5
        return {
            "indicator": f"bollinger_{window}",
            "upper": round(up, 2),
            "middle": round(mi, 2),
            "lower": round(lo, 2),
            "price": round(price, 2),
            "percent_b": round(pct_b, 3),          # >1 above upper band, <0 below lower
            "bandwidth_pct": round((up - lo) / mi * 100, 2),
        }
