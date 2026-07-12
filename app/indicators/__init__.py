"""IndicatorFactory — Factory pattern.

Tools request indicators by STRING ("rsi", "sma_50", "ema_20", "macd",
"bollinger", "atr"), so the LLM can name them in a tool call and the factory
resolves parameterized variants (window sizes) from the name itself.
"""

from app.errors.exceptions import ToolError
from app.indicators.atr import ATR
from app.indicators.base import Indicator
from app.indicators.bollinger import BollingerBands
from app.indicators.macd import MACD
from app.indicators.moving_averages import MovingAverage
from app.indicators.rsi import RSI


class IndicatorFactory:
    KNOWN = ("rsi", "sma", "ema", "macd", "bollinger", "atr")

    def get(self, name: str) -> Indicator:
        """Resolve 'rsi', 'rsi_21', 'sma_50', 'ema_12', 'macd', 'bollinger', 'atr'."""
        key = name.lower().strip()
        base, _, suffix = key.partition("_")
        window = int(suffix) if suffix.isdigit() else None

        if base == "rsi":
            return RSI()  # window is passed at compute() time via the tool
        if base in ("sma", "ema"):
            return MovingAverage(window=window or 20, kind=base)
        if base == "macd":
            return MACD()
        if base == "bollinger":
            return BollingerBands()
        if base == "atr":
            return ATR()
        raise ToolError(f"Unknown indicator '{name}'. Known: {', '.join(self.KNOWN)} "
                        f"(windows via suffix, e.g. sma_50)")


indicator_factory = IndicatorFactory()
