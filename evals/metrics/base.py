"""Metric — Template Method pattern.

`.compute(dataset, results)` is the fixed shape every metric follows: given the
dataset items (with expectations) and the harness's `EvalResult`s (with actual
behaviour), return a `MetricReport`. Each concrete metric is a Strategy — swap
one for another without touching the report generator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MetricReport:
    name: str
    score: float  # 0..1 overall
    per_item: list[dict] = field(default_factory=list)
    notes: str = ""


class Metric(ABC):
    name: str

    def compute(self, dataset: list[dict], results: list) -> MetricReport:
        """Template Method: validate inputs, then delegate to `_score`."""
        if len(dataset) != len(results):
            raise ValueError(f"{self.name}: dataset/results length mismatch "
                            f"({len(dataset)} vs {len(results)})")
        return self._score(dataset, results)

    @abstractmethod
    def _score(self, dataset: list[dict], results: list) -> MetricReport: ...
