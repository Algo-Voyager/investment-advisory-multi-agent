"""AnswerAccuracy — exact / substring / numeric-tolerance matching.

Deterministic, no LLM. Checks the harness's final_answer against a dataset
item's `expected_substrings` (all must appear, case-insensitive) — the cheap
first line of defense before RAGAS's semantic metrics.
"""

import re

from evals.metrics.base import Metric, MetricReport


class AnswerAccuracy(Metric):
    name = "answer_accuracy"

    def _score(self, dataset, results) -> MetricReport:
        per_item = []
        for item, result in zip(dataset, results):
            expected = item.get("expected_substrings", [])
            if result.error:
                per_item.append({"id": item["id"], "passed": False, "reason": result.error})
                continue
            answer_lower = result.final_answer.lower()
            missing = [e for e in expected if e.lower() not in answer_lower]
            passed = not missing
            per_item.append({"id": item["id"], "passed": passed,
                            "reason": f"missing: {missing}" if missing else ""})
        score = sum(1 for p in per_item if p["passed"]) / len(per_item) if per_item else 0.0
        return MetricReport(self.name, score, per_item)


class NumericTolerance(Metric):
    """For items with an `expected_number` + `tolerance_pct` — checks the answer
    contains a number within tolerance. Used for RAG's precise figures if needed."""

    name = "numeric_tolerance"
    _NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")

    def _score(self, dataset, results) -> MetricReport:
        per_item = []
        for item, result in zip(dataset, results):
            expected = item.get("expected_number")
            if expected is None:
                continue  # not applicable to this item
            tol = item.get("tolerance_pct", 5) / 100
            found = [float(m.replace(",", "")) for m in self._NUM.findall(result.final_answer)]
            passed = any(abs(f - expected) <= abs(expected) * tol for f in found)
            per_item.append({"id": item["id"], "passed": passed})
        score = sum(1 for p in per_item if p["passed"]) / len(per_item) if per_item else 1.0
        return MetricReport(self.name, score, per_item, notes="only scores items with expected_number")
