"""RoutingAccuracy — did the supervisor dispatch the agents the dataset expects?

Deterministic, no LLM. Uses `result.agents_used` (the harness's `visited` list)
against each item's `expected_agents`. Scored as set-overlap (order-independent,
since a valid plan can legitimately reorder steps).
"""

from evals.metrics.base import Metric, MetricReport


class RoutingAccuracy(Metric):
    name = "routing_accuracy"

    def _score(self, dataset, results) -> MetricReport:
        per_item = []
        for item, result in zip(dataset, results):
            expected = set(item.get("expected_agents", []))
            if not expected:
                continue  # item doesn't assert a specific routing expectation
            actual = set(result.agents_used)
            overlap = expected & actual
            recall = len(overlap) / len(expected) if expected else 1.0
            passed = recall >= 0.5  # at least half the expected specialists fired
            per_item.append({"id": item["id"], "passed": passed, "expected": sorted(expected),
                            "actual": sorted(actual), "recall": round(recall, 2)})
        score = sum(1 for p in per_item if p["passed"]) / len(per_item) if per_item else 0.0
        return MetricReport(self.name, score, per_item)
