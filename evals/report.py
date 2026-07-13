"""Evaluation report generator — `make eval` entry point.

Runs all 4 datasets through the harness, scores them with the appropriate
metrics, and writes `docs/evaluation_report.md` (tables + a matplotlib bar
chart) plus `docs/evaluation_results.json` (raw per-item data, for
re-analysis without re-running the graph).

Quota-aware by design: Gemini's free tier is tight (see
docs/00_prerequisites_and_setup.md §"rate limits"), so every dataset accepts a
`--limit`. Defaults here are deliberately small; pass `--full` for a complete run.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed to write a PNG
import matplotlib.pyplot as plt

from evals.harness import load_dataset, results_to_dicts, run_dataset
from evals.metrics.answer_accuracy import AnswerAccuracy
from evals.metrics.guardrail_effectiveness import GuardrailEffectiveness
from evals.metrics.ragas_metrics import RagasMetrics
from evals.metrics.routing_accuracy import RoutingAccuracy
from app.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _run_and_score(dataset_name: str, metrics: list, limit: int | None) -> dict:
    dataset = load_dataset(dataset_name)
    if limit:
        dataset = dataset[:limit]
    results = run_dataset(dataset_name, limit=limit)
    reports = [m.compute(dataset, results) for m in metrics]
    latencies = sorted(r.latency_s for r in results)

    def pct(p):
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return round(latencies[idx], 2)

    return {
        "dataset": dataset_name,
        "n_items": len(dataset),
        "metrics": [{"name": r.name, "score": r.score, "notes": r.notes,
                    "per_item": r.per_item} for r in reports],
        "latency_p50_s": pct(0.5),
        "latency_p95_s": pct(0.95),
        "errors": [r.error for r in results if r.error],
        "results": results_to_dicts(results),
    }


def _failure_examples(dataset_report: dict, n: int = 3) -> list[dict]:
    examples = []
    for metric in dataset_report["metrics"]:
        for item in metric["per_item"]:
            if not item.get("passed", True) and len(examples) < n:
                examples.append({"metric": metric["name"], **item})
    return examples


def _write_chart(all_reports: list[dict], out_path: Path) -> None:
    names, scores = [], []
    for report in all_reports:
        for metric in report["metrics"]:
            names.append(f"{report['dataset'].replace('_queries', '')}\n{metric['name']}")
            scores.append(metric["score"])

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    bars = ax.bar(names, scores, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("XZY Co-Pilot — Evaluation Scores by Dataset & Metric")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="0.7 reference")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}",
               ha="center", fontsize=8)
    plt.xticks(rotation=0, fontsize=8)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_markdown(all_reports: list[dict], chart_path: Path, md_path: Path) -> None:
    lines = [
        "# XZY Co-Pilot — Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "All LLM calls (agents, planner, synthesizer, guardrails, and RAGAS's judge) "
        "run on **Google Gemini** — no OpenAI key exists anywhere in this project.",
        "",
        "## Scores by dataset",
        "",
    ]
    for report in all_reports:
        lines.append(f"### {report['dataset']} (n={report['n_items']})")
        lines.append("")
        lines.append("| Metric | Score | Notes |")
        lines.append("|---|---|---|")
        for metric in report["metrics"]:
            lines.append(f"| {metric['name']} | {metric['score']:.3f} | {metric['notes']} |")
        lines.append("")
        lines.append(f"Latency — p50: {report['latency_p50_s']}s, p95: {report['latency_p95_s']}s")
        if report["errors"]:
            lines.append(f"\n⚠️ {len(report['errors'])} item(s) errored: {report['errors'][:3]}")
        lines.append("")

    lines.append("## Score chart")
    lines.append("")
    lines.append(f"![evaluation scores]({chart_path.name})")
    lines.append("")

    lines.append("## Top failure modes")
    lines.append("")
    any_failures = False
    for report in all_reports:
        examples = _failure_examples(report)
        if not examples:
            continue
        any_failures = True
        lines.append(f"**{report['dataset']}:**")
        for ex in examples:
            lines.append(f"- `{ex['id']}` ({ex['metric']}): {ex}")
        lines.append("")
    if not any_failures:
        lines.append("No failures recorded in this run.")
        lines.append("")

    lines.append("## Notes on methodology")
    lines.append("")
    lines.append("- `simple_queries` / `analytical_queries`: deterministic answer-accuracy "
                 "(substring match) + routing-accuracy (did the expected specialist(s) fire).")
    lines.append("- `rag_queries`: Gemini-powered RAGAS — Faithfulness, Answer Relevancy, "
                 "Context Precision — via `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` "
                 "around this project's own `get_llm(reasoning=True)` / `get_embeddings()`.")
    lines.append("- `adversarial_queries`: guardrail-effectiveness — did prompt-injection/PII/"
                 "off-topic/cross-client attempts get blocked, redacted, or refused as expected.")
    lines.append("- Clarification (`interrupt()`) is force-disabled for eval runs "
                 "(`settings.ENABLE_CLARIFICATION=False`) so batch scoring never blocks on a human.")

    md_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite and write the report")
    parser.add_argument("--limit", type=int, default=5,
                        help="items per dataset (default 5 — Gemini free tier is quota-limited; "
                             "use --full for everything)")
    parser.add_argument("--full", action="store_true", help="ignore --limit, run entire datasets")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="skip the RAGAS metric (saves Gemini reasoning-tier quota)")
    args = parser.parse_args()
    limit = None if args.full else args.limit

    DOCS.mkdir(exist_ok=True)
    all_reports = []

    for name, metrics in [
        # simple_queries has both expected_substrings AND expected_agents — both apply.
        ("simple_queries", [AnswerAccuracy(), RoutingAccuracy()]),
        # analytical_queries only asserts expected_agents (answers are open-ended,
        # not substring-matchable) — RoutingAccuracy only.
        ("analytical_queries", [RoutingAccuracy()]),
        # rag_queries has neither field — it's scored purely by RAGAS's semantic judge.
        ("rag_queries", [] if args.skip_ragas else [RagasMetrics()]),
        ("adversarial_queries", [GuardrailEffectiveness()]),
    ]:
        log.info("eval_dataset_start", dataset=name, limit=limit)
        report = _run_and_score(name, metrics, limit)
        all_reports.append(report)
        log.info("eval_dataset_done", dataset=name,
                 scores={m["name"]: m["score"] for m in report["metrics"]})

    (DOCS / "evaluation_results.json").write_text(
        json.dumps(all_reports, indent=2, default=str))

    chart_path = DOCS / "evaluation_scores.png"
    _write_chart(all_reports, chart_path)
    _write_markdown(all_reports, chart_path, DOCS / "evaluation_report.md")

    print(f"Wrote {DOCS / 'evaluation_report.md'}")
    for report in all_reports:
        for metric in report["metrics"]:
            print(f"  {report['dataset']:22} {metric['name']:14} {metric['score']:.3f}")


if __name__ == "__main__":
    main()
