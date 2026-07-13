"""RagasMetrics — Gemini-backed faithfulness, answer relevancy, context precision.

**RAGAS uses Gemini exclusively** — `LangchainLLMWrapper` wraps our
`get_llm(reasoning=True)` (gemini-3.5-flash) as the judge, and
`LangchainEmbeddingsWrapper` wraps `get_embeddings()` (gemini-embedding-001).
If any metric ever reaches for OpenAI, that's a wiring bug, not expected
behaviour — this project has no OpenAI key anywhere.

Only scored on `rag_queries` items that actually retrieved context (the two
`expect_no_results` items are skipped — RAGAS needs contexts to judge against).

`evals._ragas_compat` MUST be imported before `ragas` anywhere in this process —
see that module's docstring for why.
"""

import asyncio

import evals._ragas_compat  # noqa: F401 — import before ragas; installs the vertexai shim

from evals.metrics.base import Metric, MetricReport
from app.logging import get_logger

log = get_logger(__name__)


def _build_ragas_wrappers():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.llm.factory import get_embeddings, get_llm

    ragas_llm = LangchainLLMWrapper(get_llm(reasoning=True))
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    return ragas_llm, ragas_embeddings


class RagasMetrics(Metric):
    """Runs Faithfulness, ResponseRelevancy, and LLMContextPrecisionWithoutReference
    over every scoreable rag_queries item, averaged into one report per sub-metric."""

    name = "ragas"

    def _score(self, dataset, results) -> MetricReport:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )

        ragas_llm, ragas_embeddings = _build_ragas_wrappers()
        faithfulness = Faithfulness(llm=ragas_llm)
        # strictness=1: ResponseRelevancy's default (3) asks Gemini for 3 candidate
        # completions in one call ("n=3"), which Gemini's API rejects outright
        # ("Multiple candidates is not enabled for this model"). strictness=1 asks
        # for one generated question instead of three — a standard ragas knob, not
        # a hack; it trades a little statistical smoothing for Gemini compatibility.
        relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)
        precision = LLMContextPrecisionWithoutReference(llm=ragas_llm)

        per_item = []
        for item, result in zip(dataset, results):
            if item.get("expect_no_results") or result.error or not result.retrieved_context:
                continue  # RAGAS needs real retrieved context to judge against
            sample = SingleTurnSample(
                user_input=item["query"],
                response=result.final_answer,
                retrieved_contexts=result.retrieved_context[:5],
            )
            scores = asyncio.run(self._score_one(sample, faithfulness, relevancy, precision))
            per_item.append({"id": item["id"], **scores})
            log.info("ragas_item_scored", item_id=item["id"], **scores)

        if not per_item:
            return MetricReport(self.name, 0.0, [], notes="no items had retrieved context to score")

        avg = lambda key: sum(p[key] for p in per_item) / len(per_item)  # noqa: E731
        overall = (avg("faithfulness") + avg("answer_relevancy") + avg("context_precision")) / 3
        return MetricReport(self.name, round(overall, 3), per_item,
                            notes=f"faithfulness={avg('faithfulness'):.3f} "
                                  f"answer_relevancy={avg('answer_relevancy'):.3f} "
                                  f"context_precision={avg('context_precision'):.3f} "
                                  f"(Gemini-judged, n={len(per_item)})")

    @staticmethod
    async def _score_one(sample, faithfulness, relevancy, precision) -> dict:
        f = await faithfulness.single_turn_ascore(sample)
        r = await relevancy.single_turn_ascore(sample)
        p = await precision.single_turn_ascore(sample)
        return {"faithfulness": round(float(f), 3), "answer_relevancy": round(float(r), 3),
                "context_precision": round(float(p), 3)}
