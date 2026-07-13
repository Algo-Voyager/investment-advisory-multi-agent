"""Retriever — Facade pattern.

One call — `.query(text, filters, k)` — hides everything: Gemini query embedding,
Chroma search, metadata filtering (ticker / form / date range). Agents receive
`(content, citation)` tuples and never learn Chroma exists.

Why metadata filters matter: without `ticker`, an Apple question could surface
NVIDIA text purely by embedding similarity. Filters cut that failure mode off.

Resilience (Phase 11): if Chroma is unreachable, `Fallback` drops to a keyword
search over the on-disk mirror in `data/knowledge_base/` (populated at ingestion
time) — dumber ranking, but citations keep flowing instead of the tool erroring.
"""

from app.errors.fallback import Fallback
from app.knowledge.keyword_fallback import keyword_search
from app.knowledge.vector_store import VectorStore, get_vector_store
from app.logging import get_logger

log = get_logger(__name__)


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self._store = store or get_vector_store()

    def query(self, text: str, filters: dict | None = None, k: int = 5,
              collection: str = "sec_filings") -> list[tuple[str, dict]]:
        """Return up to k `(content, citation_dict)` tuples, best match first.

        filters: {'ticker': 'NVDA', 'form': '10-Q',
                  'date_range': ('2025-01-01', '2026-12-31')}   # all optional
        """
        filters = filters or {}

        def via_chroma():
            where = self._build_where(filters)
            hits = self._store.query(collection, text, k=k, where=where)
            return [(h["document"], h["metadata"] or {}, h["distance"]) for h in hits]

        def via_keyword():
            log.warning("retrieval_fallback_engaged", collection=collection)
            raw = keyword_search(text, ticker=filters.get("ticker"), collection=collection, k=k)
            return [(doc, meta, None) for doc, meta in raw]

        raw_hits = Fallback(via_chroma, via_keyword, name="retriever").run()
        results = []
        for doc, meta, distance in raw_hits:
            citation = {
                "ticker": meta.get("ticker"),
                "form": meta.get("form"),
                "filing_date": meta.get("filing_date", meta.get("published")),
                "section": meta.get("section"),
                "distance": distance,
            }
            results.append((doc, citation))
        log.info("retrieval", collection=collection, hits=len(results), filters=filters)
        return results

    @staticmethod
    def _build_where(filters: dict) -> dict | None:
        clauses = []
        if filters.get("ticker"):
            clauses.append({"ticker": filters["ticker"].upper()})
        if filters.get("form"):
            clauses.append({"form": filters["form"].upper()})
        if filters.get("date_range"):
            start, end = filters["date_range"]
            clauses.append({"filing_date_int": {"$gte": int(start.replace("-", ""))}})
            clauses.append({"filing_date_int": {"$lte": int(end.replace("-", ""))}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}
