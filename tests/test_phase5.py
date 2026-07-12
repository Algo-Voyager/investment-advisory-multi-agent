"""Phase 5 tests — vector store, ingestion mechanics, retriever, RAG tools.

Offline tier uses a deterministic fake embedder + tmp Chroma dir (no network,
no Gemini). The live tier runs the real acceptance flow and is skipped unless
the knowledge base has been ingested (run scripts.ingest_kb first).
"""

import hashlib
import os
from pathlib import Path

import pytest

if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.knowledge.ingestion import chunk_filing, html_to_text, stock_tickers_in_book  # noqa: E402
from app.knowledge.retriever import Retriever  # noqa: E402
from app.knowledge.vector_store import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Deterministic 8-dim 'embeddings' from character histograms — enough for
    exact-text matching in tests, zero network."""

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255 for b in digest[:8]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture()
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"), embedder=FakeEmbedder())


# ---------------------------------------------------------------- ingestion mechanics
class TestIngestionMechanics:
    def test_html_to_text_strips_tags_scripts_and_tables(self):
        html = ("<html><body><h1>Annual Report</h1><script>evil()</script>"
                "<table><tr><td>noise</td></tr></table>"
                "<p>Data center revenue grew 279% year-over-year.</p></body></html>")
        text = html_to_text(html)
        assert "Data center revenue grew 279%" in text
        assert "evil" not in text and "noise" not in text and "<p>" not in text

    def test_chunking_overlaps_and_carries_section_metadata(self):
        text = ("Item 1. Business\n" + "We design GPUs. " * 100 +
                "\nItem 1A. Risk Factors\n" + "Competition is intense. " * 100)
        chunks, metas = chunk_filing(text, {"ticker": "NVDA", "form": "10-K",
                                            "filing_date": "2026-02-01",
                                            "filing_date_int": 20260201,
                                            "accession": "acc-1"})
        assert len(chunks) > 2
        assert all(len(c) <= 1000 for c in chunks)
        sections = {m["section"] for m in metas}
        assert "Item 1" in sections and "Item 1A" in sections
        assert all(m["ticker"] == "NVDA" for m in metas)

    def test_book_scan_finds_stocks_and_skips_funds(self):
        stocks, skipped = stock_tickers_in_book()
        assert "NVDA" in stocks and "AAPL" in stocks
        assert "VTI" in skipped and "CASH" in skipped  # ETFs/cash never ingest
        assert not set(stocks) & set(skipped)


# ---------------------------------------------------------------- vector store
class TestVectorStore:
    DOCS = ["NVIDIA data center revenue grew 279% year over year",
            "Apple reported iPhone sales decline in China",
            "Microsoft Azure cloud growth accelerated to 30%"]
    METAS = [{"ticker": "NVDA", "form": "10-Q", "filing_date": "2026-05-28",
              "filing_date_int": 20260528, "section": "Item 2"},
             {"ticker": "AAPL", "form": "10-Q", "filing_date": "2026-05-01",
              "filing_date_int": 20260501, "section": "Item 2"},
             {"ticker": "MSFT", "form": "10-K", "filing_date": "2025-07-30",
              "filing_date_int": 20250730, "section": "Item 7"}]

    def _seed(self, store):
        store.upsert("sec_filings", ["d0", "d1", "d2"], self.DOCS, self.METAS)

    def test_upsert_query_roundtrip(self, store):
        self._seed(store)
        assert store.count("sec_filings") == 3
        hits = store.query("sec_filings",
                           "NVIDIA data center revenue grew 279% year over year", k=1)
        assert hits[0]["metadata"]["ticker"] == "NVDA"  # exact text → nearest by fake embed

    def test_upsert_same_ids_is_idempotent_not_duplicating(self, store):
        self._seed(store)
        self._seed(store)  # again
        assert store.count("sec_filings") == 3

    def test_where_filter_scopes_by_ticker(self, store):
        self._seed(store)
        hits = store.query("sec_filings", "revenue growth", k=3, where={"ticker": "MSFT"})
        assert len(hits) == 1 and hits[0]["metadata"]["ticker"] == "MSFT"

    def test_empty_collection_returns_empty_not_error(self, store):
        assert store.query("news_archive", "anything", k=3) == []


# ---------------------------------------------------------------- retriever facade
class TestRetriever:
    def test_returns_content_citation_tuples_with_filters(self, store):
        TestVectorStore()._seed(store)
        retriever = Retriever(store=store)
        results = retriever.query("data center growth", filters={"ticker": "NVDA"}, k=3)
        assert len(results) == 1
        content, citation = results[0]
        assert "NVIDIA" in content
        assert citation["form"] == "10-Q" and citation["filing_date"] == "2026-05-28"

    def test_date_range_filter(self, store):
        TestVectorStore()._seed(store)
        retriever = Retriever(store=store)
        recent = retriever.query("growth", filters={"date_range": ("2026-01-01", "2026-12-31")}, k=5)
        assert {c["ticker"] for _, c in recent} == {"NVDA", "AAPL"}  # MSFT 2025 excluded


# ---------------------------------------------------------------- rag tools (offline via injected store)
class TestRagTools:
    def test_search_filings_builds_citations_and_freshness(self, store, monkeypatch):
        import app.tools.rag_tools as rag

        TestVectorStore()._seed(store)
        monkeypatch.setattr(rag, "_retriever", Retriever(store=store))
        result = rag.search_filings("data center revenue growth", ticker="NVDA")
        assert result["results"][0]["citation"] == "[source: 10-Q NVDA 2026-05-28]"
        assert result["freshness_disclosure"] == "(Filing data as of 2026-05-28)"

    def test_search_filings_empty_is_honest(self, store, monkeypatch):
        import app.tools.rag_tools as rag

        monkeypatch.setattr(rag, "_retriever", Retriever(store=store))  # empty store
        result = rag.search_filings("anything", ticker="AAPL")
        assert result["results"] == []
        assert "No relevant filings" in result["message"]

    def test_rag_tools_registered_for_both_research_agents(self):
        from app.tools.registry import tool_registry

        table = tool_registry.table()
        for agent in ("market_research", "securities_analysis"):
            names = [r["tool"] for r in table if r["agent"] == agent]
            assert "search_filings" in names and "search_news_archive" in names


# ---------------------------------------------------------------- live acceptance
def _kb_ingested() -> bool:
    try:
        from app.knowledge.vector_store import get_vector_store

        return get_vector_store().count("sec_filings") > 0
    except Exception:
        return False


@pytest.mark.skipif(not _kb_ingested(), reason="knowledge base empty — run scripts.ingest_kb")
class TestLiveAcceptance:
    def test_nvda_query_is_grounded_with_citation(self):
        from app.tools.rag_tools import search_filings

        result = search_filings("data center demand and revenue growth", ticker="NVDA")
        assert result["results"], "expected NVDA chunks in the knowledge base"
        assert "NVDA" in result["results"][0]["citation"]
        assert result["freshness_disclosure"].startswith("(Filing data as of")

    def test_not_ingested_ticker_returns_no_relevant_filings(self):
        from app.tools.rag_tools import search_filings

        result = search_filings("anything at all", ticker="ZZZZ")
        assert result["results"] == []
        assert "No relevant filings" in result["message"]
