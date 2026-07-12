"""VectorStore — thin wrapper around ChromaDB with Gemini embeddings.

Design decisions:
- WE compute embeddings (via the factory's `get_embeddings()`, gemini-embedding-001
  pinned to 768 dims) and hand Chroma ready vectors. Chroma never embeds anything —
  that sidesteps its embedding-function interface churn and keeps the "one embedding
  model, one dimension, forever" rule enforced in exactly one place.
- The embedder is injectable so tests can run offline with a deterministic fake.
- Persist dir is anchored to the project root (same notebook-cwd lesson as the
  Excel repository).

Collections: sec_filings | news_archive | research_notes.
"""

from pathlib import Path

import chromadb

from app.config import settings
from app.logging import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

COLLECTIONS = ("sec_filings", "news_archive", "research_notes")


class VectorStore:
    def __init__(self, persist_dir: str | None = None, embedder=None):
        path = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        if not path.is_absolute():
            path = _ROOT / path
        self._client = chromadb.PersistentClient(path=str(path))
        self._embedder = embedder  # lazily defaults to Gemini on first use

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        return self._get_embedder().embed_query(text)

    def _get_embedder(self):
        if self._embedder is None:
            from app.llm.factory import get_embeddings  # deferred: offline paths never touch it

            self._embedder = get_embeddings()
        return self._embedder

    def collection(self, name: str):
        assert name in COLLECTIONS, f"unknown collection '{name}'"
        return self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def upsert(self, collection: str, ids: list[str], documents: list[str],
               metadatas: list[dict]) -> int:
        """Embed + upsert. Idempotent by id — re-upserting the same ids overwrites."""
        if not ids:
            return 0
        embeddings = self._embed_documents(documents)
        self.collection(collection).upsert(ids=ids, documents=documents,
                                           embeddings=embeddings, metadatas=metadatas)
        log.info("vector_upsert", collection=collection, chunks=len(ids))
        return len(ids)

    def query(self, collection: str, text: str, k: int = 5,
              where: dict | None = None) -> list[dict]:
        """Semantic search → [{'document', 'metadata', 'distance'}], best first."""
        col = self.collection(collection)
        if col.count() == 0:
            return []
        result = col.query(query_embeddings=[self._embed_query(text)], n_results=k,
                           where=where or None)
        hits = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0],
                                   result["distances"][0]):
            hits.append({"document": doc, "metadata": meta, "distance": round(dist, 4)})
        return hits

    def has_ids(self, collection: str, ids: list[str]) -> bool:
        """True if ALL the ids are already stored (ingestion's skip check)."""
        found = self.collection(collection).get(ids=ids)
        return len(found["ids"]) == len(ids)

    def count(self, collection: str) -> int:
        return self.collection(collection).count()


_default_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
