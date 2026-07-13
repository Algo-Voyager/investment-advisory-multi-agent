"""Keyword search over `data/knowledge_base/` — the Retriever's fallback when
ChromaDB is unreachable (Phase 11).

Ingestion (Phase 5) mirrors every chunk it embeds into Chroma as a small text
file here, `{ticker}/{accession}-{i}.txt`, with a metadata header. When Chroma
is down, this module does simple case-insensitive whole-word scoring over those
files — much dumber than semantic search, but it keeps citations flowing
instead of the system going dark.
"""

import re
from pathlib import Path

from app.logging import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = _ROOT / "data" / "knowledge_base"


def mirror_chunk(collection: str, chunk_id: str, text: str, metadata: dict) -> None:
    """Write one chunk as a plain-text file so keyword_search can find it later."""
    ticker = metadata.get("ticker", "misc")
    directory = KB_DIR / collection / ticker
    directory.mkdir(parents=True, exist_ok=True)
    header = "".join(f"{k}: {v}\n" for k, v in metadata.items())
    (directory / f"{chunk_id}.txt").write_text(header + "---\n" + text)


def backfill_mirror(store, collection: str = "sec_filings") -> int:
    """One-time utility: mirror chunks already in Chroma from BEFORE mirroring
    existed (e.g. Phase 5's original ingestion, run before this Phase 11 module).
    New ingestions mirror automatically; this only needs running once."""
    col = store.collection(collection)
    got = col.get(include=["documents", "metadatas"])
    count = 0
    for chunk_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        mirror_chunk(collection, chunk_id, doc, meta or {})
        count += 1
    log.info("keyword_mirror_backfilled", collection=collection, chunks=count)
    return count


def keyword_search(query: str, ticker: str | None = None, collection: str = "sec_filings",
                   k: int = 5) -> list[tuple[str, dict]]:
    """Score files by overlapping query words; return top-k (content, metadata)."""
    base = KB_DIR / collection / ticker.upper() if ticker else KB_DIR / collection
    if not base.exists():
        return []
    words = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "a", "an", "of", "in", "and"}
    if not words:
        return []

    scored = []
    for path in base.rglob("*.txt"):
        content = path.read_text()
        body = content.split("---\n", 1)[-1]
        body_words = re.findall(r"[a-z0-9]+", body.lower())
        score = sum(body_words.count(w) for w in words)
        if score > 0:
            header = dict(re.findall(r"^(\w+): (.*)$", content.split("---\n", 1)[0], re.MULTILINE))
            scored.append((score, body, header))
    scored.sort(key=lambda t: t[0], reverse=True)
    log.info("keyword_fallback_search", query=query[:60], hits=len(scored[:k]))
    return [(body, meta) for _score, body, meta in scored[:k]]
