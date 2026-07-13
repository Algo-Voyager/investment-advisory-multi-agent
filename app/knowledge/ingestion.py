"""Knowledge-base ingestion: SEC EDGAR filings → chunks → ChromaDB (Gemini embeddings).

THE critical rule (docs/data_assumptions.md §4): only INDIVIDUAL STOCKS file
10-K/10-Q/8-K. ETFs and cash (VTI, BND, QQQ, CASH…) are skipped with a log line,
never errored — so fund-only clients (CLT-001/003/004) legitimately end up with
zero filings, which is correct behaviour, not a bug.

Idempotent: a filing whose chunks are already stored is skipped, so re-running
(manually now, nightly cron in production) only adds what's new.
"""

import re

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.data.repositories import portfolio_repo
from app.errors.exceptions import CoPilotError
from app.integrations.sec_edgar_adapter import SECEdgarAdapter
from app.knowledge.keyword_fallback import mirror_chunk
from app.knowledge.vector_store import VectorStore, get_vector_store
from app.logging import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CHUNKS_PER_FILING = 120  # a full 10-K can be 500+ chunks; cap embedding cost
# (free tier ≈100 embed requests/min — the embedder sub-batches and backs off on 429)

_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-C]?)\b[.:—-]?", re.IGNORECASE)


def stock_tickers_in_book() -> tuple[list[str], list[str]]:
    """(individual_stock_tickers, skipped_non_stock_symbols) across all clients."""
    stocks, skipped = set(), set()
    for client_id in portfolio_repo.client_ids():
        for h in portfolio_repo.get(client_id).holdings:
            (stocks if h.is_individual_stock else skipped).add(h.symbol)
    return sorted(stocks), sorted(skipped)


def html_to_text(html: str) -> str:
    """Strip an EDGAR filing's HTML to readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):  # tables become noise as flat text
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t\xa0]+", " ", text)).strip()


def chunk_filing(text: str, base_meta: dict) -> tuple[list[str], list[dict]]:
    """Split into overlapping chunks; tag each with the SEC 'Item' section it falls in."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                              chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)[:MAX_CHUNKS_PER_FILING]
    metas, section = [], "unknown"
    for chunk in chunks:
        match = _ITEM_RE.search(chunk)
        if match:
            section = f"Item {match.group(1).upper()}"  # carries forward until the next header
        metas.append({**base_meta, "section": section})
    return chunks, metas


def ingest(tickers: list[str] | None = None, limit: int = 3,
           forms: tuple = ("10-K", "10-Q"), store: VectorStore | None = None,
           adapter: SECEdgarAdapter | None = None) -> dict:
    """Download recent filings for individual-stock tickers and index them.

    tickers=None → every individual stock across all client portfolios.
    """
    store = store or get_vector_store()
    adapter = adapter or SECEdgarAdapter()

    book_stocks, book_funds = stock_tickers_in_book()
    if tickers is None:
        tickers = book_stocks
        log.info("ingest_skipping_non_stocks", skipped=book_funds)
    else:
        funds_requested = [t for t in tickers if t in book_funds]
        for t in funds_requested:
            log.info("ingest_skip_fund", ticker=t,
                     reason="ETFs/cash don't file 10-K/10-Q")
        tickers = [t for t in tickers if t not in funds_requested]

    summary = {"ingested": {}, "skipped_existing": [], "failed": {}}
    for ticker in tickers:
        try:
            listing = adapter.get_recent_filings(ticker, forms=forms, limit=limit)
        except CoPilotError as exc:
            summary["failed"][ticker] = str(exc)[:120]
            log.warning("ingest_listing_failed", ticker=ticker, error=str(exc)[:100])
            continue

        for filing in listing["filings"]:
            accession = filing["accession"].replace("-", "")
            probe_id = f"{accession}-0"
            if store.has_ids("sec_filings", [probe_id]):
                summary["skipped_existing"].append(filing["accession"])
                continue  # idempotence — already indexed
            try:
                html = adapter.get_filing_document(filing["url"])
                text = html_to_text(html)
                base_meta = {
                    "ticker": ticker,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    # int yyyymmdd twin enables $gte/$lte range filters in Chroma
                    "filing_date_int": int(filing["filing_date"].replace("-", "")),
                    "accession": filing["accession"],
                }
                chunks, metas = chunk_filing(text, base_meta)
                ids = [f"{accession}-{i}" for i in range(len(chunks))]
                added = store.upsert("sec_filings", ids, chunks, metas)
                # Mirror to disk (Phase 11): keeps citations available via keyword
                # search if ChromaDB is ever unreachable at query time.
                for cid, chunk, meta in zip(ids, chunks, metas):
                    mirror_chunk("sec_filings", cid, chunk, meta)
                summary["ingested"][f"{ticker} {filing['form']} {filing['filing_date']}"] = added
                log.info("ingest_filing_done", ticker=ticker, form=filing["form"],
                         date=filing["filing_date"], chunks=added)
            except CoPilotError as exc:
                summary["failed"][f"{ticker} {filing['form']}"] = str(exc)[:120]
                log.warning("ingest_filing_failed", ticker=ticker, error=str(exc)[:100])

    summary["collection_size"] = store.count("sec_filings")
    return summary


def archive_news(tickers: list[str], store: VectorStore | None = None) -> int:
    """Optional: snapshot today's headlines into the news_archive collection,
    so 'do you see impact of X news' questions can search past coverage."""
    from app.tools.market_tools import get_recent_news

    store = store or get_vector_store()
    ids, docs, metas = [], [], []
    for ticker in tickers:
        result = get_recent_news(ticker)
        for item in result.get("news", []):
            uid = f"news-{ticker}-{abs(hash(item['title'])) % 10**10}"
            ids.append(uid)
            docs.append(f"{item['title']} ({item['publisher']}, {item['published']})")
            metas.append({"ticker": ticker, "published": item["published"],
                          "publisher": item["publisher"], "url": item["url"]})
    if ids:
        store.upsert("news_archive", ids, docs, metas)
    return len(ids)
