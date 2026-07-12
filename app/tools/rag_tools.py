"""RAG tools — knowledge-base retrieval for the research-flavoured agents.

Registered for BOTH market_research and securities_analysis (retrieval is a
first-class action for each). Every result carries:
- an inline-ready citation string  [source: {form} {ticker} {filing_date}]
- a programmatic freshness line    (Filing data as of {newest filing_date used})
  built from the chunks' metadata, NEVER from the LLM's memory — ingestion is
  offline/scheduled, so the index can lag; stale must be visible, not silent.
"""

from app.knowledge.retriever import Retriever
from app.logging import get_logger
from app.tools.registry import tool_registry

log = get_logger(__name__)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@tool_registry.register(agent="market_research")
@tool_registry.register(agent="securities_analysis")
def search_filings(query: str, ticker: str | None = None, form: str | None = None) -> dict:
    """Semantic search over ingested SEC filings (10-K/10-Q). Returns snippets with
    citations. Only individual stocks have filings — ETFs/cash never will."""
    filters = {}
    if ticker:
        filters["ticker"] = ticker
    if form:
        filters["form"] = form
    hits = _get_retriever().query(query, filters=filters, k=5)
    if not hits:
        scope = f" for {ticker}" if ticker else ""
        return {"query": query, "results": [],
                "message": f"No relevant filings in the knowledge base{scope}. "
                           f"Either nothing is ingested for this ticker, or it's an "
                           f"ETF/fund (they don't file 10-K/10-Q)."}
    results = [{
        "snippet": doc[:600],
        "citation": f"[source: {cite['form']} {cite['ticker']} {cite['filing_date']}]",
        "section": cite["section"],
        "filing_date": cite["filing_date"],
    } for doc, cite in hits]
    freshest = max(r["filing_date"] for r in results)
    return {
        "query": query,
        "results": results,
        # The agent must end its answer with this line VERBATIM (system prompt enforces it).
        "freshness_disclosure": f"(Filing data as of {freshest})",
    }


@tool_registry.register(agent="market_research")
@tool_registry.register(agent="securities_analysis")
def search_news_archive(query: str, ticker: str | None = None, since: str | None = None) -> dict:
    """Semantic search over the archived news headlines (news_archive collection)."""
    filters = {"ticker": ticker} if ticker else {}
    hits = _get_retriever().query(query, filters=filters, k=5, collection="news_archive")
    if since:
        hits = [(d, c) for d, c in hits
                if (c.get("filing_date") or "0000") >= since]
    if not hits:
        return {"query": query, "results": [],
                "message": "No matching items in the news archive."}
    return {"query": query,
            "results": [{"headline": doc, "published": cite.get("filing_date")}
                        for doc, cite in hits]}


RAG_TOOLS = tool_registry.tools_for("market_research")  # includes these after registration
