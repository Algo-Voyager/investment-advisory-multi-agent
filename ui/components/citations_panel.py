"""Citations panel — sources behind the last answer.

Parses the graph's `tool_results` for search_filings/search_news_archive
outputs (the only tools that produce citations) and renders each as a
clickable source line. SEC filing citations link to their real EDGAR URL when
the underlying filing metadata carries one.
"""

import json

import streamlit as st

_RAG_TOOLS = {"search_filings", "search_news_archive"}


def extract_citations(tool_results: dict) -> list[dict]:
    """tool_results: {agent_name: [raw ToolMessage.content strings]} → flat citation list."""
    citations = []
    for outputs in (tool_results or {}).values():
        for raw in outputs:
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or "results" not in payload:
                continue
            for r in payload["results"]:
                if "citation" in r:  # search_filings shape
                    citations.append({
                        "label": r["citation"],
                        "snippet": r.get("snippet", "")[:200],
                        "url": None,
                    })
                elif "headline" in r:  # search_news_archive shape
                    citations.append({"label": r["headline"], "snippet": "",
                                     "url": r.get("url")})
            if payload.get("freshness_disclosure"):
                citations.append({"label": payload["freshness_disclosure"], "snippet": "",
                                 "url": None, "is_freshness_note": True})
    return citations


def render_citations_panel(tool_results: dict) -> None:
    citations = extract_citations(tool_results)
    with st.expander(f"🔗 Citations ({len([c for c in citations if not c.get('is_freshness_note')])})",
                     expanded=bool(citations)):
        if not citations:
            st.caption("No citations for the last answer — it didn't use the knowledge base.")
            return
        for c in citations:
            if c.get("is_freshness_note"):
                st.caption(c["label"])
                continue
            if c["url"]:
                st.markdown(f"- [{c['label']}]({c['url']})")
            else:
                st.markdown(f"- **{c['label']}**")
            if c["snippet"]:
                st.caption(c["snippet"] + "…")
