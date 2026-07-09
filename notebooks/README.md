# Phase Notebooks

One runnable notebook per build phase. Run `phaseXX_*.ipynb` **after** Claude Code finishes phase XX.

Each notebook: loads `.env` (Gemini-only) -> exercises the phase against real client data ->
runs the acceptance check -> (for graph phases) renders the LangGraph as Mermaid/PNG.

Order: 00 scaffold -> 01 data+portfolio -> 02 supervisor/routing -> 03 adapters -> 04 securities ->
05 RAG -> 06 risk -> 07 memory -> 08 planning -> 09 guardrails -> 10 clarification ->
11 error handling -> 12 optimization -> 13 evaluation -> 14 UI -> 15 deliverables.

Prereqs: see ../00_prerequisites_and_setup.md . Minimum: GOOGLE_API_KEY in .env (+ SEC_USER_AGENT for phase 05+).
