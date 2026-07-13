# XZY Capital — Investment Advisory Co-Pilot

A multi-agent investment advisory system built on **LangGraph**, powered exclusively by **Google Gemini** (chat + embeddings). Take-home assignment prototype for a fictional boutique advisory firm.

## Quickstart

```bash
make install                 # creates .venv and installs dependencies
cp .env.example .env         # then fill in GOOGLE_API_KEY (and SEC_USER_AGENT) — see .env setup below
make run                     # quick CLI sanity check: "what do I own?" for CLT-001
make ui                      # launch the Streamlit demo UI at localhost:8501
make test                    # run the offline test suite
make notebook                # open the per-phase notebooks (start with phase00)
```

### `.env` setup (Gemini-only)

Copy `.env.example` → `.env` and fill in, at minimum:

```dotenv
GOOGLE_API_KEY=your-gemini-key-from-aistudio.google.com/apikey   # the ONLY required key
SEC_USER_AGENT=Your Name your-email@example.com                  # required once you reach RAG (Phase 5)
```

Everything else (`ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `LANGSMITH_API_KEY`) is optional and the system degrades gracefully without it — market data falls back to keyless `yfinance`, and tracing simply doesn't activate. Full details, model-tier rationale, and observed free-tier rate-limit gotchas: [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md) §4–5.

## Design docs

The System Design Document deliverable: [`docs/architecture.md`](docs/architecture.md) (C4 diagrams, live agent graph, inter-agent protocol, data flow, worked example, integrations) · [`docs/design_patterns.md`](docs/design_patterns.md) (every pattern applied, verified against file/line) · [`docs/production_plan.md`](docs/production_plan.md) (bonus: production architecture + go-to-plan).

## Phases

Each phase links its runnable companion notebook (`notebooks/phaseXX_*.ipynb`).

- [x] **[Phase 0](notebooks/phase00_scaffold.ipynb)** — scaffold, settings, Gemini factory, logging, exceptions, data in place
- [x] **[Phase 1](notebooks/phase01_data_and_portfolio_agent.ipynb)** — data models, Excel repository (union), portfolio tools, Portfolio agent, graph + CLI
- [x] **[Phase 2](notebooks/phase02_supervisor_and_routing.ipynb)** — supervisor + router (LLM/keyword strategies), Market Research agent, streaming CLI
- [x] **[Phase 3](notebooks/phase03_adapters_registry_resilience.ipynb)** — adapter chain (Finnhub/AlphaVantage/yfinance/SEC EDGAR), tool registry, cached+retry+rate-limited decorators, Gemini rate limiter
- [x] **[Phase 4](notebooks/phase04_securities_analysis.ipynb)** — Securities Analysis agent: RSI/SMA/EMA/MACD/Bollinger/ATR indicators (Strategy + Factory), technical_analysis + compare_indicators tools
- [x] **[Phase 5](notebooks/phase05_rag_knowledge_base.ipynb)** — RAG knowledge base: SEC EDGAR ingestion → ChromaDB (Gemini embeddings, 768d), Retriever facade, search_filings with citations + freshness disclosure
- [x] **[Phase 6](notebooks/phase06_risk_assessment.ipynb)** — Risk Assessment agent: volatility, beta (equity sleeve), VaR (historical/parametric Strategies), concentration, synthetic profiles + tolerance mismatch check
- [x] **[Phase 7](notebooks/phase07_memory_and_isolation.ipynb)** — persistent memory (SQLite checkpointer + long-term store), thread-id isolation, access-control interceptor (bonus: session isolation + data privacy)
- [x] **[Phase 8](notebooks/phase08_planning_and_synthesis.ipynb)** — planner (complexity Strategy + decomposition), supervisor plan-walk, Chain-of-Thought synthesizer with conflict handling
- [x] **[Phase 9](notebooks/phase09_guardrails_and_reflection.ipynb)** — guardrail pipeline (input: PII/injection/scope; output: numeric/citation/groundedness/conflict), reflection revise-loop, safe exit
- [x] **[Phase 10](notebooks/phase10_clarification_hitl.ipynb)** — human-in-the-loop clarification via `interrupt()`/`Command(resume=...)`, CLI prompt/resume flow
- [x] **[Phase 11](notebooks/phase11_error_handling_circuit_breaker.ipynb)** — per-adapter circuit breakers, Fallback helper (Retriever→keyword search, market adapters→yfinance), global node exception wrapping (no stack trace ever reaches the user)
- [ ] [Phase 12](notebooks/phase12_optimization.ipynb) — Optimization Agent (bonus, not yet implemented)
- [x] **[Phase 13](notebooks/phase13_evaluation.ipynb)** — Gemini-powered RAGAS evaluation suite (Faithfulness/ResponseRelevancy/LLMContextPrecision) + deterministic metrics (routing/answer/guardrail accuracy), 75-item hand-crafted dataset, markdown+chart report
- [x] **[Phase 14](notebooks/phase14_ui_smoke.ipynb)** — Streamlit UI: client selector, streaming turn view, portfolio panel, citations panel, human-in-the-loop clarification buttons
- [x] **[Phase 15](notebooks/phase15_deliverables.ipynb)** — architecture, design patterns, and production plan docs (see Design docs above)
