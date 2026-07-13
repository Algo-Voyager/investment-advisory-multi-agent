# XZY Capital — Investment Advisory Co-Pilot

A multi-agent investment advisory system built on **LangGraph**, powered exclusively by **Google Gemini** (chat + embeddings). Take-home assignment prototype for a fictional boutique advisory firm.

> **Start here → [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md)** — everything you must provide (TL;DR: a free `GOOGLE_API_KEY`, plus `SEC_USER_AGENT` for the RAG phase).

## Quickstart

```bash
make install                 # creates .venv and installs dependencies
cp .env.example .env         # then fill in GOOGLE_API_KEY (and SEC_USER_AGENT) — see .env setup below
make test                    # run the test suite
make notebook                # open the per-phase notebooks (start with phase00)
make ui                      # launch the Streamlit demo UI at localhost:8501
```

### `.env` setup (Gemini-only)

Copy `.env.example` → `.env` and fill in, at minimum:

```dotenv
GOOGLE_API_KEY=your-gemini-key-from-aistudio.google.com/apikey   # the ONLY required key
SEC_USER_AGENT=Your Name your-email@example.com                  # required once you reach RAG (Phase 5)
```

Everything else (`ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `LANGSMITH_API_KEY`) is optional and the system degrades gracefully without it — market data falls back to keyless `yfinance`, and tracing simply doesn't activate. Full details, model-tier rationale, and observed free-tier rate-limit gotchas: [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md) §4–5.

## Project docs

| Doc | What it is |
|---|---|
| [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md) | What you must provide; Gemini model choices; data facts |
| [`docs/data_assumptions.md`](docs/data_assumptions.md) | Documented assumptions about the portfolio data |
| [`docs/architecture.md`](docs/architecture.md) | C4 system/container diagrams, live agent graph, inter-agent protocol, data-flow tables, worked example, integrations |
| [`docs/design_patterns.md`](docs/design_patterns.md) | Every design pattern applied, verified against the actual file/line it lives in |
| [`docs/production_plan.md`](docs/production_plan.md) | Postgres/Chroma Cloud migration, Cloud Run deployment, observability, security/SOC-2, staged rollout, Gemini cost model |
| [`docs/evaluation_report.md`](docs/evaluation_report.md) | Latest Gemini-RAGAS + deterministic eval scores (`make eval` to regenerate) |
| `notebooks/phaseXX_*.ipynb` | Runnable companion notebook per phase — see the table below |

### Phase notebooks

| Phase | Notebook |
|---|---|
| 0 — Scaffold | [`phase00_scaffold.ipynb`](notebooks/phase00_scaffold.ipynb) |
| 1 — Data & Portfolio agent | [`phase01_data_and_portfolio_agent.ipynb`](notebooks/phase01_data_and_portfolio_agent.ipynb) |
| 2 — Supervisor & routing | [`phase02_supervisor_and_routing.ipynb`](notebooks/phase02_supervisor_and_routing.ipynb) |
| 3 — Adapters, registry, resilience | [`phase03_adapters_registry_resilience.ipynb`](notebooks/phase03_adapters_registry_resilience.ipynb) |
| 4 — Securities analysis | [`phase04_securities_analysis.ipynb`](notebooks/phase04_securities_analysis.ipynb) |
| 5 — RAG knowledge base | [`phase05_rag_knowledge_base.ipynb`](notebooks/phase05_rag_knowledge_base.ipynb) |
| 6 — Risk assessment | [`phase06_risk_assessment.ipynb`](notebooks/phase06_risk_assessment.ipynb) |
| 7 — Memory & isolation | [`phase07_memory_and_isolation.ipynb`](notebooks/phase07_memory_and_isolation.ipynb) |
| 8 — Planning & synthesis | [`phase08_planning_and_synthesis.ipynb`](notebooks/phase08_planning_and_synthesis.ipynb) |
| 9 — Guardrails & reflection | [`phase09_guardrails_and_reflection.ipynb`](notebooks/phase09_guardrails_and_reflection.ipynb) |
| 10 — Clarification (HITL) | [`phase10_clarification_hitl.ipynb`](notebooks/phase10_clarification_hitl.ipynb) |
| 11 — Error handling & circuit breaker | [`phase11_error_handling_circuit_breaker.ipynb`](notebooks/phase11_error_handling_circuit_breaker.ipynb) |
| 12 — Optimization (bonus, not built) | [`phase12_optimization.ipynb`](notebooks/phase12_optimization.ipynb) |
| 13 — Evaluation suite | [`phase13_evaluation.ipynb`](notebooks/phase13_evaluation.ipynb) |
| 14 — Streamlit UI | [`phase14_ui_smoke.ipynb`](notebooks/phase14_ui_smoke.ipynb) |
| 15 — Deliverables | [`phase15_deliverables.ipynb`](notebooks/phase15_deliverables.ipynb) |

## Status

- [x] **Phase 0** — scaffold, settings, Gemini factory, logging, exceptions, data in place
- [x] **Phase 1** — data models, Excel repository (union), portfolio tools, Portfolio agent, graph + CLI
- [x] **Phase 2** — supervisor + router (LLM/keyword strategies), Market Research agent, streaming CLI
- [x] **Phase 3** — adapter chain (Finnhub/AlphaVantage/yfinance/SEC EDGAR), tool registry, cached+retry+rate-limited decorators, Gemini rate limiter
- [x] **Phase 4** — Securities Analysis agent: RSI/SMA/EMA/MACD/Bollinger/ATR indicators (Strategy + Factory), technical_analysis + compare_indicators tools
- [x] **Phase 5** — RAG knowledge base: SEC EDGAR ingestion → ChromaDB (Gemini embeddings, 768d), Retriever facade, search_filings with citations + freshness disclosure
- [x] **Phase 6** — Risk Assessment agent: volatility, beta (equity sleeve), VaR (historical/parametric Strategies), concentration, synthetic profiles + tolerance mismatch check
- [x] **Phase 7** — persistent memory (SQLite checkpointer + long-term store), thread-id isolation, access-control interceptor (bonus: session isolation + data privacy)
- [x] **Phase 8** — planner (complexity Strategy + decomposition), supervisor plan-walk, Chain-of-Thought synthesizer with conflict handling
- [x] **Phase 9** — guardrail pipeline (input: PII/injection/scope; output: numeric/citation/groundedness/conflict), reflection revise-loop, safe exit
- [x] **Phase 10** — human-in-the-loop clarification via `interrupt()`/`Command(resume=...)`, CLI prompt/resume flow
- [x] **Phase 11** — per-adapter circuit breakers, Fallback helper (Retriever→keyword search, market adapters→yfinance), global node exception wrapping (no stack trace ever reaches the user)
- [ ] Phase 12 — Optimization Agent (bonus, not yet implemented)
- [x] **Phase 13** — Gemini-powered RAGAS evaluation suite (Faithfulness/ResponseRelevancy/LLMContextPrecision) + deterministic metrics (routing/answer/guardrail accuracy), 75-item hand-crafted dataset, markdown+chart report
- [x] **Phase 14** — Streamlit UI: client selector, streaming turn view, portfolio panel, citations panel, human-in-the-loop clarification buttons
- [x] **Phase 15** — architecture doc (C4 diagrams + live agent graph + inter-agent protocol), design patterns doc, production plan (Postgres/Cloud Run/observability/security/cost model), 4-query demo script, README cross-links
