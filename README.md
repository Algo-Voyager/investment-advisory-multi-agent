# XZY Capital — Investment Advisory Co-Pilot

A multi-agent investment advisory system built on **LangGraph**, powered exclusively by **Google Gemini** (chat + embeddings). Take-home assignment prototype for a fictional boutique advisory firm.

> **Start here → [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md)** — everything you must provide (TL;DR: a free `GOOGLE_API_KEY`, plus `SEC_USER_AGENT` for the RAG phase).

## Quickstart

```bash
make install                 # creates .venv and installs dependencies
cp .env.example .env         # then fill in GOOGLE_API_KEY (and SEC_USER_AGENT)
make test                    # run the test suite
make notebook                # open the per-phase notebooks (start with phase00)
```

## Project docs

| Doc | What it is |
|---|---|
| [`docs/00_prerequisites_and_setup.md`](docs/00_prerequisites_and_setup.md) | What you must provide; Gemini model choices; data facts |
| [`docs/01_claude_code_prompts.md`](docs/01_claude_code_prompts.md) | Phase-by-phase build prompts (16 phases) |
| [`docs/02_step_by_step_guide.md`](docs/02_step_by_step_guide.md) | Companion explainer for each phase |
| [`docs/data_assumptions.md`](docs/data_assumptions.md) | Documented assumptions about the portfolio data |
| `notebooks/phaseXX_*.ipynb` | Runnable companion notebook per phase |

## Status

- [x] **Phase 0** — scaffold, settings, Gemini factory, logging, exceptions, data in place
- [x] **Phase 1** — data models, Excel repository (union), portfolio tools, Portfolio agent, graph + CLI
- [x] **Phase 2** — supervisor + router (LLM/keyword strategies), Market Research agent, streaming CLI
- [x] **Phase 3** — adapter chain (Finnhub/AlphaVantage/yfinance/SEC EDGAR), tool registry, cached+retry+rate-limited decorators, Gemini rate limiter
- [x] **Phase 4** — Securities Analysis agent: RSI/SMA/EMA/MACD/Bollinger/ATR indicators (Strategy + Factory), technical_analysis + compare_indicators tools
- [ ] Phases 2–15 — see `docs/01_claude_code_prompts.md`
