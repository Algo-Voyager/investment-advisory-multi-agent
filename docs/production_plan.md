# Production Plan

This prototype runs as a single Python process with SQLite/Chroma-on-disk state, no auth beyond a CLI flag / UI dropdown, and manual `make ui` launches. This document is the path from that prototype to a production deployment, grounded in where the current code already draws the seams that make each step additive rather than a rewrite.

## 1. Persistence: SQLite → Postgres

Two SQLite databases exist today, both already behind an interface swap point:

| Today | File | Interface | Production swap |
|---|---|---|---|
| Short-term memory (checkpoints) | `.checkpoints/checkpoints.sqlite` | `get_checkpointer()` (`app/memory/checkpointer.py:21`), returns a `SqliteSaver` | `AsyncPostgresSaver` (`langgraph.checkpoint.postgres.aio`) — same factory signature, callers (`GraphBuilder.with_memory()`) never change |
| Long-term memory (decisions/preferences) | `.checkpoints/memory.sqlite` | `MemoryStore` facade (`app/memory/store.py:45`) — `save_decision`/`get_recent_decisions`/`save_preference` | Re-point `MemoryStore.__init__`'s `sqlite3.connect` at a Postgres connection (or reimplement the same 4 methods over `asyncpg`); the facade's public surface is already domain-vocabulary, not SQL-shaped, so callers (the graph nodes) are untouched |

Migration mechanics: `AsyncSqliteSaver`/`SqliteSaver` and `AsyncPostgresSaver` implement the same LangGraph `BaseCheckpointSaver` interface — `GraphBuilder.with_memory(checkpointer=...)` (`app/graph/builder.py:87`) already accepts an injected checkpointer (built for exactly this reason during Phase 14's UI work, which needed an async variant). Postgres also unlocks the thing SQLite structurally can't do: concurrent writers across multiple app instances behind a load balancer. Row-level isolation is already enforced by the existing `thread_id = f"{client_id}-{session_id}"` scheme and `MemoryStore`'s `client_id`-keyed rows — nothing about the isolation *design* changes, only the storage engine.

**Steps:** stand up managed Postgres (Cloud SQL / RDS) → run schema migration (LangGraph ships a Postgres schema setup; `MemoryStore`'s `_SCHEMA` DDL is trivially portable) → implement `PostgresMemoryStore(MemoryStore)` → flip `get_checkpointer()` and `get_memory_store()` behind an env flag → dual-write during a burn-in window → cut over.

## 2. Vector Store: local Chroma → Chroma Cloud / Qdrant

`VectorStore` (`app/knowledge/vector_store.py:29`) already isolates every Chroma-specific call behind its own class; nothing else in the codebase imports `chromadb` directly. Today it's a `chromadb.PersistentClient` pointed at `.chroma/` (single-process, single-disk). Production options, in order of migration effort:

- **Chroma Cloud** (hosted Chroma) — smallest diff: swap `PersistentClient(path=...)` for `CloudClient(...)`; the collection/query/embedding-injection API is unchanged, so `VectorStore`'s own method bodies barely move.
- **Qdrant** (self-hosted or Cloud) — more work but better production fit (horizontal scaling, snapshotting, richer filtering than Chroma's `where` clauses): reimplement `VectorStore`'s `query`/`upsert` methods against `qdrant-client`, same external interface (`Retriever` — `app/knowledge/retriever.py:23` — never learns which backend answered, by design).

Either way, the embedding model stays pinned to `gemini-embedding-001` at 768 dims (`settings.EMBEDDING_DIM`) — **never** change this post-ingestion without a full re-embed, since it silently breaks retrieval (documented today as a `TODO` in `llm/factory.py`). The keyword-search fallback (`app/knowledge/keyword_fallback.py`, wired via `Fallback` in `retriever.py:46`) stays as-is regardless of vector backend — it's the resilience floor, not a Chroma-specific hack.

## 3. Containerize + Deploy on Cloud Run

```
Dockerfile (multi-stage):
  builder: python:3.11-slim → pip install -r requirements.txt into a venv
  runtime: python:3.11-slim, copy venv, copy app/, ui/, data/ (or mount data/profiles + portfolios via a volume/GCS bucket for real deployments — don't bake client data into the image)
  CMD: streamlit run ui/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

- **Cloud Run** (serverless containers) fits this workload well: request-driven (an advisor session isn't constant traffic), scales to zero between demos, and each Gemini call is already rate-limited client-side (`GEMINI_CALLS_PER_MINUTE`), so burst concurrency is naturally capped.
- **Behind a gateway** (Cloud Run's built-in ingress, or API Gateway / a small FastAPI shim in front if the CLI's request/response shape needs to be exposed as a REST API for a real frontend instead of Streamlit): terminate TLS, enforce authN (see §5) before a request ever reaches the graph.
- **Stateful pieces move out of the container:** Postgres (§1) and the vector store (§2) become managed services the Cloud Run instance connects to over the network — the container itself becomes stateless and horizontally scalable, which SQLite-on-local-disk structurally prevents today (two Cloud Run instances can't safely share one SQLite file).
- **Secrets:** `GOOGLE_API_KEY`, `FINNHUB_API_KEY`, etc. move from `.env` to Secret Manager, injected as env vars at deploy time — `app/config.py`'s `Settings` class needs zero changes, since it already reads from the process environment via `pydantic-settings`.

## 4. Observability: LangSmith + OpenTelemetry + Grafana

- **LangSmith** is already wired (`app/config.py` activates `LANGCHAIN_TRACING_V2` whenever `LANGSMITH_API_KEY` is present) and gives per-run traces of every node, tool call, and LLM call for free — the fastest way to debug "why did this answer come out wrong" during rollout. Keep it in production, scoped to a dedicated `LANGCHAIN_PROJECT` per environment (`xzy-investment-advisory-copilot-prod` vs `-staging`).
- **OpenTelemetry** for the infrastructure layer LangSmith doesn't cover: request latency at the gateway, Cloud Run cold starts, Postgres/vector-store query latency, circuit-breaker state transitions (`app/errors/circuit_breaker.py` already emits structured `circuit_opened`/`circuit_closed`/`circuit_fail_fast` log events — an OTel exporter on the existing `structlog` pipeline (`app/logging.py`) is a low-effort bridge, not a rewrite).
- **Grafana** dashboards over the OTel/Prometheus pipeline: p50/p95 turn latency (the eval harness already computes this — `evals/report.py`'s latency stats become the SLO baseline), adapter availability (`circuit_breakers.all_states()` — `app/errors/circuit_breaker.py:121` — is already a ready-made health-check payload), Gemini quota consumption by model tier (worker vs reasoning — critical given the observed daily-cap reality documented in `00_prerequisites_and_setup.md`), and guardrail block/revise rates per `state["guardrail_events"]`.
- **Alerting:** page on sustained circuit-breaker OPEN state for a keyed adapter (Finnhub/Alpha Vantage), on Gemini 429 rate exceeding a threshold (quota exhaustion risk — this bit the team during development, see memory), and on `safe_exit` rate spiking (signals a systemic guardrail or upstream failure, not isolated bad luck).

## 5. Security

- **Per-client encryption:** encrypt `decisions`/`preferences` rows (long-term memory) and checkpoint payloads at rest — Postgres transparent data encryption (cloud-managed) covers the disk; for defense-in-depth, encrypt the `query`/`answer` columns with a per-tenant (per-`client_id`) key via envelope encryption (KMS-wrapped DEKs), so a single leaked key doesn't expose every client's advisory history at once.
- **Access control today → tomorrow:** `app/guardrails/access_control.py`'s `ContextVar`-bound `verify_client_access` (interceptor pattern, enforced inside `ExcelPortfolioRepository.get()` itself) is a solid *authorization* boundary but currently trusts whatever `client_id` the CLI flag / UI dropdown supplies — there's no *authentication* yet. Production needs a real identity layer (advisor SSO via OIDC, or client portal auth) that maps an authenticated principal to the `client_id`(s) they may bind via `set_session_client()`; the interceptor's enforcement logic doesn't change, only what feeds it.
- **Audit log:** `state["guardrail_events"]` and `state["tool_results"]` already capture a structured, per-turn record of every guard decision and every tool call with its raw output — this is 80% of an audit trail already. Production adds: (1) durable, append-only storage of this record (not just SQLite `decisions.answer` — the *evidence*, not just the final text), (2) a record of *who* accessed *which client's* data and *when* (log every `verify_client_access` call, pass or deny, not just denials), (3) retention policy aligned to the firm's compliance requirements.
- **SOC-2 checklist (illustrative, not exhaustive):**
  - [ ] Encryption in transit (TLS everywhere — gateway, DB connections, vector-store API) and at rest (see above)
  - [ ] Least-privilege IAM for the Cloud Run service account (scoped Secret Manager access, scoped Cloud SQL/Postgres roles — no broad project-editor grants)
  - [ ] Audit logging with tamper-evidence (append-only, exported to a separate log sink the app itself can't write over)
  - [ ] Documented incident-response plan (who's paged on a data-boundary violation — i.e. a `PermissionError` from `verify_client_access` firing in prod is a security event, not just a log line)
  - [ ] Vendor risk assessment for Gemini, Finnhub, Alpha Vantage, SEC EDGAR (data processed, retention, sub-processor agreements)
  - [ ] Access reviews (who can query `decisions`/`preferences` directly, bypassing the app layer — should be nobody outside a break-glass procedure)
  - [ ] Change management (this rollout process itself — §6 — is the change-management control for the app layer)
  - [ ] Annual penetration test covering the prompt-injection surface specifically (the existing `PromptInjectionGuard`/`ScopeGuard` regex patterns are a first line, not a complete defense — a real pentest should attempt prompt-injection-driven cross-client access against the interceptor, not just against the regex)

## 6. Staged Rollout — Phase 13 Evals as the Gate

The eval suite (`evals/`) already produces exactly the artifact a staged rollout needs: `docs/evaluation_report.md` with per-dataset scores (`answer_accuracy`, `routing_accuracy`, RAGAS `faithfulness`/`answer_relevancy`/`context_precision`, `guardrail_effectiveness`) and latency percentiles.

**Proposed rollout stages:**

1. **Shadow** — new build runs alongside the current production build on 100% of *replayed* traffic (real queries, no user-visible response), scored against the same 75-item dataset (`evals/datasets/*.jsonl`) plus a growing corpus of real anonymized production queries. Gate: no regression vs. the previous build's scores on any metric beyond a defined tolerance (e.g. faithfulness ≥ 0.65, guardrail_effectiveness ≥ 0.95 — current observed baselines from `docs/evaluation_report.md`).
2. **Canary (5%)** — real traffic, small slice, same eval gate re-run nightly against fresh production query logs (privacy-scrubbed) in addition to the static dataset.
3. **Progressive (25% → 50% → 100%)** — each step gated on the eval suite passing AND on the Grafana dashboards (§4) showing no latency/error-rate regression.
4. **Kill switch:** a single config flag (`ENABLE_CLARIFICATION`, `ENABLE_LLM_GUARDRAILS`, and a new `ACTIVE_GRAPH_VERSION` routing flag) lets ops revert to the last known-good build instantly without a redeploy — the same pattern already used to force-disable clarification for eval runs (`settings.ENABLE_CLARIFICATION=False`) extends naturally to a build-level rollback switch. Trigger conditions: eval regression caught post-deploy, `safe_exit` rate spike (§4 alerting), or a manual advisor-reported bad-answer pattern.

This makes the eval suite a **release gate**, not just a development-time report — exactly the role Phase 13 was built to fill.

## 7. Cost Model — Gemini Token Pricing

> **Illustrative, not a quote.** Gemini per-token pricing changes over time and by tier/region; the rates below are placeholder order-of-magnitude figures for Flash-class models to demonstrate the *shape* of the calculation. Before committing to a budget, pull current rates from `https://ai.google.dev/gemini-api/docs/pricing` and substitute them into this same formula.

**Observed call volume per turn** (from live testing, `docs/00_prerequisites_and_setup.md` §5 and this project's own traces): a multi-agent query costs **4–8 LLM calls**, split across two tiers:

| Tier | Model | Used by | Typical calls/turn | Assumed avg tokens/call (in / out) |
|---|---|---|---|---|
| Worker | `MODEL_NAME` (`gemini-3.1-flash-lite` class) | routing (simple mode), each specialist's ReAct loop, clarifier's ambiguity check | ~5 | 600 in / 200 out |
| Reasoning | `REASONING_MODEL_NAME` (`gemini-3.5-flash` class) | planner decomposition, synthesizer, reflector's 2 LLM guards | ~3 | 1,800 in / 350 out |

Illustrative Flash-tier rates used below (USD per 1M tokens): **worker** $0.10 in / $0.40 out; **reasoning** $0.30 in / $2.50 out.

**Per-query cost:**
```
worker:    5 × (600 in, 200 out)   = 3,000 in + 1,000 out tokens
           = 3,000×$0.10/1e6 + 1,000×$0.40/1e6  ≈ $0.0007
reasoning: 3 × (1,800 in, 350 out) = 5,400 in + 1,050 out tokens
           = 5,400×$0.30/1e6 + 1,050×$2.50/1e6  ≈ $0.0042
embeddings (query only, ~50 tokens):              ≈ $0.00001
────────────────────────────────────────────────────────────
≈ $0.005 per query  (roughly half a cent)
```

**Per active client, 3 sessions/week, ~3 queries/session:**
```
9 queries/week × $0.005/query ≈ $0.045/week
× 4.33 weeks/month            ≈ $0.20/month per active client
```

**At scale** (illustrative): 500 active clients → **≈ $100/month** in Gemini token spend; 5,000 active clients → **≈ $1,000/month**. Even with a 3–5× margin of error against real current pricing and real query complexity (some turns will run more tool-calling rounds than the 5/3 assumed here, and RAG-heavy queries add embedding costs proportional to filing corpus size at ingestion time, which is a one-time cost not modeled per-query above), the LLM line item is not the dominant infrastructure cost at this usage pattern — Postgres, the vector store, and Cloud Run compute are likely comparable or larger. The bigger practical constraint observed during this project's own development was **daily per-model rate limits on the free tier** (as low as 20 requests/day observed on `gemini-2.5-flash`), not dollar cost — production deployment requires billing enabled on the Gemini API key specifically to lift those caps, independent of the dollar amounts above being small.
