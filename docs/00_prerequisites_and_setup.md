# XZY Investment Advisory Co-Pilot — Prerequisites & Setup (READ THIS FIRST)

This is the **single source of truth for everything YOU (the developer) must provide** before and during the build. Both `01_claude_code_prompts.md` and `02_step_by_step_guide.md` refer back to this file. Skim it once now, fill in your `.env`, then start Phase 0.

> **LLM policy for this project: Google Gemini ONLY.** There is no OpenAI or Anthropic anywhere in this build — not for chat, not for embeddings, not for evaluation. Every place that reasons or embeds text uses the Gemini API via `langchain-google-genai`. If you ever see `ChatOpenAI`, `ChatAnthropic`, `OpenAIEmbeddings`, or an `OPENAI_API_KEY` requirement creep in, that's a bug — remove it.

---

## 1. The ONE thing you truly must provide

| # | What you provide | Where to get it | Required? | Used for |
|---|---|---|---|---|
| 1 | **`GOOGLE_API_KEY`** (a Gemini API key) | [Google AI Studio](https://aistudio.google.com/apikey) → "Create API key" (free) | **REQUIRED** | Every LLM call **and** every embedding (chat, routing, planning, synthesis, reflection, LLM-as-judge, RAG embeddings, eval metrics) |
| 2 | **`SEC_USER_AGENT`** — your name + email, e.g. `"iampkumar iampkumar03@gmail.com"` | Just your own email. No signup. | **REQUIRED for RAG (Phase 5)** | SEC EDGAR **rejects requests without a descriptive `User-Agent` containing contact info.** No key, just an email string. |

**With only those two values the entire core system runs.** Market prices and news come from `yfinance`, which needs **no key at all**. Everything below is optional.

---

## 2. Optional things (only if you want the "resilience / fallback" story)

These make the *design-patterns* story richer (Adapter fallback, Circuit Breaker) but are **not needed to demo**. Every one of them has a free tier. If a key is absent, the system must degrade gracefully to `yfinance` / keyword search — never crash.

| What | Where | Free tier | Used for |
|---|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) | 25 req/day, 5 req/min | Secondary quote/fundamentals adapter (Phase 3), fallback demo (Phase 11) |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/) | 60 req/min | More reliable **news** than yfinance (Phase 3) — recommended if you can |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) | free personal | Optional tracing during evals (Phase 13). Provider-agnostic — works fine with Gemini. |

> **Recommendation:** grab a **Finnhub** key. yfinance's `.news` and `.info` are flaky and frequently return empty — Finnhub makes the news demos far more reliable. Everything else you can skip.

---

## 3. Software you need on your machine

- **Python 3.11+** (`python --version`).
- **pip / venv** (or `uv`, if you prefer — faster).
- **Jupyter** to open the per-phase notebooks: `pip install jupyterlab` (or use the VS Code notebook UI). Each phase ships a `notebooks/phaseXX_*.ipynb` you run to verify and visualize that phase.
- **Graphviz (optional, for nicer graph pictures):** `graph.get_graph().draw_mermaid()` (text) always works with no system deps. `draw_mermaid_png()` needs network access or a local renderer; the notebooks fall back to Mermaid text if PNG rendering isn't available, so you're never blocked.
- Internet access (yfinance, SEC EDGAR, Gemini API).

---

## 4. Your `.env` file (copy `.env.example` → `.env` and fill in)

```dotenv
# ---- REQUIRED ----
GOOGLE_API_KEY=your-gemini-key-from-aistudio          # the only key the core system needs
SEC_USER_AGENT=Your Name your-email@example.com       # required by SEC EDGAR for Phase 5 RAG

# ---- Gemini model + provider config (do not change provider) ----
LLM_PROVIDER=gemini                                   # locked to gemini for this project
MODEL_NAME=gemini-2.5-flash                            # workhorse: fast + cheap, free-tier eligible, used for most agents/tools
REASONING_MODEL_NAME=gemini-3.5-flash                  # frontier GA model for planning, synthesis, reflection, LLM-judge (free-tier eligible; NOTE: Pro models are NOT free anymore)
EMBEDDING_MODEL=gemini-embedding-001                   # current stable Gemini embedding (text-embedding-004 was DEPRECATED Jan 2026 — do not use it)
EMBEDDING_DIM=768                                      # gemini-embedding-001 defaults to 3072 dims; truncate to 768. NEVER change after Chroma ingestion or retrieval breaks
LLM_TEMPERATURE=0

# ---- OPTIONAL (fallbacks / tracing; leave blank to skip) ----
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false

# ---- Local paths / logging (sensible defaults; rarely change) ----
LOG_LEVEL=INFO
CHROMA_PERSIST_DIR=.chroma
SQLITE_CHECKPOINT_PATH=.checkpoints/checkpoints.sqlite
```

> **Never commit `.env`.** `.gitignore` must include `.env`, `.chroma/`, `.checkpoints/`, and `.cache/`. Read every secret from `settings` (Phase 0's Pydantic Settings). No key ever appears in code.

---

## 5. Which Gemini model to use where (and why)

One provider, two model tiers, one embedding model — selected by `settings`, never hard-coded:

| Role | Model (env var) | Why |
|---|---|---|
| Routing, keyword-ish classification, simple tool-driven agents | `MODEL_NAME` = `gemini-2.5-flash` | Cheap + fast + free-tier eligible; most calls are structured/tool calls that don't need deep reasoning. (`gemini-3.1-flash-lite` is an even cheaper GA alternative.) |
| Planner, Synthesizer, Reflector, LLM-as-judge guardrails, RAGAS judge | `REASONING_MODEL_NAME` = `gemini-3.5-flash` | Current frontier GA model, strong multi-step reasoning, **still free-tier eligible**. ⚠️ Do NOT default to a Pro model (`gemini-2.5-pro`, `gemini-3.1-pro-preview`): as of 2026 **Pro models are excluded from the free tier** — they require billing. If you have billing enabled, feel free to point this var at a Pro model. |
| All embeddings (Chroma ingestion + query, RAGAS context metrics) | `EMBEDDING_MODEL` = `gemini-embedding-001` | The current stable Gemini text-embedding model. ⚠️ **`text-embedding-004` was deprecated Jan 14, 2026 — don't use it.** Default output is 3072-dim; pass `output_dimensionality=768` (from `EMBEDDING_DIM`) and keep it fixed — changing dimensions after Chroma ingestion silently breaks retrieval. |

**Gemini free-tier rate limits are real — and harsher than the docs suggest.** Observed on a fresh key during this build (July 2026): `gemini-2.5-flash` allows only **20 requests/DAY** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) and `gemini-2.5-flash-lite` is **closed to new users entirely** (404). The 3-series buckets (`gemini-3.1-flash-lite`, `gemini-3.5-flash`) are healthier — which is why the worker default is `gemini-3.1-flash-lite`. Practical implications:
- One multi-agent query costs **4–8 LLM calls** (router + 2 calls per agent). 20/day ≈ 3 queries. Budget accordingly.
- Daily quotas are **per model** — spreading worker vs reasoning tiers across two models doubles your headroom.
- This is exactly why Phase 3's `@cached` + `@retry` + `@rate_limited` decorators matter, and why the router has a free `KeywordRoutingStrategy` fallback (it engaged automatically during a real 429 in testing).
- **Before the live panel demo: enable billing on the key.** Flash-tier tokens cost cents; a quota-dead demo costs the interview.

> **RAGAS gotcha (Phase 13):** RAGAS defaults to OpenAI. You **must** explicitly pass a Gemini `LangchainLLMWrapper(ChatGoogleGenerativeAI(...))` and `LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(...))` to every RAGAS metric, or it will demand an `OPENAI_API_KEY`. Phase 13's prompt already tells Claude Code to do this — just be aware.

---

## 6. Data you're given vs. data you (or Claude Code) must create

| Item | Status | Action |
|---|---|---|
| `data/portfolios/portfolios.xlsx` (sheet `"Potfolios"` — sic) | **Given.** Contains **8 real clients**: CLT-001, 002, 003, 004, 005, 007, 009, 010. | Read as-is. Do **not** renumber the `CLT-XXX` IDs. |
| Clients CLT-006 & CLT-008 | **Missing** (brief says "10 clients", file has 8). | **Decide & document.** Default plan: generate 2 clearly-labelled *synthetic* clients so you have 10. Acceptable alternative: use the real 8 and note the discrepancy. Either way, **state your choice in the design doc** — silent invention looks like a data error. |
| Client **risk profiles** (risk tolerance, goals, time horizon, income needs) | **Not in the source at all** — the file has only holdings. | You/Claude Code create synthetic `data/profiles/CLT-XXX.json`. **This is a documented assumption**, not real data. Vary them so ≥2 clients have a genuine profile-vs-portfolio mismatch for the Risk demo. |
| Market prices, news, SEC filings | **You fetch** at runtime (yfinance, Finnhub, SEC EDGAR). | Handled by the tools/adapters. |

**Known data facts you'll rely on (verified against the real file):**
- **Cash rows:** `symbol = "CASH"`, and the **`quantity` column IS the dollar balance** (e.g. CLT-001 CASH quantity = 160050 → $160,050). The `Purchase Price` (~$50) on cash rows is noise — **do not** compute cost basis or returns from it. Value cash at `quantity × 1`; cost basis = quantity; return = 0.
- **`asset_class` is rich free text** (16+ values: `Individual Stock`, `Equity ETF`, `Bond ETF`, `Municipal Bond ETF`, `International Equity ETF`, `Growth ETF`, `ESG ETF`, `Clean Energy ETF`, `Cash Equivalent`, …). Keep it a plain `str`, never an enum. `is_cash = (asset_class == "Cash Equivalent")`; `is_etf = ("ETF" in asset_class)`; everything else is an individual security.
- **Fund-only clients (no individual stocks):** **CLT-001, CLT-003, CLT-004** hold only ETFs + cash. Single-stock technical analysis and SEC 10-K/10-Q retrieval are **not applicable** to them — pick a different client for those demos (see below).
- **Individual-stock holders (use these for technical analysis / SEC filings):** CLT-002 (NVDA, MSFT, TSLA), CLT-005 (AAPL, MSFT, GOOGL, NVDA, META, AMZN, …), CLT-007 (financials/energy), CLT-009 (speculative tech), CLT-010 (mixed).
- **NVDA is held by CLT-002 and CLT-005 only.** Any "technical analysis of my NVDA position" demo must target one of those — **not** CLT-003 (which holds no NVDA and no individual stocks).

---

## 7. The per-phase notebooks

Every phase produces a Jupyter notebook under `notebooks/`, e.g. `notebooks/phase01_data_and_portfolio_agent.ipynb`. Each notebook is a **runnable, visual companion** to that phase:

- a setup cell that loads `.env` and imports the phase's modules,
- cells that exercise the new capability against a *real* client from the data,
- the phase's **acceptance check** as an executable cell,
- and, for any phase that touches the graph, a **graph visualization** cell (`graph.get_graph().draw_mermaid()` / `draw_mermaid_png()`).

Run the notebook top-to-bottom after Claude Code finishes a phase. If every cell runs clean, the phase passes. Scaffolds for all 16 notebooks are generated up front (see `notebooks/`); Claude Code fleshes them out as it builds each phase.

---

## 8. Scope guidance — what to build first if time is short

The brief marks **Optimization, Session isolation, and Data privacy as bonus/optional.** Don't spread thin. Suggested cut line:

- **Core (must ship):** Phase 0–2 (scaffold, Portfolio agent, Supervisor + Market Research), Phase 4 (Securities/technical), Phase 5 (RAG), Phase 6 (Risk), Phase 9 (hallucination guardrails — *required*), Phase 10 (clarification — *required*), Phase 13 (eval report — *required*), Phase 14 (UI — *required*).
- **Bonus (build only if core is solid):** Phase 7 (memory/session isolation + data privacy), Phase 12 (optimization).
- **Infra you can describe instead of fully building if crunched:** Phase 3's five adapters and Phase 11's circuit breaker — a simpler cached+retry on yfinance is enough to *demo*; write up the full Adapter/Circuit-Breaker design in `docs/production_plan.md`.

Now open `02_step_by_step_guide.md` for Phase 0's explanation, then paste Phase 0 from `01_claude_code_prompts.md`.
