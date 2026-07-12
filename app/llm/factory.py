"""LLM & embeddings factory — Factory pattern, Gemini-only.

Every component gets its model from here; nothing else in the codebase constructs
a chat model or embedding client. Two chat tiers, one embedding model:

    get_llm()                → ChatGoogleGenerativeAI(settings.MODEL_NAME)            # worker
    get_llm(reasoning=True)  → ChatGoogleGenerativeAI(settings.REASONING_MODEL_NAME)  # reasoning
    get_embeddings()         → Gemini embeddings pinned to settings.EMBEDDING_DIM

There is deliberately NO OpenAI/Anthropic code path — `Settings` already rejects
any `LLM_PROVIDER` other than "gemini", and we re-assert it here (belt and braces).
"""

import time
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings
from app.errors.exceptions import RateLimitError
from app.logging import get_logger

log = get_logger(__name__)


class GeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """Gemini embeddings pinned to a fixed output dimension.

    gemini-embedding-001 emits 3072 dims by default; we truncate to
    settings.EMBEDDING_DIM (768) and must NEVER change it after Chroma ingestion,
    or stored vectors and query vectors stop being comparable.
    TODO(phase5): normalize truncated vectors (Google recommends it for dims < 3072).
    """

    # Free-tier reality (observed): ~100 embed-content requests/min. Bulk ingestion
    # must therefore sub-batch and back off on 429 instead of dying mid-filing.
    _BATCH = 50
    _MAX_RETRIES = 6

    def embed_documents(self, texts, **kwargs):  # type: ignore[override]
        kwargs.setdefault("output_dimensionality", settings.EMBEDDING_DIM)
        vectors: list = []
        for start in range(0, len(texts), self._BATCH):
            batch = texts[start:start + self._BATCH]
            for attempt in range(1, self._MAX_RETRIES + 1):
                try:
                    vectors.extend(super().embed_documents(batch, **kwargs))
                    break
                except Exception as exc:
                    if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                        raise
                    if attempt == self._MAX_RETRIES:
                        raise RateLimitError(
                            f"Gemini embedding quota still exhausted after "
                            f"{self._MAX_RETRIES} backoffs") from exc
                    delay = min(15 * attempt, 60)
                    log.warning("embedding_backoff", batch_start=start,
                                attempt=attempt, sleep_s=delay)
                    time.sleep(delay)
        return vectors

    def embed_query(self, text, **kwargs):  # type: ignore[override]
        kwargs.setdefault("output_dimensionality", settings.EMBEDDING_DIM)
        for attempt in range(1, 4):
            try:
                return super().embed_query(text, **kwargs)
            except Exception as exc:
                if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc) or attempt == 3:
                    raise
                log.warning("embedding_query_backoff", attempt=attempt)
                time.sleep(10 * attempt)


def _assert_gemini() -> None:
    if settings.LLM_PROVIDER != "gemini":  # unreachable if Settings validated, kept as a guard
        raise RuntimeError("This project is Gemini-only. Set LLM_PROVIDER=gemini.")


@lru_cache(maxsize=2)
def get_llm(reasoning: bool = False) -> ChatGoogleGenerativeAI:
    """Return the shared chat model. `reasoning=True` selects the heavier tier
    (planner / synthesizer / reflector / LLM-as-judge)."""
    _assert_gemini()
    model = settings.REASONING_MODEL_NAME if reasoning else settings.MODEL_NAME
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=settings.LLM_TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
        # Rate-limit Gemini itself (Phase 3): the free tier is ~10 req/min — LangChain
        # throttles BEFORE sending, so we queue briefly instead of eating a 429.
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=settings.GEMINI_CALLS_PER_MINUTE / 60,
            check_every_n_seconds=0.1,
        ),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GeminiEmbeddings:
    """Return the shared Gemini embedding client (used by Chroma + RAGAS)."""
    _assert_gemini()
    return GeminiEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
