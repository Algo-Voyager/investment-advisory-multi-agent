"""Application settings — Singleton pattern via a single module-level instance.

One `Settings` object, loaded once from `.env`, imported everywhere as:

    from app.config import settings

This project is **Gemini-only**: `LLM_PROVIDER` must be "gemini" and the only
required secret is `GOOGLE_API_KEY`. `SEC_USER_AGENT` becomes mandatory when the
RAG phase (Phase 5) starts hitting SEC EDGAR — it is validated for shape here but
only enforced as present by the EDGAR adapter.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- required ---
    GOOGLE_API_KEY: str  # the ONE key the core system needs (Google AI Studio)

    # --- required for Phase 5 (SEC EDGAR); validated for shape if provided ---
    SEC_USER_AGENT: str = ""  # e.g. "Jane Doe jane@example.com" — EDGAR rejects requests without it

    # --- optional fallbacks / tracing ---
    ALPHA_VANTAGE_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""

    # --- Gemini model config (provider is locked) ---
    LLM_PROVIDER: str = "gemini"
    MODEL_NAME: str = "gemini-2.5-flash"           # worker tier: fast, cheap, free-tier eligible
    REASONING_MODEL_NAME: str = "gemini-3.5-flash"  # reasoning tier (NOT Pro — Pro is off the free tier)
    EMBEDDING_MODEL: str = "gemini-embedding-001"   # text-embedding-004 was deprecated Jan 2026
    EMBEDDING_DIM: int = 768                        # never change after Chroma ingestion
    LLM_TEMPERATURE: float = 0.0

    # --- logging / local paths ---
    LOG_LEVEL: str = "INFO"
    CHROMA_PERSIST_DIR: str = ".chroma"
    SQLITE_CHECKPOINT_PATH: str = ".checkpoints/checkpoints.sqlite"

    @field_validator("LLM_PROVIDER")
    @classmethod
    def _gemini_only(cls, v: str) -> str:
        if v.lower() != "gemini":
            raise ValueError(
                "This project is Gemini-only. Set LLM_PROVIDER=gemini "
                "(no OpenAI/Anthropic code paths exist)."
            )
        return v.lower()

    @field_validator("SEC_USER_AGENT")
    @classmethod
    def _sec_user_agent_has_email(cls, v: str) -> str:
        if v and "@" not in v:
            raise ValueError(
                "SEC_USER_AGENT must include a contact email, "
                'e.g. "Jane Doe jane@example.com" — SEC EDGAR requires it.'
            )
        return v


settings = Settings()  # the Singleton — import this, never instantiate Settings() again
