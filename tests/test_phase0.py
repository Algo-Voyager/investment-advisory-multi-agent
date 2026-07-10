"""Phase 0 smoke tests — scaffold, settings, factory, exceptions, data files.

These run OFFLINE (no Gemini API calls): they verify construction and validation,
not remote behaviour. The live end-to-end check is notebooks/phase00_scaffold.ipynb.
"""

import os
from pathlib import Path

import pytest

# Make settings importable when no .env exists (CI, fresh clones). Only then —
# an env var would OVERRIDE a real .env in pydantic-settings and break live tests.
if not Path(__file__).resolve().parents[1].joinpath(".env").exists():
    os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

from app.config import Settings, settings  # noqa: E402
from app.errors.exceptions import (  # noqa: E402
    ClarificationNeeded,
    CoPilotError,
    HallucinationError,
    RateLimitError,
    RetrievalError,
    ToolError,
)

ROOT = Path(__file__).resolve().parents[1]


class TestSettings:
    def test_defaults_are_gemini_only(self):
        assert settings.LLM_PROVIDER == "gemini"
        assert settings.MODEL_NAME.startswith("gemini")
        assert settings.REASONING_MODEL_NAME.startswith("gemini")
        assert settings.EMBEDDING_MODEL == "gemini-embedding-001"
        assert settings.EMBEDDING_DIM == 768

    def test_non_gemini_provider_rejected(self):
        with pytest.raises(ValueError, match="Gemini-only"):
            Settings(GOOGLE_API_KEY="x", LLM_PROVIDER="openai")

    def test_sec_user_agent_must_contain_email_when_set(self):
        with pytest.raises(ValueError, match="email"):
            Settings(GOOGLE_API_KEY="x", SEC_USER_AGENT="no email here")
        ok = Settings(GOOGLE_API_KEY="x", SEC_USER_AGENT="Jane Doe jane@example.com")
        assert "@" in ok.SEC_USER_AGENT


class TestFactory:
    def test_get_llm_builds_gemini_models(self):
        from app.llm.factory import get_llm

        worker = get_llm()
        reasoner = get_llm(reasoning=True)
        assert settings.MODEL_NAME in worker.model
        assert settings.REASONING_MODEL_NAME in reasoner.model
        assert worker is get_llm()  # lru_cache: same instance back

    def test_get_embeddings_builds(self):
        from app.llm.factory import get_embeddings

        emb = get_embeddings()
        assert settings.EMBEDDING_MODEL in str(emb.model)


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ToolError, CoPilotError)
        assert issubclass(RetrievalError, CoPilotError)
        assert issubclass(HallucinationError, CoPilotError)
        assert issubclass(RateLimitError, ToolError)

    def test_clarification_carries_question_and_options(self):
        exc = ClarificationNeeded("Which tech holdings?", options=["mega-cap", "high-beta"])
        assert exc.question == "Which tech holdings?"
        assert exc.options == ["mega-cap", "high-beta"]


class TestScaffoldAndData:
    def test_folder_layout(self):
        for d in [
            "app/agents", "app/tools", "app/data", "app/integrations", "app/indicators",
            "app/knowledge", "app/graph", "app/memory", "app/guardrails", "app/errors",
            "app/llm", "data/portfolios", "data/profiles", "data/knowledge_base",
            "notebooks", "tests", "evals", "ui", "docs", "scripts",
        ]:
            assert (ROOT / d).is_dir(), f"missing directory: {d}"

    def test_meta_files_exist(self):
        for f in [".env.example", ".gitignore", "README.md", "Makefile", "pyproject.toml",
                  "docs/data_assumptions.md"]:
            assert (ROOT / f).is_file(), f"missing file: {f}"

    def test_gitignore_protects_secrets_and_state(self):
        gi = (ROOT / ".gitignore").read_text()
        for entry in [".env", ".chroma/", ".checkpoints/", ".cache/"]:
            assert entry in gi

    def test_portfolio_data_in_place_with_8_real_clients(self):
        import pandas as pd

        df = pd.read_excel(ROOT / "data/portfolios/portfolios.xlsx", sheet_name="Potfolios")
        real = sorted(df["client_id"].unique())
        assert real == ["CLT-001", "CLT-002", "CLT-003", "CLT-004", "CLT-005",
                        "CLT-007", "CLT-009", "CLT-010"]
        assert "Purchase Price" in df.columns  # the space is real; repository normalizes later

    def test_synthetic_supplement_adds_006_and_008(self):
        import pandas as pd

        path = ROOT / "data/portfolios/synthetic_supplement.xlsx"
        assert path.is_file(), "run: python -m scripts.generate_synthetic_clients"
        df = pd.read_excel(path, sheet_name="Potfolios")
        assert sorted(df["client_id"].unique()) == ["CLT-006", "CLT-008"]
        assert list(df.columns) == ["client_id", "symbol", "security_name", "asset_class",
                                    "quantity", "purchase_date", "Purchase Price", "sector"]
        # each synthetic client mixes stocks, ETFs, and cash — like the real book
        for cid in ["CLT-006", "CLT-008"]:
            sub = df[df.client_id == cid]
            assert (sub.asset_class == "Individual Stock").any()
            assert sub.asset_class.str.contains("ETF").any()
            assert (sub.asset_class == "Cash Equivalent").any()
