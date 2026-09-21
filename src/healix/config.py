"""Configuration, read once from the environment.

Every tunable that affects the corpus, the index or the answer lives here, so
the ingestion script, the benchmark and the Flask app can't drift apart on, say,
chunk size - which would silently invalidate a built index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _s(key, default):
    v = os.getenv(key)
    return default if v is None or not v.strip() else v.strip()


def _i(key, default):
    try:
        return int(_s(key, str(default)))
    except ValueError:
        return default


def _f(key, default):
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Corpus ---
    data_dir: Path = field(default_factory=lambda: ROOT / _s("DATA_DIR", "data"))
    chunk_size: int = field(default_factory=lambda: _i("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _i("CHUNK_OVERLAP", 50))
    min_page_chars: int = field(default_factory=lambda: _i("MIN_PAGE_CHARS", 120))

    # --- Embeddings ---
    # "minilm"  - sentence-transformers/all-MiniLM-L6-v2, 384-dim. The default.
    # "hashing" - deterministic 384-dim fallback, pure numpy. Lets the pipeline
    #             and the tests run where torch has no wheel (e.g. Python 3.14).
    #             Retrieval quality is much worse; it is not a substitute.
    embedding_backend: str = field(default_factory=lambda: _s("EMBEDDING_BACKEND", "minilm").lower())
    embedding_model: str = field(
        default_factory=lambda: _s("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = field(default_factory=lambda: _i("EMBEDDING_DIM", 384))
    embedding_batch_size: int = field(default_factory=lambda: _i("EMBEDDING_BATCH_SIZE", 64))

    # --- Index ---
    index_dir: Path = field(default_factory=lambda: ROOT / _s("INDEX_DIR", "artifacts/index"))
    top_k: int = field(default_factory=lambda: _i("TOP_K", 4))

    # --- Generation ---
    llm_model: str = field(default_factory=lambda: _s("LLM_MODEL", "gpt-4o"))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.2))
    # Three sentences of clinical prose fit comfortably; the cap is a backstop
    # against the model ignoring the length instruction, not the mechanism.
    llm_max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 320))
    llm_timeout: int = field(default_factory=lambda: _i("LLM_TIMEOUT", 30))
    openai_api_key: str = field(default_factory=lambda: _s("OPENAI_API_KEY", ""))

    # --- Server ---
    host: str = field(default_factory=lambda: _s("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _i("PORT", 8080))
    debug: bool = field(default_factory=lambda: _s("FLASK_DEBUG", "0") == "1")
    max_question_chars: int = field(default_factory=lambda: _i("MAX_QUESTION_CHARS", 500))

    def require_api_key(self):
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "Retrieval and indexing work without it; only generation needs it."
            )
        return self.openai_api_key


settings = Settings()
