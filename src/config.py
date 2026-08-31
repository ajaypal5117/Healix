"""Central configuration for Healix.

Every tunable lives here so that the ingestion script, the Flask app and the
benchmark harness all read the exact same values. Nothing else in the codebase
should call ``os.getenv`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    # --- Corpus ------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / _env("DATA_DIR", "data"))

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))

    # --- Embeddings --------------------------------------------------------
    # all-MiniLM-L6-v2 produces 384-dimensional vectors and runs comfortably on CPU.
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # --- Vector store ------------------------------------------------------
    # "faiss" keeps everything on local disk (default, zero external dependencies).
    # "pinecone" uses a managed serverless index.
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "faiss").lower())
    faiss_index_dir: Path = field(
        default_factory=lambda: ROOT_DIR / _env("FAISS_INDEX_DIR", "artifacts/faiss_index")
    )
    pinecone_index_name: str = field(default_factory=lambda: _env("PINECONE_INDEX_NAME", "healix"))
    pinecone_cloud: str = field(default_factory=lambda: _env("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _env("PINECONE_REGION", "us-east-1"))
    pinecone_api_key: str = field(default_factory=lambda: _env("PINECONE_API_KEY", ""))

    # --- Retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # --- Generation --------------------------------------------------------
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 320))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))

    # --- Server ------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    debug: bool = field(default_factory=lambda: _env("FLASK_DEBUG", "0") == "1")
    max_question_chars: int = field(default_factory=lambda: _env_int("MAX_QUESTION_CHARS", 500))

    def validate_for_serving(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if self.vector_backend == "pinecone" and not self.pinecone_api_key:
            raise RuntimeError("VECTOR_BACKEND=pinecone requires PINECONE_API_KEY.")
        if self.vector_backend not in {"faiss", "pinecone"}:
            raise RuntimeError(
                f"Unknown VECTOR_BACKEND '{self.vector_backend}'. Use 'faiss' or 'pinecone'."
            )


settings = Settings()
