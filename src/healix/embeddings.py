"""Embedding backends, both 384-dimensional.

`minilm` is the real one: sentence-transformers/all-MiniLM-L6-v2, a 6-layer
transformer that runs on CPU and outputs 384-dim vectors. It is the default and
the backend the reported numbers come from.

`hashing` is a deterministic fallback built from numpy alone. It exists because
torch has no wheel on every Python version (3.14 at the time of writing), and
because CI shouldn't download 90 MB of weights to check that chunking works.
Retrieval quality is much worse - it matches on token overlap, not meaning - so
it is a way to exercise the pipeline, not a substitute for the real model.

Both normalise to unit length, so the FAISS inner-product index computes cosine
similarity either way.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

from .config import settings

log = logging.getLogger(__name__)


class HashingEmbeddings:
    """Deterministic bag-of-tokens projection into 384 dimensions."""

    name = "hashing"

    def __init__(self, dim: int):
        self.dim = dim

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.standard_normal(self.dim)

    def embed_one(self, text: str) -> np.ndarray:
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return np.zeros(self.dim, dtype="float32")
        vector = np.sum([self._token_vector(t) for t in set(tokens)], axis=0)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).astype("float32")

    def embed_documents(self, texts):
        return np.vstack([self.embed_one(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self.embed_one(text)


class MiniLMEmbeddings:
    """sentence-transformers/all-MiniLM-L6-v2 via LangChain."""

    name = "minilm"

    def __init__(self, model_name: str, batch_size: int):
        from langchain_huggingface import HuggingFaceEmbeddings

        log.info("loading %s", model_name)
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )

    def embed_documents(self, texts):
        return np.asarray(self._model.embed_documents(list(texts)), dtype="float32")

    def embed_query(self, text):
        return np.asarray(self._model.embed_query(text), dtype="float32")


@lru_cache(maxsize=2)
def get_embeddings(backend: str | None = None):
    """Process-wide singleton. Loading weights takes seconds; do it once."""
    backend = (backend or settings.embedding_backend).lower()

    if backend == "hashing":
        return HashingEmbeddings(settings.embedding_dim)

    if backend != "minilm":
        raise ValueError(f"unknown EMBEDDING_BACKEND '{backend}' (use minilm or hashing)")

    model = MiniLMEmbeddings(settings.embedding_model, settings.embedding_batch_size)
    probe = model.embed_query("dimension probe")
    if len(probe) != settings.embedding_dim:
        raise RuntimeError(
            f"{settings.embedding_model} produced {len(probe)}-dim vectors, "
            f"but EMBEDDING_DIM is {settings.embedding_dim}."
        )
    return model
