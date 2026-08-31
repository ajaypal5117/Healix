"""Embedding model factory.

``all-MiniLM-L6-v2`` is a 6-layer sentence-transformer producing 384-dimensional
vectors. It is small enough to load inside a container without a GPU while still
scoring well on retrieval benchmarks, which is why it is the default here.

The heavy import lives inside the function on purpose: importing
``langchain_huggingface`` pulls in torch and sentence-transformers, which costs
several seconds and a lot of memory. Only the paths that actually embed text
should pay that, so tests and the health check stay fast.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Return a process-wide singleton embedding model.

    The weights take a few seconds to load, so caching matters: a Flask worker
    loads them once at boot rather than once per request.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    probe_dim = len(embeddings.embed_query("dimension probe"))
    if probe_dim != settings.embedding_dim:
        raise RuntimeError(
            f"{settings.embedding_model} returns {probe_dim}-dim vectors but "
            f"EMBEDDING_DIM is set to {settings.embedding_dim}."
        )
    return embeddings
