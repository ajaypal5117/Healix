"""Vector store abstraction.

Two backends are supported behind one interface:

* ``faiss``    - local, on-disk, no external service. Default.
* ``pinecone`` - managed serverless index, used for the deployed build.

Switch with the ``VECTOR_BACKEND`` environment variable. Everything downstream
(the retrieval chain, the benchmark) only ever sees a LangChain ``VectorStore``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import settings
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# FAISS
# --------------------------------------------------------------------------- #
def _build_faiss(chunks: Sequence[Document]) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    index_dir = Path(settings.faiss_index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    store = FAISS.from_documents(list(chunks), get_embeddings())
    store.save_local(str(index_dir))
    logger.info(
        "Indexed %d chunks into FAISS at %s in %.1fs",
        len(chunks),
        index_dir,
        time.perf_counter() - started,
    )
    return store


def _load_faiss() -> VectorStore:
    from langchain_community.vectorstores import FAISS

    index_dir = Path(settings.faiss_index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_dir}. Run `python store_index.py` first."
        )
    # The index is produced by this repo, so deserialising its pickle sidecar is safe.
    return FAISS.load_local(
        str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
    )


# --------------------------------------------------------------------------- #
# Pinecone
# --------------------------------------------------------------------------- #
def _ensure_pinecone_index() -> None:
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = {index["name"] for index in client.list_indexes()}
    if settings.pinecone_index_name in existing:
        return

    logger.info("Creating Pinecone index %s", settings.pinecone_index_name)
    client.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    while not client.describe_index(settings.pinecone_index_name).status["ready"]:
        time.sleep(1)


def _build_pinecone(chunks: Sequence[Document]) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore

    _ensure_pinecone_index()
    started = time.perf_counter()
    store = PineconeVectorStore.from_documents(
        documents=list(chunks),
        embedding=get_embeddings(),
        index_name=settings.pinecone_index_name,
    )
    logger.info(
        "Upserted %d chunks to Pinecone index %s in %.1fs",
        len(chunks),
        settings.pinecone_index_name,
        time.perf_counter() - started,
    )
    return store


def _load_pinecone() -> VectorStore:
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_vectorstore(chunks: Sequence[Document]) -> VectorStore:
    """Embed and persist ``chunks`` using the configured backend."""
    if settings.vector_backend == "pinecone":
        return _build_pinecone(chunks)
    return _build_faiss(chunks)


def load_vectorstore() -> VectorStore:
    """Open the already-built index."""
    if settings.vector_backend == "pinecone":
        return _load_pinecone()
    return _load_faiss()


def similarity_search(query: str, k: int | None = None) -> List[Document]:
    """Convenience helper used by the benchmark script."""
    return load_vectorstore().similarity_search(query, k=k or settings.top_k)
