"""FAISS vector index: build, persist, search.

`IndexFlatIP` over unit-normalised vectors is exact cosine search. At this
corpus size (a few thousand chunks) an approximate index would add recall risk
and tuning burden to save microseconds on a search that already takes well
under a millisecond - the round trip is dominated by embedding the query, not
by the search itself.

The index and its metadata are saved side by side, with a manifest recording
the settings that produced them. Loading an index built with a different chunk
size or embedding backend would silently return nonsense, so the manifest is
checked on load.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings
from .corpus import Chunk
from .embeddings import get_embeddings

log = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class Hit:
    text: str
    page: int
    chunk_id: int
    source: str
    score: float


class VectorIndex:
    def __init__(self, index, chunks, manifest):
        self._index = index
        self._chunks = chunks
        self.manifest = manifest

    def __len__(self):
        return len(self._chunks)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        vector = get_embeddings(self.manifest["embedding_backend"]).embed_query(query)
        vector = np.asarray([vector], dtype="float32")
        scores, positions = self._index.search(vector, min(k, len(self._chunks)))

        hits = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            record = self._chunks[position]
            hits.append(Hit(record["text"], record["page"], record["chunk_id"],
                            record["source"], float(score)))
        return hits


def build(chunks: list[Chunk], index_dir: Path | None = None) -> dict:
    """Embed and persist. Returns timing and size figures for the run report."""
    import faiss

    directory = Path(index_dir or settings.index_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model = get_embeddings()
    texts = [c.text for c in chunks]

    started = time.perf_counter()
    vectors = model.embed_documents(texts)
    embed_seconds = time.perf_counter() - started

    if vectors.shape[1] != settings.embedding_dim:
        raise RuntimeError(f"expected {settings.embedding_dim}-dim vectors, "
                           f"got {vectors.shape[1]}")

    index_started = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(directory / INDEX_FILE))
    index_seconds = time.perf_counter() - index_started

    with open(directory / META_FILE, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps({"text": chunk.text, **chunk.to_metadata()}) + "\n")

    total = time.perf_counter() - started
    manifest = {
        "chunks": len(chunks),
        "dimensions": int(vectors.shape[1]),
        "embedding_backend": model.name,
        "embedding_model": settings.embedding_model if model.name == "minilm" else "n/a",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embed_seconds": round(embed_seconds, 2),
        "index_seconds": round(index_seconds, 3),
        "total_seconds": round(total, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    log.info("indexed %d chunks in %.1fs (%.1f min)", len(chunks), total, total / 60)
    return manifest


def load(index_dir: Path | None = None) -> VectorIndex:
    import faiss

    directory = Path(index_dir or settings.index_dir)
    if not (directory / INDEX_FILE).exists():
        raise FileNotFoundError(
            f"No index at {directory}. Run `python scripts/build_index.py` first."
        )

    manifest = json.loads((directory / MANIFEST_FILE).read_text())

    if manifest["chunk_size"] != settings.chunk_size:
        raise RuntimeError(
            f"index was built with chunk_size={manifest['chunk_size']} but the "
            f"current setting is {settings.chunk_size}. Rebuild it."
        )
    if manifest["embedding_backend"] != settings.embedding_backend:
        raise RuntimeError(
            f"index was built with the '{manifest['embedding_backend']}' embedding "
            f"backend but EMBEDDING_BACKEND is '{settings.embedding_backend}'. "
            "Vectors from different backends are not comparable. Rebuild it."
        )

    index = faiss.read_index(str(directory / INDEX_FILE))
    chunks = [json.loads(line) for line in
              (directory / META_FILE).read_text(encoding="utf-8").splitlines() if line]
    return VectorIndex(index, chunks, manifest)
