"""Index build, persistence and the guards against loading a stale index."""

import dataclasses

import pytest

from healix import index
from healix.config import settings


def test_index_has_one_vector_per_chunk(built_index, chunks):
    assert len(built_index) == len(chunks)


def test_vectors_are_384_dimensional(built_index):
    assert built_index.manifest["dimensions"] == 384
    assert settings.embedding_dim == 384


def test_search_returns_requested_number_of_hits(built_index):
    assert len(built_index.search("anemia treatment", k=3)) == 3


def test_hits_are_ordered_by_descending_score(built_index):
    scores = [h.score for h in built_index.search("diabetes diagnosis", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_hits_carry_page_provenance(built_index):
    for hit in built_index.search("vitamins", k=3):
        assert hit.page >= 1
        assert hit.source.endswith(".pdf")


def test_manifest_records_how_the_index_was_built(built_index):
    manifest = built_index.manifest
    assert manifest["chunk_size"] == settings.chunk_size
    assert manifest["embedding_backend"] == "hashing"
    assert manifest["total_seconds"] >= 0


def test_loading_rejects_an_index_built_with_a_different_chunk_size(chunks, tmp_path, monkeypatch):
    """Silently querying a stale index is worse than refusing to load it."""
    index.build(chunks, tmp_path)
    monkeypatch.setattr(index, "settings", dataclasses.replace(settings, chunk_size=999))
    with pytest.raises(RuntimeError, match="chunk_size"):
        index.load(tmp_path)


def test_loading_rejects_a_different_embedding_backend(chunks, tmp_path, monkeypatch):
    """Vectors from two different backends are not comparable at all."""
    index.build(chunks, tmp_path)
    monkeypatch.setattr(index, "settings",
                        dataclasses.replace(settings, embedding_backend="minilm"))
    with pytest.raises(RuntimeError, match="backend"):
        index.load(tmp_path)


def test_missing_index_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        index.load(tmp_path / "nothing")
