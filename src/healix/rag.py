"""Retrieval-augmented generation: the path a question actually takes.

One question costs one query embedding, one vector search and one completion.
Each stage is timed separately because they have very different profiles -
retrieval is sub-millisecond once the model is warm, generation is seconds and
varies with load at the API.

`answer` takes an optional `generate` callable so the tests can exercise the
whole path with a stub instead of calling OpenAI.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from . import index as index_module
from . import llm, prompts

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    refused: bool = False

    def to_dict(self):
        return {
            "answer": self.text,
            "sources": self.sources,
            "refused": self.refused,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@lru_cache(maxsize=1)
def get_index():
    return index_module.load()


def format_sources(hits) -> list[dict]:
    seen = set()
    sources = []
    for hit in hits:
        if hit.page in seen:
            continue
        seen.add(hit.page)
        snippet = " ".join(hit.text.split())
        sources.append({
            "page": hit.page,
            "source": hit.source,
            "score": round(hit.score, 3),
            "snippet": snippet[:220] + ("..." if len(snippet) > 220 else ""),
        })
    return sources


def retrieve(question: str, k: int | None = None):
    started = time.perf_counter()
    hits = get_index().search(question, k=k)
    return hits, (time.perf_counter() - started) * 1000


def answer(question: str, generate: Callable[[list[dict]], str] | None = None) -> Answer:
    generate = generate or llm.complete
    started = time.perf_counter()

    hits, retrieval_ms = retrieve(question)

    if not hits:
        return Answer(text=prompts.REFUSAL, refused=True,
                      retrieval_ms=retrieval_ms,
                      total_ms=(time.perf_counter() - started) * 1000)

    generation_started = time.perf_counter()
    text = generate(prompts.build_messages(question, hits))
    generation_ms = (time.perf_counter() - generation_started) * 1000

    total_ms = (time.perf_counter() - started) * 1000
    log.info("q=%r retrieval=%.1fms generation=%.0fms", question[:60],
             retrieval_ms, generation_ms)

    refused = prompts.is_refusal(text)
    return Answer(
        text=text,
        # A refusal has no supporting passages by definition; listing the
        # retrieved ones would imply the answer came from them.
        sources=[] if refused else format_sources(hits),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        refused=refused,
    )


def warm_up():
    """Load the index and embedding model before the first request arrives."""
    get_index()
    retrieve("warm up")
    log.info("index warm: %d chunks", len(get_index()))
