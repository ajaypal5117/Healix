"""Retrieval-augmented generation chain.

Built with LangChain Expression Language rather than the legacy
``create_retrieval_chain`` helper, for two reasons: LCEL primitives live in
``langchain_core`` and are stable across LangChain 0.3 and 1.x, and running
retrieval as an explicit step means the documents are fetched exactly once and
can be timed and returned alongside the answer.

Per request the work is: one embedding, one vector lookup, one completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.prompts import CHAT_PROMPT
from src.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """A single response together with its provenance and timings."""

    text: str
    sources: List[Dict[str, Any]]
    retrieval_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.text,
            "sources": self.sources,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@lru_cache(maxsize=1)
def get_retriever():
    store = load_vectorstore()
    return store.as_retriever(search_type="similarity", search_kwargs={"k": settings.top_k})


@lru_cache(maxsize=1)
def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
        timeout=30,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_generation_chain():
    """prompt -> GPT-4o -> plain string."""
    logger.info("Building generation chain with %s", settings.llm_model)
    return CHAT_PROMPT | get_llm() | StrOutputParser()


def format_context(documents: Sequence[Document]) -> str:
    """Render retrieved chunks with their page numbers so the model can cite them."""
    blocks = []
    for doc in documents:
        page = doc.metadata.get("page")
        label = f"[page {page + 1}]" if isinstance(page, int) else "[page unknown]"
        blocks.append(f"{label}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def _format_sources(documents: Sequence[Document]) -> List[Dict[str, Any]]:
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for doc in documents:
        page = doc.metadata.get("page")
        key = (doc.metadata.get("source"), page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content.strip().replace("\n", " ")
        sources.append(
            {
                "source": doc.metadata.get("source", "encyclopedia"),
                # PyPDF pages are zero-indexed; humans count from one.
                "page": (page + 1) if isinstance(page, int) else None,
                "snippet": snippet[:220] + ("..." if len(snippet) > 220 else ""),
            }
        )
    return sources


def ask(question: str) -> Answer:
    """Run one question through retrieval and generation."""
    started = time.perf_counter()

    retrieval_start = time.perf_counter()
    documents = get_retriever().invoke(question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    text = get_generation_chain().invoke(
        {"context": format_context(documents), "input": question}
    )
    total_ms = (time.perf_counter() - started) * 1000

    logger.info("q=%r retrieval=%.1fms total=%.1fms", question[:60], retrieval_ms, total_ms)
    return Answer(
        text=(text or "").strip(),
        sources=_format_sources(documents),
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
    )


def warm_up() -> None:
    """Load the index, model and chain before the first user request arrives."""
    get_retriever()
    get_generation_chain()
    logger.info("Chain warm and ready")
