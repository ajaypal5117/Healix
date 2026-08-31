"""Corpus loading and chunking.

The source corpus is a 637-page (16.1 MB) medical encyclopedia PDF. Pages are
read directly with ``pypdf`` — no document-loader wrapper — so the cleaning rules
below can run before anything is embedded. Splitting recursively on paragraph
boundaries then yields roughly 1,500 chunks of ~500 characters, which is the
granularity the retriever expects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Page furniture that survives PDF extraction and pollutes embeddings.
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """Normalise whitespace, de-hyphenate line breaks and drop page numbers."""
    text = raw.replace("\u00ad", "")              # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words broken across lines
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_pdfs(data_dir: Path | None = None) -> List[Document]:
    """Read every PDF in ``data_dir`` into one Document per page."""
    from pypdf import PdfReader

    directory = Path(data_dir or settings.data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files in {directory}. Drop the medical encyclopedia PDF there first "
            "(see data/README.md)."
        )

    documents: List[Document] = []
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        logger.info("Reading %s (%d pages)", pdf.name, len(reader.pages))
        for page_number, page in enumerate(reader.pages):
            try:
                content = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - a damaged page shouldn't kill the run
                logger.warning("Could not extract page %d of %s", page_number + 1, pdf.name)
                continue
            documents.append(
                Document(page_content=content, metadata={"source": pdf.name, "page": page_number})
            )

    logger.info("Extracted %d pages from %d file(s)", len(documents), len(pdfs))
    return documents


def filter_pages(documents: Iterable[Document], min_chars: int = 120) -> List[Document]:
    """Drop blank pages, plates and index stubs that add noise but no meaning."""
    kept: List[Document] = []
    for doc in documents:
        text = clean_text(doc.page_content)
        if len(text) < min_chars:
            continue
        metadata = {
            "source": Path(doc.metadata.get("source", "unknown")).name,
            "page": doc.metadata.get("page"),
        }
        kept.append(Document(page_content=text, metadata=metadata))
    return kept


def split_documents(documents: Iterable[Document]) -> List[Document]:
    """Split pages into overlapping semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    logger.info("Produced %d chunks", len(chunks))
    return chunks


def build_corpus(data_dir: Path | None = None) -> List[Document]:
    """Full pipeline: read PDFs, clean pages, split into chunks."""
    return split_documents(filter_pages(load_pdfs(data_dir)))
