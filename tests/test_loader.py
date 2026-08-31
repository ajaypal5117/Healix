"""Tests for corpus cleaning and chunking.

These run without the PDF, without network access and without an API key, so CI
stays fast and free.
"""

from langchain_core.documents import Document

from src.config import settings
from src.loader import clean_text, filter_pages, split_documents


def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("hyper-\ntension is common") == "hypertension is common"


def test_clean_text_strips_standalone_page_numbers():
    assert "417" not in clean_text("Some clinical text.\n417\nMore text.")


def test_clean_text_collapses_runs_of_whitespace():
    assert clean_text("a     b\t\tc") == "a b c"


def test_filter_pages_drops_short_pages():
    pages = [
        Document(page_content="x" * 400, metadata={"source": "/data/enc.pdf", "page": 3}),
        Document(page_content="short", metadata={"source": "/data/enc.pdf", "page": 4}),
    ]
    kept = filter_pages(pages)
    assert len(kept) == 1
    assert kept[0].metadata["page"] == 3


def test_filter_pages_keeps_only_the_filename_as_source():
    pages = [Document(page_content="y" * 400, metadata={"source": "/tmp/data/enc.pdf", "page": 0})]
    assert filter_pages(pages)[0].metadata["source"] == "enc.pdf"


def test_split_documents_respects_chunk_size():
    document = Document(
        page_content=". ".join(f"Sentence number {n}" for n in range(400)),
        metadata={"source": "enc.pdf", "page": 1},
    )
    chunks = split_documents([document])
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= settings.chunk_size + 50 for chunk in chunks)


def test_split_documents_assigns_sequential_chunk_ids():
    document = Document(
        page_content=". ".join(f"Clause {n}" for n in range(200)),
        metadata={"source": "enc.pdf", "page": 1},
    )
    chunks = split_documents([document])
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))


def test_split_documents_preserves_page_metadata():
    document = Document(page_content="z" * 2000, metadata={"source": "enc.pdf", "page": 12})
    assert all(chunk.metadata["page"] == 12 for chunk in split_documents([document]))
