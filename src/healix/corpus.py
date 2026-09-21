"""Corpus loading, cleaning and chunking.

The source is a single medical encyclopedia PDF. Extraction is done page by
page with pypdf so the cleaning rules below run before anything is embedded -
PDF text carries hyphenated line breaks, running heads and bare page numbers
that would otherwise end up inside chunks and pollute their vectors.

`stats()` reports the corpus figures (pages, megabytes, chunk count) so they
are measured from the file rather than written down somewhere.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

SOFT_HYPHEN = "\u00ad"
HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
# Running heads: a short all-caps line repeated at the top of every page.
RUNNING_HEAD = re.compile(r"^[A-Z][A-Z \-'&]{4,60}$", re.MULTILINE)
SPACES = re.compile(r"[ \t\u00a0]+")
BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int
    source: str

    def to_metadata(self):
        return {"page": self.page, "chunk_id": self.chunk_id, "source": self.source}


def clean_page(raw: str) -> str:
    text = raw.replace(SOFT_HYPHEN, "")
    text = HYPHEN_BREAK.sub(r"\1\2", text)      # re-join words split across lines
    text = PAGE_NUMBER_LINE.sub("", text)
    text = SPACES.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()


def strip_running_heads(pages: list[str], threshold: float = 0.25) -> list[str]:
    """Remove short all-caps lines that repeat across many pages.

    An encyclopedia repeats its section title at the top of every page. Left in,
    that phrase appears in hundreds of chunks and pulls retrieval toward whatever
    section has the most pages.
    """
    counts: dict[str, int] = {}
    for page in pages:
        for line in set(RUNNING_HEAD.findall(page)):
            counts[line.strip()] = counts.get(line.strip(), 0) + 1

    repeated = {line for line, n in counts.items() if n >= max(3, len(pages) * threshold)}
    if not repeated:
        return pages

    log.info("removing %d running head(s)", len(repeated))
    cleaned = []
    for page in pages:
        for line in repeated:
            page = page.replace(line, " ")
        cleaned.append(SPACES.sub(" ", page).strip())
    return cleaned


def read_pdf(path: Path) -> list[str]:
    """One cleaned string per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages):
        try:
            pages.append(clean_page(page.extract_text() or ""))
        except Exception:
            log.warning("could not extract page %d of %s", number + 1, path.name)
            pages.append("")
    return pages


def find_pdfs(data_dir: Path | None = None) -> list[Path]:
    """Every PDF under `data_dir`.

    The test fixture lives in data/fixtures/, so a scan of data/ skips it -
    otherwise a real run would silently index the fixture alongside the
    encyclopedia. Pointing at data/fixtures/ explicitly still finds it.
    """
    directory = Path(data_dir or settings.data_dir)
    scanning_fixtures = "fixtures" in directory.parts
    pdfs = sorted(
        p for p in directory.rglob("*.pdf")
        if scanning_fixtures or "fixtures" not in p.parts
    )
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF in {directory}. Put the encyclopedia there - see data/README.md."
        )
    return pdfs


def split_page(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph, then sentence, then word boundaries.

    Chunks are cut at the largest separator that still fits, so a chunk rarely
    ends mid-sentence. Overlap carries the tail of one chunk into the next, so a
    definition spanning a boundary is retrievable from either side.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = -1
            for separator in separators:
                found = text.rfind(separator, start + size // 2, end)
                if found > cut:
                    cut = found + len(separator)
                    break
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(data_dir: Path | None = None) -> list[Chunk]:
    pdfs = find_pdfs(data_dir)
    chunks: list[Chunk] = []
    index = 0

    for pdf in pdfs:
        pages = strip_running_heads(read_pdf(pdf))
        log.info("%s: %d pages", pdf.name, len(pages))
        for page_number, page in enumerate(pages):
            if len(page) < settings.min_page_chars:
                continue                      # plates, blank pages, index stubs
            for piece in split_page(page, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(piece, page_number + 1, index, pdf.name))
                index += 1

    log.info("%d chunks", len(chunks))
    return chunks


def stats(data_dir: Path | None = None) -> dict:
    """Corpus figures, measured from the file."""
    pdfs = find_pdfs(data_dir)
    total_bytes = sum(p.stat().st_size for p in pdfs)
    chunks = build_chunks(data_dir)
    pages = max((c.page for c in chunks), default=0)
    characters = sum(len(c.text) for c in chunks)

    return {
        "files": [p.name for p in pdfs],
        "bytes": total_bytes,
        "megabytes": round(total_bytes / 1_048_576, 1),
        "pages_with_text": pages,
        "chunks": len(chunks),
        "characters": characters,
        "mean_chunk_chars": round(characters / len(chunks), 1) if chunks else 0,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
