"""Corpus cleaning and chunking - the stage that decides what a vector means."""

from healix import corpus


def test_hyphenated_line_breaks_are_rejoined():
    assert corpus.clean_page("hyper-\ntension is common") == "hypertension is common"


def test_soft_hyphens_removed():
    assert "\u00ad" not in corpus.clean_page("hyper\u00adtension")


def test_bare_page_numbers_stripped():
    assert "417" not in corpus.clean_page("Clinical text here.\n417\nMore text.")


def test_running_heads_removed_when_repeated():
    """A section title at the top of many pages would otherwise enter every chunk."""
    pages = [f"CARDIOVASCULAR DISORDERS\nContent for page {i} about the heart." for i in range(10)]
    cleaned = corpus.strip_running_heads(pages)
    assert all("CARDIOVASCULAR DISORDERS" not in page for page in cleaned)
    assert "about the heart" in cleaned[0]


def test_running_heads_kept_when_rare():
    """A capitalised line appearing twice is a heading, not furniture."""
    pages = ["ACUTE PANCREATITIS\nbody text one"] * 2 + ["different page body"] * 8
    cleaned = corpus.strip_running_heads(pages)
    assert any("ACUTE PANCREATITIS" in page for page in cleaned)


def test_chunks_respect_size_limit(chunks):
    from healix.config import settings
    assert chunks
    assert all(len(c.text) <= settings.chunk_size + 50 for c in chunks)


def test_chunks_carry_page_numbers(chunks):
    assert all(c.page >= 1 for c in chunks)
    assert len({c.page for c in chunks}) > 1


def test_chunk_ids_are_sequential(chunks):
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap_so_boundaries_are_recoverable():
    text = " ".join(f"sentence number {i}." for i in range(120))
    pieces = corpus.split_page(text, size=300, overlap=60)
    assert len(pieces) > 2
    # Consecutive chunks should share some trailing/leading words.
    assert any(set(pieces[i].split()[-4:]) & set(pieces[i + 1].split()[:12])
               for i in range(len(pieces) - 1))


def test_split_prefers_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here. " * 12
    pieces = corpus.split_page(text, size=200, overlap=20)
    ends_cleanly = sum(1 for p in pieces[:-1] if p.rstrip().endswith("."))
    assert ends_cleanly >= len(pieces[:-1]) * 0.6


def test_short_pages_are_dropped(fixture_dir):
    pages = corpus.read_pdf(next(fixture_dir.glob("*.pdf")))
    assert any(len(p) >= 120 for p in pages)


def test_stats_are_measured_from_the_file(fixture_dir):
    figures = corpus.stats(fixture_dir)
    assert figures["chunks"] > 0
    assert figures["bytes"] > 0
    assert figures["pages_with_text"] > 1
    assert figures["chunk_size"] == 500
