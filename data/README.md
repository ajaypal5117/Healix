# Corpus

Put the encyclopedia PDF here, then:

```bash
python scripts/build_index.py
```

The reference build uses a 637-page, 16.1 MB medical encyclopedia, which at the
default chunk size produces roughly 1,500 chunks.

`*.pdf` in this directory is gitignored. A 16 MB copyrighted binary does not
belong in git history, and redistributing it is not something a public
repository should do. `scripts/corpus_stats.py` reports the real figures from
whatever file you supply.

`fixtures/mini_encyclopedia.pdf` is different: a small generated PDF, committed
deliberately so the tests and CI run without the real corpus. It is regenerated
by `scripts/make_fixture.py`.

Scanned PDFs need OCR first (`ocrmypdf input.pdf output.pdf`) - pypdf extracts
no text from page images.
