# Corpus

Place the source encyclopedia PDF in this directory, then run:

```bash
python store_index.py
```

The reference build uses *The Gale Encyclopedia of Medicine* (637 pages, 16.1 MB),
which produces roughly 1,500 chunks at the default chunk size of 500 characters.

PDFs are excluded from version control by `.gitignore` — the repository stays
small and the corpus stays out of Git history. Any text-based medical PDF works;
scanned PDFs need OCR first (e.g. `ocrmypdf input.pdf output.pdf`).
