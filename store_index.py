"""Build the searchable knowledge base.

    python store_index.py

Reads every PDF in ``data/``, cleans and chunks it, embeds each chunk with
all-MiniLM-L6-v2 and writes the vectors to the configured backend. Run this once
before starting the app, and again whenever the corpus changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import settings
from src.loader import build_corpus
from src.vectorstore import build_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
logger = logging.getLogger("store_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the medical corpus.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help=f"Directory containing the source PDFs (default: {settings.data_dir})",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    logger.info("Backend: %s | embedding: %s", settings.vector_backend, settings.embedding_model)
    chunks = build_corpus(args.data_dir)

    if not chunks:
        logger.error("No chunks produced - is the PDF text-based rather than scanned?")
        return 1

    characters = sum(len(chunk.page_content) for chunk in chunks)
    logger.info(
        "%d chunks | %.1f chars/chunk average | %.1f MB of text",
        len(chunks),
        characters / len(chunks),
        characters / 1_048_576,
    )

    build_vectorstore(chunks)

    elapsed = time.perf_counter() - started
    logger.info("Index built in %.1fs (%.1f min)", elapsed, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
