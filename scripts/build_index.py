"""Build the searchable knowledge base.

    python scripts/build_index.py

Reads the PDF, chunks it, embeds every chunk and writes a FAISS index plus a
manifest. Prints the wall clock, which is the "indexed in under N minutes"
figure - measured, not estimated.
"""

import argparse
import logging
import sys

import _bootstrap  # noqa: F401

from healix import corpus, index
from healix.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    args = parser.parse_args()

    log.info("embedding backend: %s", settings.embedding_backend)
    chunks = corpus.build_chunks(args.data_dir)

    if not chunks:
        log.error("no chunks produced - is the PDF scanned rather than text-based?")
        return 1

    manifest = index.build(chunks, args.index_dir)

    print("\nindex built")
    print(f"  chunks           {manifest['chunks']:,}")
    print(f"  dimensions       {manifest['dimensions']}")
    print(f"  backend          {manifest['embedding_backend']}")
    print(f"  embedding time   {manifest['embed_seconds']:.1f}s")
    print(f"  index time       {manifest['index_seconds']:.2f}s")
    print(f"  total            {manifest['total_seconds']:.1f}s "
          f"({manifest['total_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
