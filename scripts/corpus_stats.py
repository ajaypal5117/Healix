"""Report the corpus figures, measured from the PDF.

    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --data-dir data/fixtures

Prints pages, file size and chunk count. These are the numbers the README
quotes; run it rather than trusting what is written down.
"""

import argparse
import json
import logging

import _bootstrap  # noqa: F401

from healix import corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    figures = corpus.stats(args.data_dir)

    if args.json:
        print(json.dumps(figures, indent=2))
        return

    print("\ncorpus")
    print(f"  files            {', '.join(figures['files'])}")
    print(f"  size             {figures['megabytes']} MB ({figures['bytes']:,} bytes)")
    print(f"  pages with text  {figures['pages_with_text']}")
    print(f"  chunks           {figures['chunks']:,}")
    print(f"  chunk size       {figures['chunk_size']} chars, "
          f"{figures['chunk_overlap']} overlap")
    print(f"  mean chunk       {figures['mean_chunk_chars']} chars")
    print(f"  total text       {figures['characters']:,} chars")


if __name__ == "__main__":
    main()
