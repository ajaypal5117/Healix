"""Latency benchmark.

    python benchmark.py                # retrieval only
    python benchmark.py --end-to-end   # retrieval + GPT-4o generation (costs tokens)

Reports p50 / p95 / max so the performance numbers in the README are
reproducible rather than asserted.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

QUESTIONS = [
    "What causes iron deficiency anaemia?",
    "How is type 2 diabetes diagnosed?",
    "What are the symptoms of appendicitis?",
    "Describe the stages of wound healing.",
    "What is the difference between a virus and a bacterium?",
    "How does the kidney regulate blood pressure?",
    "What are common complications of untreated hypertension?",
    "Explain how vaccines produce immunity.",
    "What is the treatment for a mild concussion?",
    "Which vitamins are fat soluble?",
]


def _time(fn: Callable[[str], object], question: str) -> float:
    started = time.perf_counter()
    fn(question)
    return (time.perf_counter() - started) * 1000


def report(label: str, samples: List[float]) -> None:
    ordered = sorted(samples)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    print(f"\n{label} over {len(samples)} queries")
    print(f"  p50  {statistics.median(ordered):7.1f} ms")
    print(f"  p95  {p95:7.1f} ms")
    print(f"  max  {ordered[-1]:7.1f} ms")
    print(f"  mean {statistics.mean(ordered):7.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Healix latency.")
    parser.add_argument("--end-to-end", action="store_true", help="Include LLM generation")
    parser.add_argument("--rounds", type=int, default=3, help="Passes over the question set")
    args = parser.parse_args()

    from src.chain import ask, get_retriever

    retriever = get_retriever()
    retriever.invoke("warm up")  # exclude one-off model load from the numbers

    retrieval: List[float] = []
    for _ in range(args.rounds):
        retrieval.extend(_time(retriever.invoke, q) for q in QUESTIONS)
    report("Retrieval", retrieval)

    if args.end_to_end:
        end_to_end = [_time(ask, q) for q in QUESTIONS]
        report("End to end (retrieval + GPT-4o)", end_to_end)


if __name__ == "__main__":
    main()
