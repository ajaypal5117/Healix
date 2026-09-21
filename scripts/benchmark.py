"""Latency benchmark.

    python scripts/benchmark.py                 retrieval only, no API key needed
    python scripts/benchmark.py --end-to-end    adds GPT-4o generation (spends tokens)

Reports p50/p95/p99 rather than a mean, because a mean hides the tail and the
tail is what a user notices. The model and index are warmed before timing so
the one-off load cost doesn't land in the first sample.
"""

import argparse
import json
import statistics
import time

import _bootstrap  # noqa: F401

from healix import rag
from healix.config import settings

QUESTIONS = [
    "What causes iron deficiency anaemia?",
    "How is type 2 diabetes diagnosed?",
    "What are the symptoms of appendicitis?",
    "Describe the stages of wound healing.",
    "What is the difference between a virus and a bacterium?",
    "How does the kidney regulate blood pressure?",
    "What are the complications of untreated hypertension?",
    "Explain how vaccines produce immunity.",
    "What is the treatment for a mild concussion?",
    "Which vitamins are fat soluble?",
    "What are the warning signs of a stroke?",
    "How is asthma managed long term?",
]


def percentile(values, p):
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered))) - 1))
    return ordered[position]


def report(label, samples, unit="ms"):
    print(f"\n{label}  (n={len(samples)})")
    print(f"  p50   {statistics.median(samples):8.1f} {unit}")
    print(f"  p95   {percentile(samples, 95):8.1f} {unit}")
    print(f"  p99   {percentile(samples, 99):8.1f} {unit}")
    print(f"  max   {max(samples):8.1f} {unit}")
    print(f"  mean  {statistics.mean(samples):8.1f} {unit}")
    return {
        "n": len(samples),
        "p50": round(statistics.median(samples), 1),
        "p95": round(percentile(samples, 95), 1),
        "p99": round(percentile(samples, 99), 1),
        "max": round(max(samples), 1),
        "mean": round(statistics.mean(samples), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args()

    rag.warm_up()
    print(f"index: {len(rag.get_index())} chunks | backend: "
          f"{rag.get_index().manifest['embedding_backend']} | top_k: {settings.top_k}")

    retrieval = []
    for _ in range(args.rounds):
        for question in QUESTIONS:
            started = time.perf_counter()
            rag.retrieve(question)
            retrieval.append((time.perf_counter() - started) * 1000)

    results = {"retrieval_ms": report("Retrieval (embed query + vector search)", retrieval)}

    if args.end_to_end:
        totals, generation = [], []
        for question in QUESTIONS:
            result = rag.answer(question)
            totals.append(result.total_ms)
            generation.append(result.generation_ms)
        results["generation_ms"] = report("Generation (GPT-4o)", generation)
        results["end_to_end_s"] = report(
            "End to end", [t / 1000 for t in totals], unit="s")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
