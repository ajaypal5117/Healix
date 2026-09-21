"""Measure whether the prompt constraints actually reduce unsupported output.

    python scripts/evaluate_grounding.py --dry-run     structure check, no API calls
    python scripts/evaluate_grounding.py               constrained prompt only
    python scripts/evaluate_grounding.py --ablation    constrained vs unconstrained

The claim "constrained prompts reduce hallucinations" is only worth anything
next to the unconstrained baseline, which is what --ablation runs: the same
questions, the same retrieved passages, one prompt with the constraints and one
without.

Three things are checked automatically:

  refusal rate       on questions the corpus cannot answer, how often the model
                     says so instead of inventing an answer
  numeric grounding  every number in the answer must also appear in the
                     retrieved passages - a figure that is not in the context
                     came from the model, which is the clearest machine-checkable
                     signal of fabrication
  length compliance  share of answers within the three-sentence cap

None of these prove correctness. A human still has to read the answers, which is
what eval/gold/labels.template.csv is for.
"""

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from healix import prompts, rag

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT / "eval" / "questions.json"

UNCONSTRAINED_SYSTEM = (
    "You are a helpful medical assistant. Use the passages below to answer the "
    "question.\n\nPassages:\n{context}"
)

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numeric_grounding(answer: str, context: str):
    """(grounded, total) numbers in the answer that also appear in the context."""
    in_answer = NUMBER.findall(answer or "")
    if not in_answer:
        return 0, 0
    in_context = set(NUMBER.findall(context or ""))
    grounded = sum(1 for n in in_answer if n in in_context)
    return grounded, len(in_answer)


def unconstrained_messages(question, hits):
    return [
        {"role": "system",
         "content": UNCONSTRAINED_SYSTEM.format(context=prompts.format_context(hits))},
        {"role": "user", "content": question},
    ]


def run_variant(questions, build_messages, label):
    from healix import llm

    rows = []
    for item in questions:
        hits, _ = rag.retrieve(item["question"])
        context = prompts.format_context(hits)
        answer = llm.complete(build_messages(item["question"], hits))

        grounded, total = numeric_grounding(answer, context)
        rows.append({
            "question": item["question"],
            "in_corpus": item["in_corpus"],
            "answer": answer,
            "refused": prompts.is_refusal(answer),
            "sentences": prompts.count_sentences(answer),
            "numbers_grounded": grounded,
            "numbers_total": total,
        })
        verdict = "refused" if rows[-1]["refused"] else f"{rows[-1]['sentences']} sentences"
        print(f"  [{label}] {item['question'][:52]:52} {verdict}")
    return rows


def summarise(rows, label):
    out_of_corpus = [r for r in rows if not r["in_corpus"]]
    in_corpus = [r for r in rows if r["in_corpus"]]

    refusal_rate = (sum(r["refused"] for r in out_of_corpus) / len(out_of_corpus)
                    if out_of_corpus else None)
    false_refusal = (sum(r["refused"] for r in in_corpus) / len(in_corpus)
                     if in_corpus else None)

    numbers_total = sum(r["numbers_total"] for r in rows)
    numbers_grounded = sum(r["numbers_grounded"] for r in rows)
    grounding = numbers_grounded / numbers_total if numbers_total else None

    within = sum(1 for r in rows if r["sentences"] <= 3) / len(rows) if rows else None

    print(f"\n{label}")
    if refusal_rate is not None:
        print(f"  refusal on out-of-corpus   {refusal_rate:.0%} "
              f"({sum(r['refused'] for r in out_of_corpus)}/{len(out_of_corpus)})")
    if false_refusal is not None:
        print(f"  false refusal in-corpus    {false_refusal:.0%} "
              f"({sum(r['refused'] for r in in_corpus)}/{len(in_corpus)})")
    if grounding is not None:
        print(f"  numeric grounding          {grounding:.0%} "
              f"({numbers_grounded}/{numbers_total} figures found in context)")
    if within is not None:
        print(f"  within 3 sentences         {within:.0%}")

    return {
        "label": label,
        "refusal_rate_out_of_corpus": refusal_rate,
        "false_refusal_rate_in_corpus": false_refusal,
        "numeric_grounding": grounding,
        "within_three_sentences": within,
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--ablation", action="store_true",
                        help="also run the unconstrained baseline")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the question set and retrieval, make no API calls")
    parser.add_argument("--out", default=str(ROOT / "eval" / "grounding_report.json"))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    print(f"{len(questions)} questions "
          f"({sum(q['in_corpus'] for q in questions)} in corpus, "
          f"{sum(not q['in_corpus'] for q in questions)} out of corpus)")

    if args.dry_run:
        rag.warm_up()
        for item in questions:
            hits, ms = rag.retrieve(item["question"])
            print(f"  {item['question'][:56]:56} {len(hits)} hits  {ms:5.1f} ms")
        print("\ndry run only - no answers generated")
        return

    rag.warm_up()
    results = []

    print("\nconstrained prompt")
    constrained = run_variant(questions, prompts.build_messages, "constrained")
    results.append(summarise(constrained, "Constrained prompt"))

    if args.ablation:
        print("\nunconstrained baseline")
        baseline = run_variant(questions, unconstrained_messages, "baseline")
        results.append(summarise(baseline, "Unconstrained baseline"))

    Path(args.out).write_text(json.dumps(
        {"summary": results, "constrained": constrained,
         "baseline": baseline if args.ablation else None}, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
