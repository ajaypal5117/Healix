# Evaluation

Two things are measured, and they answer different questions.

## Automatic - `scripts/evaluate_grounding.py`

`questions.json` holds 18 questions, deliberately split:

- **10 in-corpus** - standard reference questions a medical encyclopedia covers.
- **8 out-of-corpus** - recent trials, current guideline numbers, drug prices,
  local hospital information, questions about the reader personally. Each one is
  answerable-sounding and *not* in a general encyclopedia, which is exactly the
  situation where a model invents something.

Three machine-checkable measures:

| Measure | What it catches |
|---|---|
| Refusal rate on out-of-corpus questions | fabrication when the corpus has no answer |
| Numeric grounding | figures in the answer that appear nowhere in the retrieved passages |
| Sentence compliance | whether the three-sentence constraint holds |

Numeric grounding is the sharpest of the three. A number that is not in the
context was produced by the model, and in a medical answer a fabricated dosage
or percentage is the most dangerous possible output.

Run `--ablation` to get the unconstrained baseline alongside. Without it the
numbers describe the system but say nothing about whether the constraints did
anything.

## Manual - `gold/labels.template.csv`

The automatic checks cannot tell whether a grounded, three-sentence, confident
answer is *correct*. That needs someone reading the cited pages. The template
has the judgement columns; fill them in against the page numbers the API
returns.
