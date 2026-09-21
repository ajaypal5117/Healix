# Constraining the model

## The problem

In a RAG system the retriever is rarely the source of hallucination. The failure
is the generator: handed four passages and a question, a capable model will
produce fluent medical prose that goes beyond what the passages say, because
that is what it was trained to do. In a medical context a fabricated dosage or
percentage is the most dangerous output the system can produce, precisely
because it reads as authoritative.

## Four constraints

Each one is in `src/healix/prompts.py` and each is testable:

1. **Passages are the only permitted source.** Stated explicitly, and backed by
   giving the model nothing else in context.
2. **A sanctioned refusal.** `REFUSAL` is an exact string the model is told to
   emit when the passages do not answer the question. Without a permitted way
   to say "not here", a model answers anyway - refusing is not a behaviour it
   defaults to.
3. **Three sentences maximum.** Elaboration is where unsupported claims appear.
   Three sentences of clinical prose hold an answer and leave no room to drift.
   `LLM_MAX_TOKENS` is a backstop, not the mechanism - the instruction is.
4. **No invented specifics.** Drug names, dosages, measurements and figures are
   called out by name because they are the tokens a reader would act on.

## Measuring whether it works

`scripts/evaluate_grounding.py` runs `eval/questions.json`: 10 questions a
medical encyclopedia covers and 8 it cannot - recent trials, current guideline
numbers, drug prices, local hospital information, questions about the reader
personally. Every out-of-corpus question is answerable-sounding, which is
exactly the situation that produces invention.

| Measure | What it catches |
|---|---|
| Refusal rate on out-of-corpus questions | fabrication where the corpus is silent |
| False refusal rate on in-corpus questions | constraints tightened past usefulness |
| Numeric grounding | figures in the answer absent from the retrieved passages |
| Sentence compliance | whether the length cap holds |

Numeric grounding is the sharpest signal and the only fully automatic one. Every
number in the answer is checked against the retrieved context; one that is not
there was produced by the model.

The false-refusal rate matters as much as the refusal rate. A system that
refuses everything scores perfectly on hallucination and is useless, so both are
reported.

## The ablation

```bash
python scripts/evaluate_grounding.py --ablation
```

Same questions, same retrieved passages, two prompts: one with the constraints
and one plain baseline. Without the baseline the numbers describe the system but
say nothing about whether the constraints caused anything.

## What this does not measure

None of it establishes that a grounded, three-sentence, confident answer is
*correct*. Every figure could come from the passages and the answer still
misread them. That needs a person reading the cited pages against the answer,
which is what `eval/gold/labels.template.csv` is for.
