# Healix

A question-answering assistant over a single medical encyclopedia. It answers
from the book and shows the pages it used; when the book has no answer, it says
so instead of inventing one.

```
PDF --> clean + chunk --> MiniLM 384-dim --> FAISS --> top-4 passages
                                                           |
                                          constrained prompt + GPT-4o
                                                           |
                                        3 sentences + page citations
```

## Why it's built this way

A general model answering medical questions is confidently wrong often enough to
be unusable. Three things address that, and none of them is the retriever:

- **The model sees only the retrieved passages.** Four chunks and a question,
  with an instruction that they are the only permitted source.
- **A sanctioned way to refuse.** An exact refusal string the model is told to
  emit when the passages don't cover the question. Without a permitted "not
  here", a model answers anyway.
- **Three sentences, hard.** Elaboration is where unsupported claims appear.

Full reasoning in [`docs/grounding.md`](docs/grounding.md).

## Quick start

```bash
pip install -r requirements-dev.txt
cp .env.example .env              # OPENAI_API_KEY is needed for answers only

pytest                            # 46 tests, no API key, no model download

# put the encyclopedia PDF in data/ first - see data/README.md
python scripts/corpus_stats.py    # pages, size, chunk count
python scripts/build_index.py     # builds the index, prints the wall clock
python app.py                     # http://localhost:8080
```

## Reproducing the numbers

Every figure has a command behind it. Only the last one costs money.

| Figure | Command | Needs |
|---|---|---|
| Pages, file size, chunk count | `python scripts/corpus_stats.py` | the PDF |
| Index build time | `python scripts/build_index.py` | the PDF |
| Embedding dimensions | `curl localhost:8080/api/stats` | a built index |
| Retrieval latency, p50/p95/p99 | `python scripts/benchmark.py` | a built index |
| End-to-end response time | `python scripts/benchmark.py --end-to-end` | API key |
| Refusal and grounding rates | `python scripts/evaluate_grounding.py --ablation` | API key |

Everything above the last two rows runs locally on CPU at no cost. The API
returns `retrieval_ms`, `generation_ms` and `total_ms` on every answer, so the
latency figures are checkable from outside the benchmark too.

## The corpus is not in this repo

`data/*.pdf` is gitignored. A 16 MB copyrighted encyclopedia does not belong in
git history, and a public repository should not redistribute it. Supply your
own copy and `scripts/corpus_stats.py` reports its real figures.

`data/fixtures/mini_encyclopedia.pdf` is committed on purpose: a small generated
PDF with nine entries, repeated running heads and bare page numbers, so the
cleaning rules have something real to strip and CI runs without the corpus.
Regenerate it with `scripts/make_fixture.py`.

## Running without torch

`sentence-transformers` needs a Python version torch publishes wheels for
(3.10-3.13 at the time of writing). Where that isn't available:

```bash
EMBEDDING_BACKEND=hashing python scripts/build_index.py
```

This swaps in a deterministic 384-dimensional projection built from numpy alone.
The pipeline runs end to end and the tests pass, but it matches on token overlap
rather than meaning, so retrieval is much worse. It is a way to exercise the
system, not a substitute for the model - the reported numbers come from
`minilm`, and the index manifest records which backend built it so the two can
never be mixed.

## Layout

```
src/healix/
  config.py      every tunable, from the environment
  corpus.py      PDF extraction, cleaning, chunking, corpus stats
  embeddings.py  MiniLM (default) and the hashing fallback
  index.py       FAISS build/load/search, with a manifest guard
  prompts.py     the constrained prompt and the refusal contract
  llm.py         OpenAI wrapper, imported lazily
  rag.py         retrieve then generate, timed per stage
  api.py         Flask routes
scripts/
  corpus_stats.py        corpus figures, measured from the file
  build_index.py         build the knowledge base
  benchmark.py           latency, p50/p95/p99
  evaluate_grounding.py  refusal + grounding, with an ablation
  make_fixture.py        regenerate the test PDF
```

## API

```bash
curl -X POST localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What causes iron deficiency anaemia?"}'
```

```json
{
  "answer": "Iron deficiency anemia develops when iron stores fall below...",
  "sources": [{"page": 412, "score": 0.71, "snippet": "Iron deficiency anemia..."}],
  "refused": false,
  "retrieval_ms": 41.3,
  "generation_ms": 2140.2,
  "total_ms": 2181.5
}
```

`GET /api/stats` returns the manifest of the index the running instance actually
loaded - chunk count, dimensions, backend, build time. `GET /api/health` is the
container probe.

A refusal comes back with `refused: true` and an empty `sources` list, because
listing the retrieved passages under a refusal would imply the answer came from
them. The frontend styles it differently for the same reason.

## Deployment

Docker image, pushed to ECR, rolled out to EC2 by
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) on every push to `main`,
verified against `/api/health` before the job passes.

The embedding weights are baked into the image so a cold start doesn't wait on a
90 MB download. The index is mounted as a volume instead - rebuilding it on
every deploy would add minutes to each rollout for a corpus that rarely changes.

Repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`ECR_REPOSITORY`, `ECR_REGISTRY`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`,
`OPENAI_API_KEY`.

## Known issues

- **Answer correctness isn't automatically measured.** The grounding harness
  checks that figures come from the passages and that the model refuses when it
  should. It cannot tell whether a grounded three-sentence answer misread its
  source. That needs a human reading cited pages; the template is in `eval/gold/`.
- Dense retrieval only. A BM25 hybrid is the change most likely to improve
  recall on rare drug names and abbreviations.
- No reranker. A cross-encoder over the top 20 would sharpen precision at k=4
  for roughly 50 ms.
- No conversation memory - each question is answered independently, so
  follow-ups like "what about in children?" lose their referent.
- British and American spellings aren't normalised. MiniLM mostly absorbs this;
  the hashing backend doesn't at all.
- Scanned PDFs need OCR first; pypdf extracts nothing from page images.

## Not medical advice

A reference tool over a static text. Not a diagnostic aid, and not something to
make decisions about anyone's health with.

## Author

Pal Ajay Ramsagar - github.com/ajaypal5117

MIT licensed.
