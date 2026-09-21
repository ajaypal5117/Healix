# Retrieval

## Embeddings

`sentence-transformers/all-MiniLM-L6-v2`: six transformer layers, 384-dimensional
output, runs on CPU. It is small enough to sit inside a container without a GPU
while scoring well on retrieval benchmarks, which is the trade the whole design
rests on - a larger encoder would improve recall slightly and break the latency
budget badly.

Vectors are L2-normalised at encode time, so inner product *is* cosine
similarity and FAISS needs no separate normalisation step.

## The hashing fallback

`EMBEDDING_BACKEND=hashing` swaps in a deterministic 384-dim projection built
from numpy alone: every token is hashed to a seed, expanded to a random vector,
and summed across the unique tokens of the text.

It exists for two practical reasons. torch does not publish wheels for every
Python version - on 3.14 the real backend cannot be installed at all - and CI
should not download 90 MB of weights to check that chunking works.

It matches on token overlap, not meaning. Ask it about "anaemia" when the text
says "anemia" and it will miss. **It is a way to exercise the pipeline, not a
substitute for the model**, and the reported retrieval numbers come from
`minilm`. The index manifest records which backend built it, and the loader
refuses to mix them.

## Chunking

| Setting | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 500 chars | one encyclopedia topic per chunk |
| `CHUNK_OVERLAP` | 50 chars | a definition crossing a boundary stays retrievable |
| `TOP_K` | 4 | enough context for a three-sentence answer without diluting it |

Splitting prefers paragraph breaks, then sentence ends, then word boundaries, so
chunks rarely end mid-sentence. `tests/test_corpus.py` pins that behaviour.

## Cleaning

PDF text extraction produces artefacts that would otherwise end up inside
vectors:

- **Hyphenated line breaks** - "hyper-\ntension" becomes "hypertension" rather
  than two meaningless tokens.
- **Bare page numbers** - a line containing only digits carries no meaning.
- **Running heads** - a section title repeated at the top of every page. Left
  in, that phrase enters hundreds of chunks and pulls retrieval toward whichever
  section has the most pages. Removed when a short all-caps line repeats across
  a quarter of pages or more.
- **Short pages** - plates, blank pages and index stubs are dropped below
  `MIN_PAGE_CHARS`.

## Known weaknesses

- Dense retrieval only. Rare drug names and abbreviations would benefit from a
  BM25 hybrid, which is the single change most likely to improve recall.
- No reranker. A cross-encoder over the top 20 would improve precision at k=4
  at the cost of roughly 50 ms.
- No query expansion, so a question phrased very differently from the
  encyclopedia's wording retrieves poorly.
- British and American spellings are not normalised ("anaemia" vs "anemia").
  MiniLM largely absorbs this; the hashing backend does not at all.
