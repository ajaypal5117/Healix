# Architecture

```
            PDF                         question
             |                             |
        corpus.py                          |
   clean pages, strip running heads        |
   chunk at 500 chars / 50 overlap         |
             |                             |
      embeddings.py  ----------------------+
      MiniLM, 384-dim, CPU                 |
             |                             |
         index.py                          |
   FAISS IndexFlatIP + manifest        search top-4
             |                             |
             +-------------> hits ---------+
                                |
                           prompts.py
                    constrained system prompt
                                |
                             llm.py
                          GPT-4o, 3 sentences
                                |
                             rag.py
                   answer + page citations + timings
                                |
                             api.py
                       Flask / AJAX frontend
```

## Decisions worth explaining

**Why 500-character chunks.** Encyclopedia entries are dense and self-contained:
a definition, its causes, its presentation, its treatment, in a few hundred
characters. At 1,000 a chunk spans two unrelated conditions and its vector sits
between both, matching neither well. At 200 the treatment sentence gets
separated from the condition it treats. 500 with 50 overlap keeps one topic per
chunk while letting a definition that straddles a boundary be retrieved from
either side.

**Why `IndexFlatIP` rather than an approximate index.** Exact inner-product
search over unit-normalised vectors is cosine similarity with no recall loss.
At a few thousand vectors the search is microseconds; the round trip is
dominated by embedding the query, not searching. HNSW or IVF would add tuning
and a recall cliff to optimise a stage that is not the bottleneck.

**Why the manifest is checked on load.** An index built at chunk size 500 and
queried by a process configured for 1,000 returns results that look plausible
and are wrong. The loader compares the manifest against current settings and
refuses rather than serving nonsense. Same for the embedding backend: vectors
from two different models are not comparable at all.

**Why generation is a lazy import.** `llm.py` imports `openai` inside the
function. That is what lets indexing, retrieval, the benchmark and the entire
test suite run with no API key and no openai package installed.

**Why one gunicorn worker with threads.** The model and the index are held in
memory once; a second worker doubles the resident set. The work is waiting on
the OpenAI API, so threads hide that latency without duplicating memory.

**Why the index is a volume, not baked into the image.** Rebuilding on every
deploy would add minutes to each rollout for a corpus that changes rarely. The
embedding *weights* are baked in, because those are fixed and downloading them
on every cold start delays the first request.

## Request path

One question costs one query embedding, one vector search, one completion.
Timings are reported per stage in the API response, which is what makes the
latency numbers checkable from the outside rather than only in a benchmark.
