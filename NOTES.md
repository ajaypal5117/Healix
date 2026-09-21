# Working notes

Running log of decisions, things that broke, and what's still open.

## Decisions

**Chunk size 500.** Tried 1000 first. Encyclopedia entries are short and dense,
so a 1000-char chunk regularly spanned two unrelated conditions and its vector
sat between both, matching neither. 200 split treatment away from the condition
it treats. 500 with 50 overlap keeps one topic per chunk.

**Exact FAISS index, not HNSW.** A few thousand vectors. The search is
microseconds either way and the round trip is dominated by embedding the query.
An approximate index would have added tuning and a recall cliff to optimise a
stage that isn't the bottleneck.

**The refusal is an exact string.** Started with "say you don't know" in the
prompt, which produced a different phrasing every time and made it impossible to
measure the refusal rate automatically. Pinning the exact sentence made it
checkable, and `prompts.is_refusal` is what the eval and the frontend both use.

**Lazy OpenAI import.** Moved `import openai` inside the function so the tests,
the benchmark and indexing all run with no key and no package. Worth doing early
- it kept CI from ever needing a secret.

**Hashing embedding backend.** Added after hitting a machine where torch had no
wheel for the installed Python. It's not a quality substitute and the README says
so, but it means the pipeline is always runnable and CI never downloads 90 MB.

## Bugs worth remembering

- The index loader would happily open an index built at a different chunk size
  and return plausible nonsense. Now the manifest records chunk size, overlap and
  backend, and the loader refuses on mismatch. Two tests pin it.
- Refusals were coming back with the retrieved passages attached as sources,
  which implied the answer came from them. Refusals now carry an empty source
  list and the frontend styles them differently.
- `strip_running_heads` removed real headings at first - the threshold was too
  low, so a section title appearing on two pages got deleted. Raised to a
  quarter of pages with a floor of three.
- The fixture PDF was invisible to `find_pdfs` when pointed at directly, because
  the filter that stops `data/` scans picking up fixtures also fired on an
  explicit `data/fixtures` path.
- `$.trim` was removed in jQuery 4 and the send button silently did nothing.
  Client now uses native `.trim()` and `.trigger('focus')`.

## Open

- [ ] BM25 hybrid. Dense-only retrieval misses rare drug names; this is the
      highest-value change left.
- [ ] Cross-encoder rerank over the top 20. Roughly 50 ms for better precision
      at k=4.
- [ ] Conversation memory. "What about in children?" currently loses its
      referent entirely.
- [ ] Normalise British/American spellings before embedding.
- [ ] Human correctness labelling - the automatic checks can't catch a grounded
      answer that misreads its source.
- [ ] Streaming responses. Most of the end-to-end time is generation, and a
      user seeing the first sentence immediately would feel much faster even at
      the same total.
