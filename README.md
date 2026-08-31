# Healix

A retrieval-augmented chatbot that answers medical questions from one 637-page
encyclopedia and shows the pages each answer came from. Nothing is answered from
the language model's own memory — if the corpus doesn't cover the question,
Healix says so.

```
question ──▶ MiniLM embedding ──▶ vector search (top-4) ──▶ GPT-4o ──▶ 3-sentence answer
                                        │                                    │
                                  1,500 chunks                        page citations
```

## Why it's built this way

A general model answering medical questions is confidently wrong often enough to
be useless. Three constraints do the work here:

- **Retrieval before generation.** GPT-4o only ever sees four passages pulled
  from the encyclopedia, and the prompt forbids drawing on anything else.
- **A hard length cap.** Answers are limited to three sentences. Most
  hallucination happens in the elaboration, so the elaboration is removed.
- **Visible provenance.** Every answer ships with the page numbers behind it, so
  a wrong answer is checkable rather than plausible.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (`create_retrieval_chain`) |
| Embeddings | `all-MiniLM-L6-v2`, 384-dim, CPU |
| Vector store | FAISS on disk (default) or Pinecone serverless |
| Generation | GPT-4o via `langchain-openai` |
| API | Flask + gunicorn |
| Frontend | Bootstrap 5, jQuery, AJAX |
| Delivery | Docker → Amazon ECR → EC2, via GitHub Actions |

## Measured performance

Reproduce with `python benchmark.py --end-to-end` (t3.medium, FAISS backend,
30 retrieval samples):

| Metric | Value |
|---|---|
| Vector retrieval | under 100 ms p95 |
| End to end, including generation | 1.5 – 4 s |
| Full corpus index build | under 10 minutes |
| Chunks indexed | ~1,500 from 637 pages (16.1 MB) |

Retrieval is fast because the index is small and local; the variance in the
end-to-end figure is almost entirely the OpenAI completion.

## Running it

```bash
git clone https://github.com/tanu2k4/healix.git
cd healix

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY
```

Put a medical encyclopedia PDF in `data/` (see `data/README.md`), build the
index, then start the server:

```bash
python store_index.py       # one-time; ~7 minutes for 637 pages on CPU
python app.py               # http://localhost:8080
```

### Docker

```bash
docker build -t healix .
docker run -p 8080:8080 --env-file .env -v "$PWD/artifacts:/app/artifacts" healix
```

The embedding model is baked into the image, so containers start without
downloading weights.

## API

```bash
curl -X POST http://localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What causes iron deficiency anaemia?"}'
```

```json
{
  "answer": "Iron deficiency anaemia results from inadequate dietary iron, poor absorption, or chronic blood loss...",
  "sources": [
    { "source": "encyclopedia.pdf", "page": 412, "snippet": "Iron deficiency anemia is caused by..." }
  ],
  "retrieval_ms": 41.3,
  "total_ms": 2180.6
}
```

`GET /health` returns the liveness status and the active vector backend.

## Layout

```
healix/
├── app.py                  Flask routes
├── store_index.py          corpus → chunks → vectors
├── benchmark.py            latency harness
├── src/
│   ├── config.py           every tunable, read from the environment
│   ├── loader.py           PDF loading, cleaning, chunking
│   ├── embeddings.py       cached MiniLM singleton
│   ├── vectorstore.py      FAISS / Pinecone behind one interface
│   ├── prompts.py          the constrained system prompt
│   └── chain.py            retriever + GPT-4o, with timings
├── templates/index.html
├── static/css/style.css
├── static/js/chat.js
├── tests/
└── .github/workflows/cicd.yml
```

## Deployment

`.github/workflows/cicd.yml` runs lint and tests on every push, then on `main`
builds the image, pushes it to ECR, and restarts the container on EC2 over SSH.
The rollout is verified against `/health` before the job passes.

Required repository secrets:

`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`,
`ECR_REGISTRY`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `OPENAI_API_KEY`, and —
if using the managed store — `VECTOR_BACKEND`, `PINECONE_API_KEY`,
`PINECONE_INDEX_NAME`.

## Configuration

Chunk size, `top_k`, the embedding model, the LLM and the vector backend are all
environment variables; see `.env.example`. Switching to Pinecone is one variable:

```bash
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=...
python store_index.py
```

## Tests

```bash
pytest
```

The suite covers text cleaning, chunk boundaries, metadata propagation, request
validation and error handling. The chain is stubbed, so no index or API key is
needed.

## Known limits

- Answers reflect one encyclopedia. A gap in the corpus is a gap in Healix.
- Retrieval is dense-only; rare drug names and abbreviations would benefit from a
  BM25 hybrid.
- No conversation memory — each question is answered independently.
- Scanned PDFs need OCR before indexing.

## Not medical advice

This is a reference tool over a static text. It is not a diagnostic aid and
should not be used to make decisions about anyone's health.

## License

MIT — see [LICENSE](LICENSE).
