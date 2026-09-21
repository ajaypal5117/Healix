# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app/src \
    PORT=8080

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first so this layer survives code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the embedding weights into the image. Without this every cold start
# pulls ~90 MB from HuggingFace before serving its first request.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

RUN useradd --create-home --uid 10001 healix && chown -R healix:healix /app
USER healix

EXPOSE 8080

# start-period is generous: the worker loads the model and index before it
# answers the first probe, and a tighter window kills a healthy container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/health || exit 1

# One worker, several threads: the model and index are held in memory once.
# A second worker would double the resident set for no throughput gain on a
# workload that spends its time waiting on the OpenAI API.
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-"]
