"""Healix Flask application.

Routes
------
GET  /         chat interface
POST /api/chat JSON question -> JSON answer, sources and timings
GET  /health   liveness probe for the container / load balancer
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from src.chain import ask, warm_up
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("healix")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.get("/")
def index():
    return render_template("index.html", model=settings.llm_model)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": settings.vector_backend}), 200


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Type a question to search the encyclopedia."}), 400
    if len(question) > settings.max_question_chars:
        return (
            jsonify(
                {
                    "error": f"Questions are limited to {settings.max_question_chars} "
                    "characters. Try a shorter one."
                }
            ),
            400,
        )

    try:
        answer = ask(question)
    except Exception:  # noqa: BLE001 - surface a clean message, log the detail
        logger.exception("Failed to answer question")
        return (
            jsonify({"error": "The retrieval service is unavailable. Try again in a moment."}),
            503,
        )

    return jsonify(answer.to_dict()), 200


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "No such endpoint."}), 404


def create_app() -> Flask:
    """Factory used by gunicorn: `gunicorn "app:create_app()"`."""
    settings.validate_for_serving()
    warm_up()
    return app


if __name__ == "__main__":
    settings.validate_for_serving()
    warm_up()
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
