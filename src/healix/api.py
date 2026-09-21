"""Flask application.

GET  /           chat interface
POST /api/chat   {"question": "..."} -> answer, sources, per-stage timings
GET  /api/health liveness probe for the container and the load balancer
GET  /api/stats  index manifest - what the running instance actually loaded
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from . import rag
from .config import ROOT, settings

log = logging.getLogger("healix.api")


def create_app(warm=True) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    if warm:
        rag.warm_up()

    @app.get("/")
    def index():
        return render_template("index.html", model=settings.llm_model)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.get("/api/stats")
    def stats():
        try:
            manifest = rag.get_index().manifest
        except Exception as error:
            return jsonify({"error": str(error)}), 503
        return jsonify(manifest), 200

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()

        if not question:
            return jsonify({"error": "Type a question to search the encyclopedia."}), 400
        if len(question) > settings.max_question_chars:
            return jsonify({
                "error": f"Questions are limited to {settings.max_question_chars} characters."
            }), 400

        try:
            result = rag.answer(question)
        except Exception:
            log.exception("failed to answer")
            return jsonify({
                "error": "The service is unavailable right now. Try again in a moment."
            }), 503

        return jsonify(result.to_dict()), 200

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "No such endpoint."}), 404

    return app
