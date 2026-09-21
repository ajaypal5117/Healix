"""Flask routes, with retrieval and generation both stubbed."""

import pytest

from healix import rag
from healix.api import create_app


@pytest.fixture
def client(built_index, monkeypatch, stub_llm):
    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    monkeypatch.setattr(rag, "warm_up", lambda: None)
    monkeypatch.setattr("healix.llm.complete", stub_llm)

    app = create_app(warm=False)
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Healix" in response.data


def test_stats_exposes_the_loaded_index_manifest(client):
    payload = client.get("/api/stats").get_json()
    assert payload["dimensions"] == 384
    assert payload["chunks"] > 0


def test_chat_returns_answer_and_sources(client):
    response = client.post("/api/chat", json={"question": "What causes anaemia?"})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["answer"]
    assert payload["sources"]
    assert payload["retrieval_ms"] >= 0


def test_empty_question_rejected(client):
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_overlong_question_rejected(client):
    assert client.post("/api/chat", json={"question": "a" * 5000}).status_code == 400


def test_malformed_body_does_not_500(client):
    assert client.post("/api/chat", data="not json",
                       content_type="application/json").status_code == 400


def test_unknown_route_returns_json(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_generation_failure_returns_503_not_a_stack_trace(client, monkeypatch):
    def boom(question):
        raise RuntimeError("upstream down")

    monkeypatch.setattr("healix.api.rag.answer", boom)
    response = client.post("/api/chat", json={"question": "anything"})
    assert response.status_code == 503
    assert "upstream down" not in response.get_data(as_text=True)
