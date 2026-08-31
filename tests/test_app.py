"""Route tests.

The retrieval chain is stubbed out so these tests need neither an index nor an
OpenAI key.
"""

import pytest

import app as app_module
from src.chain import Answer
from src.config import Settings


@pytest.fixture()
def client(monkeypatch):
    def fake_ask(question: str) -> Answer:
        return Answer(
            text=f"Stub answer for: {question}",
            sources=[{"source": "enc.pdf", "page": 42, "snippet": "context snippet"}],
            retrieval_ms=12.5,
            total_ms=1800.0,
        )

    monkeypatch.setattr(app_module, "ask", fake_ask)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Healix" in response.data


def test_chat_returns_answer_and_sources(client):
    response = client.post("/api/chat", json={"question": "What is anaemia?"})
    assert response.status_code == 200

    payload = response.get_json()
    assert "Stub answer" in payload["answer"]
    assert payload["sources"][0]["page"] == 42
    assert payload["retrieval_ms"] == 12.5


def test_chat_rejects_empty_question(client):
    response = client.post("/api/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_chat_rejects_overlong_question(client):
    response = client.post("/api/chat", json={"question": "a" * 5000})
    assert response.status_code == 400


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_settings_reject_unknown_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="Unknown VECTOR_BACKEND"):
        Settings().validate_for_serving()


def test_settings_require_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings().validate_for_serving()
