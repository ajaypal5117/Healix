"""The retrieve-then-generate path, with generation stubbed."""

from healix import prompts, rag


def test_answer_returns_text_sources_and_timings(built_index, monkeypatch, stub_llm):
    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    result = rag.answer("What causes iron deficiency anaemia?", generate=stub_llm)

    assert result.text
    assert result.sources
    assert result.retrieval_ms > 0
    assert result.total_ms >= result.retrieval_ms


def test_sources_carry_page_numbers(built_index, monkeypatch, stub_llm):
    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    for source in rag.answer("asthma management", generate=stub_llm).sources:
        assert source["page"] >= 1
        assert source["snippet"]


def test_sources_are_deduplicated_by_page(built_index, monkeypatch, stub_llm):
    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    sources = rag.answer("vitamins", generate=stub_llm).sources
    assert len(sources) == len({s["page"] for s in sources})


def test_a_refusal_carries_no_sources(built_index, monkeypatch):
    """Listing passages under a refusal would imply the answer came from them."""
    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    result = rag.answer("What did the FDA approve in March 2024?",
                        generate=lambda messages: prompts.REFUSAL)
    assert result.refused is True
    assert result.sources == []


def test_generation_time_is_measured_separately(built_index, monkeypatch):
    import time

    monkeypatch.setattr(rag, "get_index", lambda: built_index)

    def slow(messages):
        time.sleep(0.05)
        return "An answer."

    result = rag.answer("anything", generate=slow)
    assert result.generation_ms >= 45
    assert result.retrieval_ms < result.generation_ms


def test_to_dict_is_json_serialisable(built_index, monkeypatch, stub_llm):
    import json

    monkeypatch.setattr(rag, "get_index", lambda: built_index)
    payload = rag.answer("stroke signs", generate=stub_llm).to_dict()
    json.dumps(payload)
    assert set(payload) >= {"answer", "sources", "refused", "retrieval_ms", "total_ms"}
