"""The prompt is the hallucination control, so its structure is pinned here."""

from healix import prompts


def test_system_prompt_restricts_to_provided_passages():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "only" in text and "passages" in text


def test_system_prompt_states_the_exact_refusal_string():
    assert prompts.REFUSAL in prompts.SYSTEM_PROMPT


def test_system_prompt_caps_length_at_three_sentences():
    assert "three sentences" in prompts.SYSTEM_PROMPT.lower()


def test_system_prompt_forbids_invented_figures():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "never invent" in text
    assert "dosages" in text or "dosage" in text


def test_context_is_labelled_with_page_numbers(built_index):
    hits = built_index.search("wound healing", k=3)
    context = prompts.format_context(hits)
    assert "[page" in context
    assert str(hits[0].page) in context


def test_messages_carry_system_and_user_roles(built_index):
    messages = prompts.build_messages("What is asthma?", built_index.search("asthma", k=2))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "What is asthma?"


def test_retrieved_text_appears_in_the_system_message(built_index):
    hits = built_index.search("vitamins", k=2)
    messages = prompts.build_messages("q", hits)
    assert hits[0].text[:40] in messages[0]["content"]


def test_refusal_is_detected():
    assert prompts.is_refusal(prompts.REFUSAL)
    assert prompts.is_refusal(f"  {prompts.REFUSAL}  ")
    assert not prompts.is_refusal("Iron deficiency anemia is caused by blood loss.")


def test_sentence_counting():
    assert prompts.count_sentences("One. Two. Three.") == 3
    assert prompts.count_sentences("Only one sentence") == 1


def test_sentence_counting_ignores_abbreviations():
    """'e.g.' must not read as a sentence boundary or the length check is wrong."""
    assert prompts.count_sentences("Fat-soluble vitamins, e.g. A and D, are stored.") == 1
