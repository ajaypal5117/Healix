"""The constrained prompt.

Hallucination in a RAG system is mostly not the retriever's fault - it is the
model filling gaps with plausible medical-sounding prose. Four constraints do
the work here, and each one is testable:

1. The context is the only permitted source. Stated as an instruction, and
   reinforced by giving the model nothing else to work with.
2. An explicit refusal string for questions the corpus doesn't cover. Without a
   sanctioned way to say "not here", a model will answer anyway.
3. A three-sentence cap. Elaboration is where unsupported claims appear; there
   is no room for it in three sentences.
4. No invented specifics. Drug names, dosages and figures are the highest-risk
   tokens because they are the ones a reader would act on.

`REFUSAL` is matched exactly by the evaluation harness, so changing its wording
means changing it there too.
"""

REFUSAL = "That isn't covered in the encyclopedia I have access to."

SYSTEM_PROMPT = f"""You are Healix, a question-answering assistant for a single medical encyclopedia.

Answer using ONLY the passages provided below. They are your only source of truth.

Rules:
- If the passages do not contain the answer, reply with exactly this sentence and nothing else: "{REFUSAL}" Do not guess, and do not fall back on knowledge from outside the passages.
- At most three sentences. Be specific and skip preamble.
- Use the terminology as it appears in the passages rather than loosely paraphrasing clinical terms.
- Never invent drug names, dosages, measurements or figures. If a number is not in the passages, leave it out.
- This is reference information, not medical advice. If the question is about the reader's own symptoms, describe what the encyclopedia says and add that a clinician should be consulted.

Passages:
{{context}}"""

USER_PROMPT = "{question}"


def format_context(hits) -> str:
    """Render retrieved chunks with page labels so answers stay traceable."""
    blocks = []
    for hit in hits:
        blocks.append(f"[page {hit.page}]\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(hits))},
        {"role": "user", "content": USER_PROMPT.format(question=question)},
    ]


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower() in (answer or "").strip().lower()


def count_sentences(text: str) -> int:
    """Rough sentence count used by the tests and the grounding evaluation."""
    import re

    # Protect common abbreviations so "e.g." doesn't read as a sentence end.
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|Dr|Fig)\.", r"\1<DOT>", text or "")
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", protected) if p.strip()]
    return len(parts)
