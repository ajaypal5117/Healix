"""OpenAI wrapper.

Imported lazily so that indexing, retrieval, the benchmark and the whole test
suite run without the openai package or an API key. Only generation needs them.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.require_api_key(), timeout=settings.llm_timeout)


def complete(messages: list[dict]) -> str:
    response = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
