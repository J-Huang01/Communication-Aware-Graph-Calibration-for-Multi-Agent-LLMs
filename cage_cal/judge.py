from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


# --- provider routing -------------------------------------------------------

_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-haiku-4.5",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
        # Kept for fallback only; default A2 judge is Llama-3.3-70B on OpenRouter.
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
    },
}


@lru_cache(maxsize=1)
def _get_judge_client() -> tuple[OpenAI, str]:
    provider = os.environ.get("JUDGE_PROVIDER", "openrouter").lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown JUDGE_PROVIDER {provider!r}; choose from {list(_PROVIDERS)}")
    cfg = _PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"], "")
    if not key:
        raise EnvironmentError(
            f"JUDGE_PROVIDER={provider} but {cfg['key_env']} is empty. "
            f"export {cfg['key_env']}=... before running."
        )
    model = os.environ.get("JUDGE_MODEL", cfg["default_model"])
    client = OpenAI(base_url=cfg["base_url"], api_key=key)
    return client, model


# --- prompt -----------------------------------------------------------------

_SYSTEM = (
    "You are an answer-equivalence judge. Given a question, a gold answer, "
    "and a predicted answer, decide whether the predicted answer is correct "
    "(semantically equivalent to the gold answer) for the question. "
    "Be strict on factual correctness but lenient on surface form (case, "
    "articles, trailing punctuation, paraphrase).\n"
    "Respond with strict JSON, no prose: "
    '{"correct": true|false, "reason": "<one short clause>"}'
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def judge_one(question: str, gold: str, pred: str) -> bool:
    if not pred:
        return False
    # Cheap fast path: case-insensitive exact match avoids an API call.
    if gold.strip().lower() == pred.strip().lower():
        return True
    client, model = _get_judge_client()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=120,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",
             "content": f"Question: {question}\nGold: {gold}\nPredicted: {pred}\n\nReturn JSON only."},
        ],
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return False
    try:
        return bool(json.loads(m.group(0)).get("correct", False))
    except json.JSONDecodeError:
        return False


def judge_batch(items: Iterable[tuple[str, str, str]]) -> list[bool]:
    """items: iterable of (question, gold, predicted)."""
    return [judge_one(q, g, p) for q, g, p in items]
