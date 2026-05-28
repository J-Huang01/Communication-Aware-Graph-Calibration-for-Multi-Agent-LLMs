from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class BackboneConfig:
    name: str            # short id used as agent prefix, e.g. "qwen3-8b"
    base_url: str
    model: str           # model id the server expects
    api_key: str
    is_local: bool       # True = local vLLM (logprobs supported)


# --- registry ---------------------------------------------------------------
# Edit base_url / model fields to match your deployment.

REGISTRY: dict[str, BackboneConfig] = {
    "qwen3-8b": BackboneConfig(
        name="qwen3-8b",
        base_url=os.environ.get("VLLM_QWEN_URL", "http://localhost:8001/v1"),
        model="qwen3-8b",
        api_key="EMPTY",
        is_local=True,
    ),
    "llama-3.1-8b": BackboneConfig(
        name="llama-3.1-8b",
        base_url=os.environ.get("VLLM_LLAMA_URL", "http://localhost:8002/v1"),
        model="llama-3.1-8b",
        api_key="EMPTY",
        is_local=True,
    ),
    "gemma-3-12b": BackboneConfig(
        name="gemma-3-12b",
        base_url=os.environ.get("VLLM_GEMMA_URL", "http://localhost:8003/v1"),
        model="gemma-3-12b",
        api_key="EMPTY",
        is_local=True,
    ),
    "phi-4": BackboneConfig(
        name="phi-4",
        base_url=os.environ.get("VLLM_PHI_URL", "http://localhost:8004/v1"),
        model="phi-4",
        api_key="EMPTY",
        is_local=True,
    ),
    # Frontier APIs (>20B always API). Route via OpenRouter where supported
    # (single key, single billing line, no logprobs from frontier anyway).
    "llama-3.3-70b": BackboneConfig(
        name="llama-3.3-70b",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        is_local=False,
    ),
    "qwen3.6-27b": BackboneConfig(
        name="qwen3.6-27b",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-27b",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        is_local=False,
    ),
}


def get_client(backbone: str) -> tuple[OpenAI, BackboneConfig]:
    cfg = REGISTRY[backbone]
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key), cfg


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat(
    backbone: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    return_logprobs: bool = False,
    top_logprobs: int = 20,
) -> dict[str, Any]:
    """Single Chat Completions call. Returns a dict with 'text' and (optionally) 'logprobs'.

    For Qwen3-* (hybrid thinking) we explicitly disable the <think>...</think>
    block so Qwen3's behaviour is uniform with the other backbones.
    """
    client, cfg = get_client(backbone)
    kwargs: dict[str, Any] = dict(
        model=cfg.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if return_logprobs and cfg.is_local:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs
    # Qwen3 hybrid: disable thinking mode for cross-backbone uniformity.
    # Local vLLM uses chat_template_kwargs; OpenRouter exposes a
    # `reasoning` field. Both Qwen3 (local) and Qwen3.6 (OpenRouter)
    # default to thinking, so we have to turn it off explicitly or
    # token budgets get eaten by reasoning traces.
    if backbone.startswith("qwen3-"):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif backbone.startswith("qwen3.6"):
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    out: dict[str, Any] = {"text": text, "model": cfg.model, "backbone": backbone}
    if return_logprobs and cfg.is_local and choice.logprobs is not None:
        out["logprobs"] = [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top": [{"token": x.token, "logprob": x.logprob} for x in (t.top_logprobs or [])],
            }
            for t in choice.logprobs.content or []
        ]
    return out
