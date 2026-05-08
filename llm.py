"""
llm.py — Thin wrapper around Ollama for text generation and embeddings.
"""

from __future__ import annotations

import json
import re
import time

import requests

import config


def complete(
    prompt: str,
    model: str = config.DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 600,
    retries: int = 2,
) -> str:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     4096,
        },
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2)
    return ""


def embed(text: str, model: str = config.EMBED_MODEL) -> list[float] | None:
    """Return a normalised embedding vector, or None if the embed model is unavailable."""
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception:
        return None


def rate_importance(description: str, model: str = config.DEFAULT_MODEL) -> float:
    """
    Ask the LLM to rate the poignancy / importance of a memory on a scale 1–10.
    Returns a float; falls back to 5.0 on parse failure.
    """
    prompt = (
        "On a scale of 1 to 10 (1=mundane, 10=extremely important/urgent), "
        "rate the importance of this event for a software development team:\n\n"
        f'"{description}"\n\n'
        "Reply with ONLY a single integer (1-10). No explanation."
    )
    raw = complete(prompt, model=model, temperature=0.1, max_tokens=5)
    digits = re.findall(r"\d+", raw)
    if digits:
        val = int(digits[0])
        return float(max(1, min(10, val)))
    return 5.0
