"""Thin wrapper around the Groq chat completions API used by every agent."""

import json
import re
from typing import Optional

from groq import Groq

from config import get_groq_api_key

_client: Optional[Groq] = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=get_groq_api_key())
    return _client


def chat(model: str, system: str, user: str, temperature: float = 0.3) -> str:
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def chat_json(model: str, system: str, user: str) -> dict:
    """Call the model and parse a JSON object out of its reply.

    Small models occasionally wrap JSON in markdown fences, so we strip those
    and fall back to a regex search for the first {...} block.
    """
    raw = chat(model, system + "\nRespond ONLY with a valid JSON object.", user, 0.0)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
