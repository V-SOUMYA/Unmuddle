"""A small wrapper around the LLM, so the rest of the code never talks to an API directly.

Every call asks for JSON back. Tests use FakeLLM, so they run without a key or network.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Protocol

from unmuddle.config import LLM_MODEL


class LLM(Protocol):
    def json(self, prompt: str) -> Any:
        """Send a prompt and return the parsed JSON reply."""
        ...


class GeminiLLM:
    """Calls Google Gemini and parses its JSON reply."""

    def __init__(self, model: str = LLM_MODEL, api_key: str | None = None, retries: int = 3):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.retries = retries

    def json(self, prompt: str) -> Any:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                reply = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return json.loads(reply.text)
            except Exception as e:  # network errors, rate limits, bad JSON
                last_error = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} tries: {last_error}")


class FakeLLM:
    """Test double: returns whatever `respond(prompt)` returns, and records prompts."""

    def __init__(self, respond: Callable[[str], Any]):
        self.respond = respond
        self.prompts: list[str] = []

    def json(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.respond(prompt)
