from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise LLMRequestError("The model returned an invalid JSON response.")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMRequestError("The model returned an invalid JSON response.") from exc
    if not isinstance(value, dict):
        raise LLMRequestError("The model response was JSON, but not an object.")
    return value


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
        self.model = model or (
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            if self.provider == "groq"
            else os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        self._groq = None
        self._gemini = None

    @property
    def configured(self) -> bool:
        if self.provider == "groq":
            return bool(os.getenv("GROQ_API_KEY"))
        if self.provider == "gemini":
            return bool(os.getenv("GEMINI_API_KEY"))
        return False

    def _ensure_client(self) -> None:
        if self.provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise LLMConfigurationError("GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")
            if self._groq is None:
                from groq import Groq
                self._groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        elif self.provider == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise LLMConfigurationError("GEMINI_API_KEY is missing. Add it to .env or Streamlit secrets.")
            if self._gemini is None:
                from google import genai
                self._gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            raise LLMConfigurationError(f"Unsupported LLM_PROVIDER '{self.provider}'. Use groq or gemini.")

    def generate(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1600, temperature: float = 0.2) -> str:
        self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                if self.provider == "groq":
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "temperature": temperature,
                        "max_completion_tokens": max_tokens,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = self._groq.chat.completions.create(**kwargs)
                    return response.choices[0].message.content or ""
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json" if json_mode else "text/plain",
                )
                response = self._gemini.models.generate_content(model=self.model, contents=user, config=config)
                return response.text or ""
            except Exception as exc:  # Provider SDK exceptions vary by version.
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (2**attempt))
        raise LLMRequestError(
            "The model request failed after retries. Check the API key, model name, quota, and network connection."
        ) from last_error

    def generate_json(self, system: str, user: str, *, max_tokens: int = 1600) -> dict[str, Any]:
        return _extract_json(self.generate(system, user, json_mode=True, max_tokens=max_tokens))
