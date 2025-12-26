"""
Simple HTTP client for interacting with a local LM Studio server.

Uses LM Studio's OpenAI-compatible endpoints:
  POST {BASE_URL}/chat/completions
Where BASE_URL should include /v1, e.g. http://127.0.0.1:1234/v1
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests


class LMStudioClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        # Expect base_url like http://127.0.0.1:1234/v1
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        # path should start with /chat/completions
        return f"{self.base_url}{path}"

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        timeout_s: int = 600,
        **kwargs: Any,
    ) -> str:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        r = requests.post(
            self._url("/chat/completions"),
            headers=self._headers(),
            json=body,
            timeout=(10, timeout_s),
        )
        r.raise_for_status()
        data = r.json()

        # Helpful error if LM Studio returns {"error": ...}
        if "choices" not in data:
            raise RuntimeError(f"LM Studio returned unexpected response: {data}")

        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.6,
        timeout_s: int = 600,
        **kwargs: Any,
    ) -> Iterable[str]:
        """
        True streaming via Server-Sent Events.
        Yields text fragments as they arrive.
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        with requests.post(
            self._url("/chat/completions"),
            headers=self._headers(),
            json=body,
            stream=True,
            timeout=(10, timeout_s),
        ) as r:
            r.raise_for_status()

            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue

                # LM Studio/OpenAI-style SSE lines often look like: "data: {...}"
                line = raw
                if line.startswith("data: "):
                    line = line[6:]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                if "choices" not in chunk:
                    continue

                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
