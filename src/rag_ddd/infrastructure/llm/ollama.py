from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_ddd.domain.ports import LLM


class OllamaLLM(LLM):
    """LLM adapter for Ollama — uses the OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key="unused")
        self._model = model

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
