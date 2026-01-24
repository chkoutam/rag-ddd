from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_ddd.domain.ports import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        enable_tracing: bool = False,
        project_name: str | None = None,
    ) -> None:
        client = OpenAI(api_key=api_key)
        if enable_tracing:
            from rag_ddd.infrastructure.observability import get_tracked_openai_client
            client = get_tracked_openai_client(client, project_name=project_name)
        self._client = client
        self._model = model

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]
