from __future__ import annotations

from typing import List, Sequence

from rag_ddd.domain.ports import EmbeddingModel


class BGEEmbedder(EmbeddingModel):
    """Local embedding adapter using sentence-transformers (BGE-M3).

    The model is loaded lazily on first call to avoid slow import at startup.
    Runs entirely on-device — no data leaves the machine.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(list(texts), normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
