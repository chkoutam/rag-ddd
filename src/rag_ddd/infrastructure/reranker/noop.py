from __future__ import annotations

from typing import List

from rag_ddd.domain.entities import Query, RetrievedChunk
from rag_ddd.domain.ports import Reranker


class NoOpReranker(Reranker):
    def rerank(self, query: Query, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        _ = query
        return chunks[:top_k]
