from __future__ import annotations

from typing import Iterable, List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_ddd.domain.entities import Chunk, RetrievedChunk
from rag_ddd.domain.ports import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, api_key: str | None, collection: str) -> None:
        self._collection = collection
        self._client = QdrantClient(url=url, api_key=api_key)
        self._vector_size: int | None = None

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        if self._vector_size is None:
            raise ValueError("Vector size is unknown; cannot create collection.")
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, chunks: Iterable[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        embeddings_list = list(embeddings)
        chunks_list = list(chunks)
        if not chunks_list:
            return
        if self._vector_size is None:
            self._vector_size = len(embeddings_list[0])
        self.ensure_collection()

        points = []
        for chunk, vector in zip(chunks_list, embeddings_list):
            payload = {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            points.append(models.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection, points=points)

    def query(self, embedding: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        if not self._client.collection_exists(self._collection):
            return []
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(embedding),
                limit=top_k,
                with_payload=True,
            )
            results = response.points
        else:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=list(embedding),
                limit=top_k,
            )
        retrieved: List[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=str(point.id),
                doc_id=str(payload.get("doc_id", "")),
                text=str(payload.get("text", "")),
                metadata=payload.get("metadata", {}),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=float(point.score or 0)))
        return retrieved

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )
