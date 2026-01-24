from __future__ import annotations

from typing import Sequence

from pymongo import MongoClient

from rag_ddd.domain.entities import Chunk
from rag_ddd.domain.ports import ChunkStore


class MongoChunkStore(ChunkStore):
    def __init__(self, uri: str, database: str, collection: str = "chunks") -> None:
        self._client = MongoClient(uri)
        self._collection = self._client[database][collection]

    def upsert(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        self.delete_by_doc_id(doc_id)
        if not chunks:
            return
        documents = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self._collection.insert_many(documents)

    def list_by_doc_id(self, doc_id: str) -> list[Chunk]:
        cursor = self._collection.find({"doc_id": doc_id})
        return [
            Chunk(
                chunk_id=str(doc.get("chunk_id", "")),
                doc_id=str(doc.get("doc_id", "")),
                text=str(doc.get("text", "")),
                metadata=doc.get("metadata", {}) or {},
            )
            for doc in cursor
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._collection.delete_many({"doc_id": doc_id})
