from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Mapping, Sequence

from rag_ddd.domain.entities import Chunk, Document, NLPResult, Query, RetrievedChunk

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        raise NotImplementedError

class Chunker(ABC):
    @abstractmethod
    def chunk(self, documents: Iterable[Document]) -> List[Chunk]:
        raise NotImplementedError
    
class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: Iterable[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: Query, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        raise NotImplementedError


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise NotImplementedError


class DocumentStore(ABC):
    @abstractmethod
    def create(self, document: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, doc_id: str, patch: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, doc_id: str) -> Mapping[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, filters: Mapping[str, Any] | None = None) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        raise NotImplementedError


class ChunkStore(ABC):
    @abstractmethod
    def upsert(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_by_doc_id(self, doc_id: str) -> List[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError


class BlobStore(ABC):
    @abstractmethod
    def put(self, path: str, content: bytes, content_type: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, path: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        raise NotImplementedError


class NLPEnricher(ABC):
    @abstractmethod
    def enrich(self, text: str) -> NLPResult:
        raise NotImplementedError
