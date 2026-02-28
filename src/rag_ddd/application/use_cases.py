from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4
from typing import List

import opik

from rag_ddd.application.dto import (
    DeleteDocumentResult,
    IngestDocumentResult,
    QueryResult,
    UploadResult,
)
from rag_ddd.domain.entities import Answer, Document, Query, RetrievedChunk
from rag_ddd.domain.ports import (
    BlobStore,
    Cache,
    Chunker,
    ChunkStore,
    DocumentStore,
    EmbeddingModel,
    LLM,
    NLPEnricher,
    Reranker,
    VectorStore,
)


@dataclass
class RAGQueryUseCase:
    embedder: EmbeddingModel
    vector_store: VectorStore
    llm: LLM
    cache: Cache | None
    reranker: Reranker | None
    retrieval_top_k: int
    rerank_top_k: int

    @opik.track(name="rag_query")
    def execute(self, query_text: str) -> QueryResult:
        cache_key = f"rag:answer:{query_text}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return QueryResult(answer=Answer(text=cached, sources=[]))

        # Embed, retrieve, optionally rerank, then generate answer.
        query = Query(text=query_text)
        query_embedding = self.embedder.embed([query_text])[0]
        retrieved = self.vector_store.query(query_embedding, self.retrieval_top_k)

        if self.reranker:
            retrieved = self.reranker.rerank(query, retrieved, self.rerank_top_k)

        prompt = self._build_prompt(query_text, retrieved)
        answer_text = self.llm.generate(prompt)
        answer = Answer(text=answer_text, sources=retrieved)

        if self.cache:
            self.cache.set(cache_key, answer_text, ttl_seconds=3600)

        return QueryResult(answer=answer)

    def _build_prompt(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        context_blocks = "\n\n".join(
            f"[Source {i + 1}]\n{chunk.chunk.text}" for i, chunk in enumerate(retrieved)
        )
        return (
            "You are a helpful assistant. Use the context to answer the question. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_blocks}\n\nQuestion: {query}\nAnswer:"
        )


@dataclass
class UploadDocumentUseCase:
    blob_store: BlobStore
    document_store: DocumentStore
    bucket_prefix: str = "documents"
    chunk_size: int = 0
    chunk_overlap: int = 0

    @opik.track(name="upload_document")
    def execute(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UploadResult:
        now = datetime.now(timezone.utc).isoformat()
        # Idempotency guard: avoid duplicate uploads by checksum.
        checksum = sha256(content).hexdigest()
        existing = self.document_store.list({"checksum": checksum})
        if existing:
            doc = existing[0]
            return UploadResult(
                doc_id=str(doc.get("_id", "")),
                gcs_path=str(doc.get("gcs_path", "")),
                status="DUPLICATE",
            )
        storage_key = f"{self.bucket_prefix}/{uuid4()}/{filename}"
        gcs_path = self.blob_store.put(storage_key, content, content_type=content_type)
        doc_id = self.document_store.create(
            {
                "filename": filename,
                "gcs_path": gcs_path,
                "status": "UPLOADED",
                "metadata": metadata or {},
                "checksum": checksum,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_version": f"{self.chunk_size}:{self.chunk_overlap}",
                "created_at": now,
                "updated_at": now,
            }
        )
        return UploadResult(doc_id=doc_id, gcs_path=gcs_path, status="UPLOADED")


@dataclass
class IngestDocumentUseCase:
    document_store: DocumentStore
    chunker: Chunker
    embedder: EmbeddingModel
    vector_store: VectorStore
    chunk_store: ChunkStore | None = None
    nlp_enricher: NLPEnricher | None = None
    embedding_batch_size: int = 64
    chunk_size: int = 0
    chunk_overlap: int = 0

    @opik.track(name="ingest_document")
    def execute(self, doc_id: str) -> IngestDocumentResult:
        document = self.document_store.get(doc_id)
        if not document:
            raise ValueError(f"Document not found: {doc_id}")
        text = str(document.get("text", "")).strip()
        if not text:
            self.document_store.update(doc_id, {"status": "ERROR", "updated_at": self._now()})
            return IngestDocumentResult(doc_id=doc_id, chunks=0, status="ERROR")

        # NLP enrichment: extract entities, classify, summarize
        nlp_metadata: dict[str, object] = {}
        if self.nlp_enricher:
            nlp_result = self.nlp_enricher.enrich(text)
            nlp_metadata = {
                "nlp_category": nlp_result.category,
                "nlp_category_confidence": nlp_result.category_confidence,
                "nlp_entities": [
                    {"text": e.text, "label": e.label} for e in nlp_result.entities
                ],
                "nlp_title": nlp_result.title,
                "nlp_author": nlp_result.author,
                "nlp_key_concepts": nlp_result.key_concepts,
                "nlp_summary": nlp_result.summary,
            }
            self.document_store.update(doc_id, {
                "nlp_enrichment": nlp_metadata,
                "status": "ENRICHED",
                "updated_at": self._now(),
            })

        # Mark indexing in-progress, then chunk and embed.
        self.document_store.update(doc_id, {"status": "INDEXING", "updated_at": self._now()})
        doc_domain = self._to_domain_document(doc_id, document)
        # Inject NLP metadata into each chunk's metadata
        if nlp_metadata:
            from dataclasses import replace

            merged_meta = {**doc_domain.metadata, **{
                "category": nlp_metadata.get("nlp_category", ""),
                "key_concepts": nlp_metadata.get("nlp_key_concepts", []),
            }}
            doc_domain = replace(doc_domain, metadata=merged_meta)

        chunks = self.chunker.chunk([doc_domain])
        if not chunks:
            self.document_store.update(doc_id, {"status": "ERROR", "updated_at": self._now()})
            return IngestDocumentResult(doc_id=doc_id, chunks=0, status="ERROR")
        if self.chunk_store:
            self.chunk_store.upsert(doc_id, chunks)

        self.document_store.update(
            doc_id,
            {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_version": f"{self.chunk_size}:{self.chunk_overlap}",
            },
        )
        chunk_count = self._index_chunks(doc_id, chunks)
        return IngestDocumentResult(doc_id=doc_id, chunks=chunk_count, status="INDEXED")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_domain_document(doc_id: str, document: dict[str, object]) -> Document:
        return Document(
            doc_id=doc_id,
            text=str(document.get("text", "")),
            metadata=dict(document.get("metadata") or {}),
        )

    def _index_chunks(self, doc_id: str, chunks: List[Chunk]) -> int:
        embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            end = start + self.embedding_batch_size
            batch_texts = [chunk.text for chunk in chunks[start:end]]
            embeddings.extend(self.embedder.embed(batch_texts))

        # Upsert into vector store and persist indexing status.
        self.vector_store.upsert(chunks, embeddings)
        self.document_store.update(
            doc_id,
            {
                "status": "INDEXED",
                "chunk_count": len(chunks),
                "updated_at": self._now(),
            },
        )
        return len(chunks)


@dataclass
class ReindexDocumentUseCase:
    vector_store: VectorStore
    ingest_use_case: IngestDocumentUseCase

    @opik.track(name="reindex_document")
    def execute(self, doc_id: str) -> IngestDocumentResult:
        if self.ingest_use_case.chunk_store:
            chunks = self.ingest_use_case.chunk_store.list_by_doc_id(doc_id)
            if chunks:
                # Fast path: re-embed from stored chunks.
                self.vector_store.delete_by_doc_id(doc_id)
                chunk_count = self.ingest_use_case._index_chunks(doc_id, chunks)
                return IngestDocumentResult(doc_id=doc_id, chunks=chunk_count, status="INDEXED")
        self.vector_store.delete_by_doc_id(doc_id)
        return self.ingest_use_case.execute(doc_id)


@dataclass
class DeleteDocumentUseCase:
    document_store: DocumentStore
    blob_store: BlobStore
    vector_store: VectorStore
    chunk_store: ChunkStore | None = None

    @opik.track(name="delete_document")
    def execute(self, doc_id: str) -> DeleteDocumentResult:
        document = self.document_store.get(doc_id)
        if not document:
            return DeleteDocumentResult(doc_id=doc_id, status="NOT_FOUND")
        gcs_path = str(document.get("gcs_path", "")).strip()
        if gcs_path:
            self.blob_store.delete(gcs_path)
        # Ensure all downstream stores are cleaned up.
        self.vector_store.delete_by_doc_id(doc_id)
        if self.chunk_store:
            self.chunk_store.delete_by_doc_id(doc_id)
        self.document_store.delete(doc_id)
        return DeleteDocumentResult(doc_id=doc_id, status="DELETED")


@dataclass
class RechunkDocumentUseCase:
    document_store: DocumentStore
    chunker: Chunker
    chunk_store: ChunkStore
    vector_store: VectorStore
    embedder: EmbeddingModel
    embedding_batch_size: int = 64
    chunk_size: int = 0
    chunk_overlap: int = 0

    @opik.track(name="rechunk_document")
    def execute(self, doc_id: str) -> IngestDocumentResult:
        document = self.document_store.get(doc_id)
        if not document:
            raise ValueError(f"Document not found: {doc_id}")
        text = str(document.get("text", "")).strip()
        if not text:
            self.document_store.update(doc_id, {"status": "ERROR", "updated_at": self._now()})
            return IngestDocumentResult(doc_id=doc_id, chunks=0, status="ERROR")

        # Rebuild chunks and reindex embeddings.
        self.document_store.update(doc_id, {"status": "RECHUNKING", "updated_at": self._now()})
        chunks = self.chunker.chunk([self._to_domain_document(doc_id, document)])
        if not chunks:
            self.document_store.update(doc_id, {"status": "ERROR", "updated_at": self._now()})
            return IngestDocumentResult(doc_id=doc_id, chunks=0, status="ERROR")

        self.chunk_store.upsert(doc_id, chunks)
        self.document_store.update(
            doc_id,
            {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_version": f"{self.chunk_size}:{self.chunk_overlap}",
            },
        )
        self.vector_store.delete_by_doc_id(doc_id)
        self._index_chunks(doc_id, chunks)
        return IngestDocumentResult(doc_id=doc_id, chunks=len(chunks), status="INDEXED")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_domain_document(doc_id: str, document: dict[str, object]) -> Document:
        return Document(
            doc_id=doc_id,
            text=str(document.get("text", "")),
            metadata=dict(document.get("metadata") or {}),
        )

    def _index_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            end = start + self.embedding_batch_size
            batch_texts = [chunk.text for chunk in chunks[start:end]]
            embeddings.extend(self.embedder.embed(batch_texts))

        # Replace vectors and update document status.
        self.vector_store.upsert(chunks, embeddings)
        self.document_store.update(
            doc_id,
            {
                "status": "INDEXED",
                "chunk_count": len(chunks),
                "updated_at": self._now(),
            },
        )
