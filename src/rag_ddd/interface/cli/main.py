from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from rag_ddd.application.use_cases import (
    DeleteDocumentUseCase,
    IngestDocumentUseCase,
    RAGQueryUseCase,
    RechunkDocumentUseCase,
    ReindexDocumentUseCase,
    UploadDocumentUseCase,
)
from rag_ddd.config.settings import Settings
from rag_ddd.infrastructure.blob_store.gcs import GCSBlobStore
from rag_ddd.infrastructure.cache.redis_cache import RedisCache
from rag_ddd.infrastructure.chunk_store.mongo import MongoChunkStore
from rag_ddd.infrastructure.chunking.recursive import RecursiveChunker
from rag_ddd.infrastructure.document_store.mongo import MongoDocumentStore
from rag_ddd.infrastructure.embeddings.openai import OpenAIEmbeddingModel
from rag_ddd.infrastructure.llm.openai import OpenAILLM
from rag_ddd.infrastructure.loaders.file_loader import FileLoader
from rag_ddd.infrastructure.observability import configure_opik
from rag_ddd.infrastructure.reranker.noop import NoOpReranker
from rag_ddd.infrastructure.vector_store.qdrant import QdrantVectorStore

app = typer.Typer(add_completion=False)
settings = Settings()

# Configure Opik observability
if settings.opik_enabled:
    configure_opik(
        api_key=settings.opik_api_key,
        workspace=settings.opik_workspace,
        project_name=settings.opik_project_name,
        use_local=settings.opik_use_local,
    )


def build_query_use_case() -> RAGQueryUseCase:
    embedder = OpenAIEmbeddingModel(
        settings.openai_api_key,
        settings.embedding_model,
        enable_tracing=settings.opik_enabled,
        project_name=settings.opik_project_name,
    )
    llm = OpenAILLM(
        settings.openai_api_key,
        settings.llm_model,
        enable_tracing=settings.opik_enabled,
        project_name=settings.opik_project_name,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    cache = RedisCache(settings.redis_url)
    reranker = NoOpReranker()
    query_use_case = RAGQueryUseCase(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        cache=cache,
        reranker=reranker,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )
    return query_use_case


def build_document_use_cases() -> tuple[
    UploadDocumentUseCase,
    IngestDocumentUseCase,
    ReindexDocumentUseCase,
    DeleteDocumentUseCase,
    RechunkDocumentUseCase,
    FileLoader,
]:
    loader = FileLoader(allowed_extensions={".pdf", ".txt", ".md"}, progress=True)
    blob_store = GCSBlobStore(settings.gcs_bucket)
    document_store = MongoDocumentStore(
        uri=settings.mongodb_uri,
        database=settings.mongodb_database,
        collection=settings.mongodb_documents_collection,
    )
    chunk_store = MongoChunkStore(
        uri=settings.mongodb_uri,
        database=settings.mongodb_database,
        collection=settings.mongodb_chunks_collection,
    )
    chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap, progress=True)
    embedder = OpenAIEmbeddingModel(
        settings.openai_api_key,
        settings.embedding_model,
        enable_tracing=settings.opik_enabled,
        project_name=settings.opik_project_name,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    upload_use_case = UploadDocumentUseCase(
        blob_store=blob_store,
        document_store=document_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    ingest_use_case = IngestDocumentUseCase(
        document_store=document_store,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        chunk_store=chunk_store,
        embedding_batch_size=settings.embedding_batch_size,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    reindex_use_case = ReindexDocumentUseCase(
        vector_store=vector_store,
        ingest_use_case=ingest_use_case,
    )
    delete_use_case = DeleteDocumentUseCase(
        document_store=document_store,
        blob_store=blob_store,
        vector_store=vector_store,
        chunk_store=chunk_store,
    )
    rechunk_use_case = RechunkDocumentUseCase(
        document_store=document_store,
        chunker=chunker,
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedder=embedder,
        embedding_batch_size=settings.embedding_batch_size,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return upload_use_case, ingest_use_case, reindex_use_case, delete_use_case, rechunk_use_case, loader


@app.command()
def query(text: str) -> None:
    query_use_case = build_query_use_case()
    result = query_use_case.execute(text)
    print(result.answer.text)
    if result.answer.sources:
        print("\nSources:")
        for item in result.answer.sources:
            print(f"- {item.chunk.metadata.get('source', '')} (score: {item.score:.3f})")


@app.command()
def upload(path: str) -> None:
    upload_use_case, _, _, _, _, loader = build_document_use_cases()
    documents = loader.load(path)
    if not documents:
        print("No documents found.")
        return

    for document in documents:
        source_path = Path(str(document.metadata.get("source", "")))
        if not source_path.exists():
            print(f"Skipped missing file: {source_path}")
            continue
        content_type = _content_type_for_path(source_path)
        result = upload_use_case.execute(
            filename=source_path.name,
            content=source_path.read_bytes(),
            content_type=content_type,
            metadata=document.metadata,
        )
        _update_document_text(result.doc_id, document.text)
        print(f"Uploaded {source_path.name} -> {result.doc_id}")


@app.command("ingest-doc")
def ingest_document(doc_id: str) -> None:
    _, ingest_use_case, _, _, _, _ = build_document_use_cases()
    result = ingest_use_case.execute(doc_id)
    print(f"Ingested {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("reindex-doc")
def reindex_document(doc_id: str) -> None:
    _, _, reindex_use_case, _, _, _ = build_document_use_cases()
    result = reindex_use_case.execute(doc_id)
    print(f"Reindexed {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("delete-doc")
def delete_document(doc_id: str) -> None:
    _, _, _, delete_use_case, _, _ = build_document_use_cases()
    result = delete_use_case.execute(doc_id)
    print(f"Delete {result.doc_id} (status={result.status})")


@app.command("rechunk-doc")
def rechunk_document(doc_id: str) -> None:
    _, _, _, _, rechunk_use_case, _ = build_document_use_cases()
    result = rechunk_use_case.execute(doc_id)
    print(f"Rechunked {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("list-docs")
def list_documents(status: str | None = None) -> None:
    document_store = _document_store()
    filters: dict[str, object] = {}
    if status:
        filters["status"] = status
    documents = document_store.list(filters if filters else None)
    if not documents:
        print("No documents found.")
        return
    for doc in documents:
        print(f"{doc.get('_id')} {doc.get('status', '')} {doc.get('filename', '')}")


@app.command("doc-status")
def document_status(doc_id: str) -> None:
    document_store = _document_store()
    document = document_store.get(doc_id)
    if not document:
        print(f"Document not found: {doc_id}")
        return
    print(f"{document.get('_id')} status={document.get('status', '')}")


def _content_type_for_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".md":
        return "text/markdown"
    return None


def _update_document_text(doc_id: str, text: str) -> None:
    document_store = _document_store()
    document_store.update(doc_id, {"text": text, "status": "READY"})


def _document_store() -> MongoDocumentStore:
    return MongoDocumentStore(
        uri=settings.mongodb_uri,
        database=settings.mongodb_database,
        collection=settings.mongodb_documents_collection,
    )


if __name__ == "__main__":
    app()
