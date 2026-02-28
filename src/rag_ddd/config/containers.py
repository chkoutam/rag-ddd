"""Composition root — wires ports to adapters based on settings.

Each build_* function reads the provider from Settings and returns
the correct adapter implementation. The use cases and interfaces
never import adapters directly; they call these factories instead.
"""

from __future__ import annotations

from rag_ddd.config.settings import Settings
from rag_ddd.domain.ports import (
    BlobStore,
    Cache,
    ChunkStore,
    Chunker,
    DocumentLoader,
    DocumentStore,
    EmbeddingModel,
    LLM,
    NLPEnricher,
    Reranker,
    VectorStore,
)


# ── LLM ────────────────────────────────────────────────────────


def build_llm(settings: Settings) -> LLM:
    match settings.llm_provider:
        case "openai":
            from rag_ddd.infrastructure.llm.openai import OpenAILLM

            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            return OpenAILLM(
                api_key=settings.openai_api_key,
                model=settings.openai_llm_model,
                enable_tracing=settings.tracer_provider == "opik" and settings.opik_enabled,
                project_name=settings.opik_project_name,
            )
        case "ollama":
            from rag_ddd.infrastructure.llm.ollama import OllamaLLM

            return OllamaLLM(
                base_url=settings.ollama_url,
                model=settings.ollama_model,
            )
        case "vllm":
            from rag_ddd.infrastructure.llm.vllm import VllmLLM

            return VllmLLM(
                base_url=settings.vllm_url,
                model=settings.vllm_model,
            )
        case _:
            raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


# ── Embeddings ─────────────────────────────────────────────────


def build_embedder(settings: Settings) -> EmbeddingModel:
    match settings.embedder_provider:
        case "openai":
            from rag_ddd.infrastructure.embeddings.openai import OpenAIEmbeddingModel

            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDER_PROVIDER=openai")
            return OpenAIEmbeddingModel(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                enable_tracing=settings.tracer_provider == "opik" and settings.opik_enabled,
                project_name=settings.opik_project_name,
            )
        case "bge":
            from rag_ddd.infrastructure.embeddings.bge import BGEEmbedder

            return BGEEmbedder(model_name=settings.bge_embedding_model)
        case _:
            raise ValueError(f"Unknown EMBEDDER_PROVIDER: {settings.embedder_provider}")


# ── Reranker ───────────────────────────────────────────────────


def build_reranker(settings: Settings, embedder: EmbeddingModel) -> Reranker | None:
    match settings.reranker_type:
        case "none":
            return None
        case "embedding":
            from rag_ddd.infrastructure.reranker.embedding import EmbeddingReranker

            return EmbeddingReranker(embedder)
        case "cross-encoder":
            from rag_ddd.infrastructure.reranker.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
            )
        case "bge":
            from rag_ddd.infrastructure.reranker.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(
                model_name=settings.bge_reranker_model,
                device=settings.reranker_device,
            )
        case _:
            raise ValueError(f"Unknown RERANKER_TYPE: {settings.reranker_type}")


# ── Vector Store ───────────────────────────────────────────────


def build_vector_store(settings: Settings) -> VectorStore:
    from rag_ddd.infrastructure.vector_store.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )


# ── Cache ──────────────────────────────────────────────────────


def build_cache(settings: Settings) -> Cache:
    from rag_ddd.infrastructure.cache.redis_cache import RedisCache

    return RedisCache(settings.redis_url)


# ── Document Store ─────────────────────────────────────────────


def build_document_store(settings: Settings) -> DocumentStore:
    match settings.doc_store_provider:
        case "mongo":
            from rag_ddd.infrastructure.document_store.mongo import MongoDocumentStore

            return MongoDocumentStore(
                uri=settings.mongodb_uri,
                database=settings.mongodb_database,
                collection=settings.mongodb_documents_collection,
            )
        case _:
            raise ValueError(f"Unknown DOC_STORE_PROVIDER: {settings.doc_store_provider}")


# ── Chunk Store ────────────────────────────────────────────────


def build_chunk_store(settings: Settings) -> ChunkStore:
    match settings.doc_store_provider:
        case "mongo":
            from rag_ddd.infrastructure.chunk_store.mongo import MongoChunkStore

            return MongoChunkStore(
                uri=settings.mongodb_uri,
                database=settings.mongodb_database,
                collection=settings.mongodb_chunks_collection,
            )
        case _:
            raise ValueError(f"Unknown DOC_STORE_PROVIDER: {settings.doc_store_provider}")


# ── Blob Store ─────────────────────────────────────────────────


def build_blob_store(settings: Settings) -> BlobStore:
    match settings.blob_store_provider:
        case "gcs":
            from rag_ddd.infrastructure.blob_store.gcs import GCSBlobStore

            return GCSBlobStore(bucket=settings.gcs_bucket)
        case "minio":
            from rag_ddd.infrastructure.blob_store.minio import MinIOBlobStore

            return MinIOBlobStore(
                url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket=settings.minio_bucket,
                secure=settings.minio_secure,
            )
        case _:
            raise ValueError(f"Unknown BLOB_STORE_PROVIDER: {settings.blob_store_provider}")


# ── Chunker ────────────────────────────────────────────────────


def build_chunker(settings: Settings, progress: bool = False) -> Chunker:
    match settings.chunker_type:
        case "recursive":
            from rag_ddd.infrastructure.chunking.recursive import RecursiveChunker

            return RecursiveChunker(
                settings.chunk_size,
                settings.chunk_overlap,
                progress=progress,
            )
        case "semantic":
            from rag_ddd.infrastructure.chunking.semantic import SemanticChunker

            return SemanticChunker(
                embedding_model_name=settings.bge_embedding_model,
            )
        case _:
            raise ValueError(f"Unknown CHUNKER_TYPE: {settings.chunker_type}")


# ── Document Loader ────────────────────────────────────────────


def build_loader(settings: Settings, progress: bool = False) -> DocumentLoader:
    match settings.parser_provider:
        case "pypdf":
            from rag_ddd.infrastructure.loaders.file_loader import FileLoader

            return FileLoader(
                allowed_extensions={".pdf", ".txt", ".md"},
                progress=progress,
            )
        case "docling":
            from rag_ddd.infrastructure.loaders.docling_loader import DoclingLoader

            return DoclingLoader(progress=progress)
        case _:
            raise ValueError(f"Unknown PARSER_PROVIDER: {settings.parser_provider}")


# ── NLP Enricher ──────────────────────────────────────────────


def build_nlp_enricher(settings: Settings) -> NLPEnricher:
    match settings.nlp_enrichment:
        case "spacy":
            from rag_ddd.infrastructure.nlp.enrichment_pipeline import SpacyNLPEnricher

            return SpacyNLPEnricher(
                spacy_model=settings.spacy_model,
                max_summary_sentences=settings.nlp_max_summary_sentences,
            )
        case "none":
            from rag_ddd.infrastructure.nlp.enrichment_pipeline import NoOpNLPEnricher

            return NoOpNLPEnricher()
        case _:
            raise ValueError(f"Unknown NLP_ENRICHMENT: {settings.nlp_enrichment}")


# ── Observability ──────────────────────────────────────────────


def configure_tracing(settings: Settings) -> None:
    match settings.tracer_provider:
        case "opik":
            if settings.opik_enabled:
                from rag_ddd.infrastructure.observability import configure_opik

                configure_opik(
                    api_key=settings.opik_api_key,
                    workspace=settings.opik_workspace,
                    project_name=settings.opik_project_name,
                    use_local=settings.opik_use_local,
                )
        case "langfuse":
            pass  # LangFuse adapter will be added in a later step
        case "none":
            pass


# ── Composite builders (use cases) ─────────────────────────────


def build_query_use_case(settings: Settings) -> "RAGQueryUseCase":
    from rag_ddd.application.use_cases import RAGQueryUseCase

    embedder = build_embedder(settings)
    return RAGQueryUseCase(
        embedder=embedder,
        vector_store=build_vector_store(settings),
        llm=build_llm(settings),
        cache=build_cache(settings),
        reranker=build_reranker(settings, embedder),
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )


def build_document_use_cases(
    settings: Settings,
) -> tuple[
    "UploadDocumentUseCase",
    "IngestDocumentUseCase",
    "ReindexDocumentUseCase",
    "DeleteDocumentUseCase",
    "RechunkDocumentUseCase",
    "DocumentLoader",
]:
    from rag_ddd.application.use_cases import (
        DeleteDocumentUseCase,
        IngestDocumentUseCase,
        RechunkDocumentUseCase,
        ReindexDocumentUseCase,
        UploadDocumentUseCase,
    )

    loader = build_loader(settings, progress=True)
    blob_store = build_blob_store(settings)
    document_store = build_document_store(settings)
    chunk_store = build_chunk_store(settings)
    chunker = build_chunker(settings, progress=True)
    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)
    nlp_enricher = build_nlp_enricher(settings)

    upload_uc = UploadDocumentUseCase(
        blob_store=blob_store,
        document_store=document_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    ingest_uc = IngestDocumentUseCase(
        document_store=document_store,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        chunk_store=chunk_store,
        nlp_enricher=nlp_enricher,
        embedding_batch_size=settings.embedding_batch_size,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    reindex_uc = ReindexDocumentUseCase(
        vector_store=vector_store,
        ingest_use_case=ingest_uc,
    )
    delete_uc = DeleteDocumentUseCase(
        document_store=document_store,
        blob_store=blob_store,
        vector_store=vector_store,
        chunk_store=chunk_store,
    )
    rechunk_uc = RechunkDocumentUseCase(
        document_store=document_store,
        chunker=chunker,
        chunk_store=chunk_store,
        vector_store=vector_store,
        embedder=embedder,
        embedding_batch_size=settings.embedding_batch_size,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return upload_uc, ingest_uc, reindex_uc, delete_uc, rechunk_uc, loader
