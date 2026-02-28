from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Provider selection ──────────────────────────────────────
    llm_provider: Literal["openai", "ollama", "vllm"] = "openai"
    embedder_provider: Literal["openai", "bge"] = "openai"
    blob_store_provider: Literal["gcs", "minio"] = "gcs"
    doc_store_provider: Literal["mongo"] = "mongo"
    chunker_type: Literal["recursive", "semantic"] = "recursive"
    parser_provider: Literal["pypdf", "docling"] = "pypdf"
    tracer_provider: Literal["opik", "langfuse", "none"] = "opik"
    nlp_enrichment: Literal["spacy", "none"] = "none"

    # ── OpenAI (cloud) ──────────────────────────────────────────
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4.1-mini"

    # ── Ollama (local, Apple Silicon) ───────────────────────────
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "mistral"

    # ── vLLM (GPU NVIDIA) ──────────────────────────────────────
    vllm_url: str = "http://localhost:8000/v1"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # ── BGE embeddings (local) ─────────────────────────────────
    bge_embedding_model: str = "BAAI/bge-m3"

    # ── Qdrant (vector store) ──────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_docs"

    # ── Redis (cache) ──────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── MongoDB (document store) ───────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "rag_ddd"
    mongodb_documents_collection: str = "documents"
    mongodb_chunks_collection: str = "chunks"

    # ── MinIO (self-hosted blob storage) ───────────────────────
    minio_url: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    # ── GCS (cloud blob storage) ───────────────────────────────
    gcs_bucket: str = "rag-ddd-documents"

    # ── NLP enrichment ─────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"
    nlp_max_summary_sentences: int = 5

    # ── Chunking ───────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 64

    # ── Retrieval & reranking ──────────────────────────────────
    retrieval_top_k: int = 6
    rerank_top_k: int = 3
    reranker_type: Literal["none", "embedding", "cross-encoder", "bge"] = "embedding"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None

    # ── Opik observability ─────────────────────────────────────
    opik_enabled: bool = True
    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_project_name: str = "rag-ddd"
    opik_use_local: bool = False

    # ── LangFuse observability (self-hosted) ───────────────────
    langfuse_url: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ── Backward compatibility aliases ─────────────────────────
    @property
    def embedding_model(self) -> str:
        if self.embedder_provider == "openai":
            return self.openai_embedding_model
        return self.bge_embedding_model

    @property
    def llm_model(self) -> str:
        match self.llm_provider:
            case "openai":
                return self.openai_llm_model
            case "ollama":
                return self.ollama_model
            case "vllm":
                return self.vllm_model
