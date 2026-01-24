from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_docs"
    redis_url: str = "redis://localhost:6379/0"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "rag_ddd"
    mongodb_documents_collection: str = "documents"
    mongodb_chunks_collection: str = "chunks"
    gcs_bucket: str = "rag-ddd-documents"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 64
    retrieval_top_k: int = 6
    rerank_top_k: int = 3

    # Opik observability settings
    opik_enabled: bool = True
    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_project_name: str = "rag-ddd"
    opik_use_local: bool = False
