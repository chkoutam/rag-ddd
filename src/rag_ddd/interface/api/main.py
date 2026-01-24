from __future__ import annotations

from fastapi import FastAPI

from rag_ddd.application.use_cases import RAGQueryUseCase
from rag_ddd.config.settings import Settings
from rag_ddd.infrastructure.cache.redis_cache import RedisCache
from rag_ddd.infrastructure.embeddings.openai import OpenAIEmbeddingModel
from rag_ddd.infrastructure.llm.openai import OpenAILLM
from rag_ddd.infrastructure.observability import configure_opik
from rag_ddd.infrastructure.reranker.noop import NoOpReranker
from rag_ddd.infrastructure.vector_store.qdrant import QdrantVectorStore
from rag_ddd.interface.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

settings = Settings()

# Configure Opik observability
if settings.opik_enabled:
    configure_opik(
        api_key=settings.opik_api_key,
        workspace=settings.opik_workspace,
        project_name=settings.opik_project_name,
        use_local=settings.opik_use_local,
    )


def build_app() -> FastAPI:
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

    app = FastAPI(title="RAG DDD", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = query_use_case.execute(request.text)
        sources = [
            SourceChunk(
                doc_id=item.chunk.doc_id,
                text=item.chunk.text,
                score=item.score,
                metadata=item.chunk.metadata,
            )
            for item in result.answer.sources
        ]
        return QueryResponse(answer=result.answer.text, sources=sources)

    return app


app = build_app()
