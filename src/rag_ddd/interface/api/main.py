from __future__ import annotations

from fastapi import FastAPI

from rag_ddd.config.containers import build_query_use_case, configure_tracing
from rag_ddd.config.settings import Settings
from rag_ddd.interface.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
)

settings = Settings()
configure_tracing(settings)


def build_app() -> FastAPI:
    query_use_case = build_query_use_case(settings)

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
