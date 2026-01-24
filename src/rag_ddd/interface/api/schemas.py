from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class QueryRequest(BaseModel):
    text: str


class SourceChunk(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
