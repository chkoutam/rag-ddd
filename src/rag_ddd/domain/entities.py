from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NLPEntity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class NLPResult:
    entities: List[NLPEntity] = field(default_factory=list)
    category: str = "unknown"
    category_confidence: float = 0.0
    title: str | None = None
    author: str | None = None
    key_concepts: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk]
