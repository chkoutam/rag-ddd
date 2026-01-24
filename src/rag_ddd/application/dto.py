from __future__ import annotations

from dataclasses import dataclass

from rag_ddd.domain.entities import Answer


@dataclass(frozen=True)
class QueryResult:
    answer: Answer


@dataclass(frozen=True)
class UploadResult:
    doc_id: str
    gcs_path: str
    status: str


@dataclass(frozen=True)
class IngestDocumentResult:
    doc_id: str
    chunks: int
    status: str


@dataclass(frozen=True)
class DeleteDocumentResult:
    doc_id: str
    status: str
