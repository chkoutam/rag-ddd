from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from rag_ddd.config.containers import (
    build_document_store,
    build_document_use_cases,
    build_query_use_case,
    configure_tracing,
)
from rag_ddd.config.settings import Settings

app = typer.Typer(add_completion=False)
settings = Settings()
configure_tracing(settings)


@app.command()
def query(text: str) -> None:
    query_use_case = build_query_use_case(settings)
    result = query_use_case.execute(text)
    print(result.answer.text)
    if result.answer.sources:
        print("\nSources:")
        for item in result.answer.sources:
            print(f"- {item.chunk.metadata.get('source', '')} (score: {item.score:.3f})")


@app.command()
def upload(path: str) -> None:
    upload_use_case, _, _, _, _, loader = build_document_use_cases(settings)
    documents = loader.load(path)
    if not documents:
        print("No documents found.")
        return

    document_store = build_document_store(settings)
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
        document_store.update(result.doc_id, {"text": document.text, "status": "READY"})
        print(f"Uploaded {source_path.name} -> {result.doc_id}")


@app.command("ingest-doc")
def ingest_document(doc_id: str) -> None:
    _, ingest_use_case, _, _, _, _ = build_document_use_cases(settings)
    result = ingest_use_case.execute(doc_id)
    print(f"Ingested {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("reindex-doc")
def reindex_document(doc_id: str) -> None:
    _, _, reindex_use_case, _, _, _ = build_document_use_cases(settings)
    result = reindex_use_case.execute(doc_id)
    print(f"Reindexed {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("delete-doc")
def delete_document(doc_id: str) -> None:
    _, _, _, delete_use_case, _, _ = build_document_use_cases(settings)
    result = delete_use_case.execute(doc_id)
    print(f"Delete {result.doc_id} (status={result.status})")


@app.command("rechunk-doc")
def rechunk_document(doc_id: str) -> None:
    _, _, _, _, rechunk_use_case, _ = build_document_use_cases(settings)
    result = rechunk_use_case.execute(doc_id)
    print(f"Rechunked {result.doc_id} with {result.chunks} chunks (status={result.status})")


@app.command("list-docs")
def list_documents(status: str | None = None) -> None:
    document_store = build_document_store(settings)
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
    document_store = build_document_store(settings)
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


if __name__ == "__main__":
    app()
