# CLI Reference

All commands are available via `rag-cli`.

## Query

```bash
rag-cli query "<question>"
```

## Document Pipeline (MongoDB + GCS + Qdrant)

```bash
rag-cli upload <path>
rag-cli ingest-doc <doc_id>
rag-cli reindex-doc <doc_id>
rag-cli rechunk-doc <doc_id>
rag-cli delete-doc <doc_id>
rag-cli list-docs
rag-cli list-docs --status READY
rag-cli doc-status <doc_id>
```

## Notes

- `upload` parses files locally, uploads raw files to GCS, and stores metadata + extracted text in MongoDB.
- `ingest-doc` creates embeddings and indexes chunks into Qdrant.
- `reindex-doc` rebuilds embeddings; it reuses chunks from MongoDB if available.
- `rechunk-doc` regenerates chunks with the current chunk settings, then reindexes.
