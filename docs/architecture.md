# Architecture (DDD RAG)

## Dependency Flow (Clean Architecture)

```mermaid
flowchart LR
  Interface[interface] --> Application[application]
  Interface --> Infrastructure[infrastructure]
  Interface --> Config[config]
  Application --> Domain[domain]
  Infrastructure --> Domain
```

## Layer Overview

- Domain: business objects (entities/value objects) + ports (interfaces).
- Application: use cases + DTOs that package results.
- Infrastructure: adapters implementing ports (IO, models, stores).
- Interface: API/CLI composition root; maps IO to use cases.
- Config: settings for adapters and use case defaults.

## Domain Model (Core)

```mermaid
classDiagram
  class Document {
    +doc_id: str
    +text: str
    +metadata: dict
  }
  class Chunk {
    +chunk_id: str
    +doc_id: str
    +text: str
    +metadata: dict
  }
  class Query {
    +text: str
  }
  class RetrievedChunk {
    +chunk: Chunk
    +score: float
  }
  class Answer {
    +text: str
    +sources: List~RetrievedChunk~
  }

  RetrievedChunk --> Chunk
  Answer --> RetrievedChunk
```

Notes:
- `Document` and `Chunk` represent persisted knowledge.
- `Query` and `Answer` are value objects for the interaction.
- `RetrievedChunk` is a projection of retrieval results (chunk + score).

## Ports and Adapters

```mermaid
classDiagram
  class DocumentLoader {
    +load(path)
  }
  class Chunker {
    +chunk(documents)
  }
  class VectorStore {
    +ensure_collection()
    +upsert(chunks, embeddings)
    +query(embedding, top_k)
  }
  class EmbeddingModel {
    +embed(texts)
  }
  class LLM {
    +generate(prompt)
  }
  class Cache {
    +get(key)
    +set(key, value, ttl)
  }
  class Reranker {
    +rerank(query, chunks, top_k)
  }
  class DocumentStore {
    +create(doc)
    +update(doc_id, patch)
    +get(doc_id)
    +list(filters)
  }
  class BlobStore {
    +put(path, bytes)
    +get(path)
    +delete(path)
  }

  class FileLoader
  class RecursiveChunker
  class OpenAIEmbeddingModel
  class OpenAILLM
  class QdrantVectorStore
  class RedisCache
  class NoOpReranker
  class MongoDocumentStore
  class GCSBlobStore

  DocumentLoader <|.. FileLoader
  Chunker <|.. RecursiveChunker
  EmbeddingModel <|.. OpenAIEmbeddingModel
  LLM <|.. OpenAILLM
  VectorStore <|.. QdrantVectorStore
  Cache <|.. RedisCache
  Reranker <|.. NoOpReranker
  DocumentStore <|.. MongoDocumentStore
  BlobStore <|.. GCSBlobStore
```

## Application Layer

```mermaid
classDiagram
  class UploadDocumentUseCase {
    +execute(filename, content)
  }
  class IngestDocumentUseCase {
    +execute(doc_id)
  }
  class ReindexDocumentUseCase {
    +execute(doc_id)
  }
  class RechunkDocumentUseCase {
    +execute(doc_id)
  }
  class DeleteDocumentUseCase {
    +execute(doc_id)
  }
  class RAGQueryUseCase {
    +execute(query_text)
  }
  class UploadResult {
    +doc_id: str
  }
  class IngestDocumentResult {
    +doc_id: str
    +chunks: int
  }
  class DeleteDocumentResult {
    +doc_id: str
    +status: str
  }
  class QueryResult {
    +answer: Answer
  }

  UploadDocumentUseCase --> UploadResult
  IngestDocumentUseCase --> IngestDocumentResult
  ReindexDocumentUseCase --> IngestDocumentResult
  RechunkDocumentUseCase --> IngestDocumentResult
  DeleteDocumentUseCase --> DeleteDocumentResult
  RAGQueryUseCase --> QueryResult
```

## Interfaces

- API (FastAPI): `QueryRequest`, `QueryResponse`.
- CLI (Typer): `upload`, `ingest-doc`, `query`, `reindex-doc`, `rechunk-doc`, `delete-doc`.
- Both are composition roots wiring adapters + use cases.

## Storage Model (Production)

- MongoDB is the source of truth for document metadata and extracted text.
- Google Cloud Storage (GCS) stores raw files (PDFs, etc.).
- Qdrant stores embeddings for retrieval.

Example MongoDB collections:
- `documents`: one doc per source file, with status and metadata.
- `chunks`: persisted chunks for fast reindexing.
- `ingest_jobs`: ingestion status and errors.

Example document fields (`documents`):
- `_id`, `title`, `source_uri`, `gcs_path`, `status`
- `text`, `metadata`, `created_at`, `updated_at`
- `version`, `checksum`

## Production Ingest Flow (MongoDB + GCS)

```mermaid
sequenceDiagram
  participant API as Interface
  participant Blob as BlobStore (GCS)
  participant Store as DocumentStore (MongoDB)
  participant Chunker as Chunker
  participant Embed as EmbeddingModel
  participant VS as VectorStore (Qdrant)

  API->>Blob: put(file)
  API->>Store: create(document metadata + gcs_path + status=UPLOADED)
  API->>Chunker: chunk(extracted text)
  API->>Embed: embed(chunks)
  API->>VS: upsert(chunks, embeddings)
  API->>Store: update(status=INDEXED, chunk_count, checksum)
```

## Production Query Flow (MongoDB + GCS)

```mermaid
sequenceDiagram
  participant API as Interface
  participant Cache as Cache
  participant Embed as EmbeddingModel
  participant VS as VectorStore (Qdrant)
  participant Store as DocumentStore (MongoDB)
  participant LLM as LLM

  API->>Cache: get(cache_key)
  API->>Embed: embed([query_text])
  API->>VS: query(embedding, top_k)
  API->>Store: get(doc metadata for sources)
  API->>LLM: generate(prompt)
  API->>Cache: set(cache_key, answer, ttl)
```

## Key Flows

### Ingest Flow

```mermaid
sequenceDiagram
  participant CLI/API as Interface
  participant Ingest as RAGIngestUseCase
  participant Loader as DocumentLoader
  participant Chunker as Chunker
  participant Embed as EmbeddingModel
  participant Store as VectorStore

  CLI/API->>Ingest: execute(path)
  Ingest->>Loader: load(path)
  Ingest->>Chunker: chunk(documents)
  Ingest->>Embed: embed(texts)
  Ingest->>Store: upsert(chunks, embeddings)
```

### Query Flow

```mermaid
sequenceDiagram
  participant CLI/API as Interface
  participant QueryUC as RAGQueryUseCase
  participant Cache as Cache
  participant Embed as EmbeddingModel
  participant Store as VectorStore
  participant Rerank as Reranker
  participant LLM as LLM

  CLI/API->>QueryUC: execute(query_text)
  QueryUC->>Cache: get(cache_key)
  QueryUC->>Embed: embed([query_text])
  QueryUC->>Store: query(embedding, top_k)
  QueryUC->>Rerank: rerank(query, chunks, top_k)
  QueryUC->>LLM: generate(prompt)
  QueryUC->>Cache: set(cache_key, answer, ttl)
```
