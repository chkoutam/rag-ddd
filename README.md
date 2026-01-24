# rag-ddd

Production-ready RAG (Retrieval-Augmented Generation) in Python with DDD architecture.

## Quick start

1) Copy env file

```
cp .env.example .env
```

2) Start dependencies
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

```
docker compose up -d
```

3) Run API

```
uvicorn rag_ddd.interface.api.main:app --reload
```

4) Ingest and query

```
rag-cli ingest /path/to/pdfs
rag-cli query "Your question"
```

## Architecture

See `docs/architecture.md` for diagrams.
