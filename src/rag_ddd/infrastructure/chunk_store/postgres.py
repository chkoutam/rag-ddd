from __future__ import annotations

import json
from typing import Sequence

import psycopg
from psycopg.rows import dict_row

from rag_ddd.domain.entities import Chunk
from rag_ddd.domain.ports import ChunkStore


class PostgresChunkStore(ChunkStore):
    """ChunkStore backed by PostgreSQL."""

    def __init__(self, dsn: str, table: str = "chunks") -> None:
        self._dsn = dsn
        self._table = table
        self._ensure_table()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT,
                    text TEXT,
                    metadata JSONB
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_doc_id
                ON {self._table} (doc_id)
            """)
            conn.commit()

    def upsert(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        self.delete_by_doc_id(doc_id)
        if not chunks:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        f"""
                        INSERT INTO {self._table} (doc_id, chunk_id, text, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (chunk.doc_id, chunk.chunk_id, chunk.text, json.dumps(chunk.metadata)),
                    )
            conn.commit()

    def list_by_doc_id(self, doc_id: str) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, doc_id, text, metadata FROM {self._table} WHERE doc_id = %s",
                (doc_id,),
            ).fetchall()
        return [
            Chunk(
                chunk_id=str(row.get("chunk_id", "")),
                doc_id=str(row.get("doc_id", "")),
                text=str(row.get("text", "")),
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self._table} WHERE doc_id = %s",
                (doc_id,),
            )
            conn.commit()
