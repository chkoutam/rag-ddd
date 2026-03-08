from __future__ import annotations

import json
from typing import Iterable, List, Sequence

import psycopg
from psycopg.rows import dict_row

from rag_ddd.domain.entities import Chunk, RetrievedChunk
from rag_ddd.domain.ports import VectorStore


class PgVectorStore(VectorStore):
    """VectorStore backed by PostgreSQL + pgvector extension.

    Uses cosine distance (<=>).  The table is created automatically on first
    use; the vector dimension is inferred from the first batch of embeddings.
    """

    def __init__(self, dsn: str, table: str = "embeddings") -> None:
        self._dsn = dsn
        self._table = table
        self._dim: int | None = None

    # ── internal helpers ──────────────────────────────────────────────────

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        # Register the pgvector type so psycopg can handle the vector column.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        try:
            from pgvector.psycopg import register_vector
            register_vector(conn)
        except ImportError:
            pass  # fall back to plain list serialisation
        return conn

    def _table_exists(self, conn: psycopg.Connection) -> bool:
        row = conn.execute(
            "SELECT to_regclass(%s) AS r", (self._table,)
        ).fetchone()
        return row is not None and row.get("r") is not None  # type: ignore[union-attr]

    # ── VectorStore port ──────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        if self._dim is None:
            raise ValueError("Vector dimension unknown; call upsert first.")
        with self._connect() as conn:
            self._create_table_if_needed(conn, self._dim)

    def _create_table_if_needed(self, conn: psycopg.Connection, dim: int) -> None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                chunk_id  TEXT PRIMARY KEY,
                doc_id    TEXT NOT NULL,
                text      TEXT NOT NULL,
                metadata  JSONB,
                embedding vector({dim}) NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_doc_id
            ON {self._table} (doc_id)
        """)
        # IVFFlat index for ANN search — only useful once the table has rows,
        # but CREATE INDEX IF NOT EXISTS is idempotent.
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_embedding
            ON {self._table} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        conn.commit()

    def upsert(self, chunks: Iterable[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        chunks_list = list(chunks)
        embeddings_list = list(embeddings)
        if not chunks_list:
            return

        if self._dim is None:
            self._dim = len(embeddings_list[0])

        with self._connect() as conn:
            if not self._table_exists(conn):
                self._create_table_if_needed(conn, self._dim)

            with conn.cursor() as cur:
                for chunk, vector in zip(chunks_list, embeddings_list):
                    vec_str = "[" + ",".join(str(v) for v in vector) + "]"
                    cur.execute(
                        f"""
                        INSERT INTO {self._table}
                            (chunk_id, doc_id, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET doc_id    = EXCLUDED.doc_id,
                                text      = EXCLUDED.text,
                                metadata  = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.text,
                            json.dumps(chunk.metadata),
                            vec_str,
                        ),
                    )
            conn.commit()

    def query(self, embedding: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        with self._connect() as conn:
            if not self._table_exists(conn):
                return []
            rows = conn.execute(
                f"""
                SELECT chunk_id, doc_id, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self._table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k),
            ).fetchall()

        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    text=str(row["text"]),
                    metadata=row.get("metadata") or {},
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        with self._connect() as conn:
            if not self._table_exists(conn):
                return
            conn.execute(
                f"DELETE FROM {self._table} WHERE doc_id = %s",
                (doc_id,),
            )
            conn.commit()
