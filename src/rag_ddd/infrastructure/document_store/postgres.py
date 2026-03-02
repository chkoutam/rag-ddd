from __future__ import annotations

import json
from typing import Any, Mapping

import psycopg
from psycopg.rows import dict_row

from rag_ddd.domain.ports import DocumentStore


class PostgresDocumentStore(DocumentStore):
    """DocumentStore backed by PostgreSQL JSONB."""

    def __init__(self, dsn: str, table: str = "documents") -> None:
        self._dsn = dsn
        self._table = table
        self._ensure_table()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    data JSONB NOT NULL
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_data_gin
                ON {self._table} USING GIN (data)
            """)
            conn.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_{self._table}_checksum
                ON {self._table} ((data->>'checksum'))
                WHERE data->>'checksum' IS NOT NULL
            """)
            conn.commit()

    def create(self, document: Mapping[str, Any]) -> str:
        with self._connect() as conn:
            row = conn.execute(
                f"INSERT INTO {self._table} (data) VALUES (%s::jsonb) RETURNING id",
                (json.dumps(dict(document)),),
            ).fetchone()
            conn.commit()
            return str(row["id"])

    def update(self, doc_id: str, patch: Mapping[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {self._table} SET data = data || %s::jsonb WHERE id = %s::uuid",
                (json.dumps(dict(patch)), doc_id),
            )
            conn.commit()

    def get(self, doc_id: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT id, data FROM {self._table} WHERE id = %s::uuid",
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        return self._serialize(row)

    def list(self, filters: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
        with self._connect() as conn:
            if filters:
                rows = conn.execute(
                    f"SELECT id, data FROM {self._table} WHERE data @> %s::jsonb",
                    (json.dumps(dict(filters)),),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT id, data FROM {self._table}",
                ).fetchall()
        return [self._serialize(row) for row in rows]

    def delete(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self._table} WHERE id = %s::uuid",
                (doc_id,),
            )
            conn.commit()

    @staticmethod
    def _serialize(row: dict) -> Mapping[str, Any]:
        data = dict(row["data"])
        data["_id"] = str(row["id"])
        return data
