from __future__ import annotations

import re
from typing import Any

from ..config import get_settings


class PGVectorAdapter:
    """PostgreSQL + pgvector adapter using cosine distance and JSONB equality filters."""

    name = "pgvector"

    def __init__(self, connection: Any | None = None, initialize_schema: bool = True, vector_factory: Any | None = None, jsonb_factory: Any | None = None) -> None:
        settings = get_settings()
        self.table_name = settings.pgvector_table
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.table_name):
            raise ValueError("PGVECTOR_TABLE must be a simple SQL identifier")
        self.dimension = settings.vector_dimension
        if connection is None:
            import psycopg
            from psycopg.rows import dict_row
            from pgvector.psycopg import register_vector

            self.connection = psycopg.connect(settings.pgvector_dsn, row_factory=dict_row)
            self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.connection.commit()
            register_vector(self.connection)
        else:
            self.connection = connection
        self._vector_factory = vector_factory
        self._jsonb_factory = jsonb_factory
        if initialize_schema:
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}', "
            f"embedding VECTOR({self.dimension}) NOT NULL)"
        )
        self.connection.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw "
            f"ON {self.table_name} USING hnsw (embedding vector_cosine_ops)"
        )
        self.connection.commit()

    def _vector(self, values: list[float]) -> Any:
        if self._vector_factory is not None:
            return self._vector_factory(values)
        from pgvector import Vector
        return Vector(values)

    def _jsonb(self, value: dict[str, Any]) -> Any:
        if self._jsonb_factory is not None:
            return self._jsonb_factory(value)
        from psycopg.types.json import Jsonb
        return Jsonb(value)

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        sql = (
            f"INSERT INTO {self.table_name} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding"
        )
        params = [
            (record["id"], record.get("text", ""), self._jsonb(record.get("metadata", {})), self._vector(record["vector"]))
            for record in records
        ]
        self.connection.cursor().executemany(sql, params)
        self.connection.commit()

    def search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        where = ""
        params: list[Any] = [self._vector(query_vector)]
        if metadata_filter:
            where = " WHERE metadata @> %s"
            params.append(self._jsonb(metadata_filter))
        params.extend([self._vector(query_vector), top_k])
        sql = (
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score FROM {self.table_name}"
            f"{where} ORDER BY embedding <=> %s LIMIT %s"
        )
        rows = self.connection.execute(sql, params).fetchall()
        return [
            {"id": row["id"], "text": row["text"], "metadata": row["metadata"], "score": float(row["score"])}
            for row in rows
        ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.connection.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (ids,))
        self.connection.commit()

    def clear(self) -> None:
        self.connection.execute(f"TRUNCATE TABLE {self.table_name}")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
