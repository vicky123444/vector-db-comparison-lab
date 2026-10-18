from __future__ import annotations

from typing import Any

from ..embeddings import cosine_similarity


class MemoryAdapter:
    """In-memory adapter for offline contract tests; not a benchmark backend."""

    name = "memory-test-double"

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.records[record["id"]] = record

    def search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        scored = []
        for record in self.records.values():
            metadata = record.get("metadata", {})
            if metadata_filter and not all(metadata.get(key) == value for key, value in metadata_filter.items()):
                continue
            scored.append({
                "id": record["id"],
                "score": cosine_similarity(query_vector, record["vector"]),
                "text": record.get("text", ""),
                "metadata": metadata,
            })
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def delete(self, ids: list[str]) -> None:
        for identifier in ids:
            self.records.pop(identifier, None)

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        self.clear()
