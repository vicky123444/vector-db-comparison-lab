from __future__ import annotations

from typing import Any, Protocol


class VectorStore(Protocol):
    def upsert(self, records: list[dict[str, Any]]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def delete(self, ids: list[str]) -> None: ...

    def close(self) -> None: ...

    def clear(self) -> None: ...
