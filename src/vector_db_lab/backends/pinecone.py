from __future__ import annotations

import os
from typing import Any

from ..config import get_settings


class PineconeAdapter:
    """Pinecone vector-index adapter. Requires a pre-created compatible index."""

    name = "pinecone"

    def __init__(self, client: Any | None = None, index: Any | None = None) -> None:
        settings = get_settings()
        self.index_name = settings.pinecone_index
        self.namespace = settings.pinecone_namespace
        self.dimension = settings.vector_dimension
        if index is not None:
            self.client = client
            self.index = index
            return
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key or api_key.lower().startswith(("replace", "your-", "example")):
            raise ValueError("Set PINECONE_API_KEY in .env to use the Pinecone adapter")
        from pinecone import Pinecone

        self.client = client or Pinecone(api_key=api_key)
        try:
            names = set(self.client.list_indexes().names())
        except AttributeError:
            names = {item["name"] for item in self.client.list_indexes()}
        if self.index_name not in names:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' was not found. Create a cosine index "
                f"with dimension {self.dimension}, then set PINECONE_INDEX in .env."
            )
        index_factory = getattr(self.client, "Index", None) or getattr(self.client, "index")
        self.index = index_factory(self.index_name)

    @staticmethod
    def build_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata_filter:
            return None
        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        vectors = []
        for record in records:
            metadata = dict(record.get("metadata", {}))
            metadata["text"] = record.get("text", "")
            vectors.append({"id": record["id"], "values": record["vector"], "metadata": metadata})
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            filter=self.build_filter(metadata_filter),
            include_metadata=True,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        results = []
        for match in matches:
            if isinstance(match, dict):
                identifier, score, metadata = match.get("id", ""), match.get("score", 0.0), match.get("metadata") or {}
            else:
                identifier, score, metadata = match.id, match.score, match.metadata or {}
            metadata = dict(metadata)
            text = metadata.pop("text", "")
            results.append({"id": identifier, "score": float(score or 0.0), "text": text, "metadata": metadata})
        return results

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.index.delete(ids=ids, namespace=self.namespace)

    def clear(self) -> None:
        self.index.delete(delete_all=True, namespace=self.namespace)

    def close(self) -> None:
        close = getattr(self.index, "close", None)
        if callable(close):
            close()
