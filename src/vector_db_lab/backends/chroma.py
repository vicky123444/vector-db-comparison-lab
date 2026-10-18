from __future__ import annotations

from typing import Any

from ..config import get_settings


class ChromaAdapter:
    """Persistent Chroma adapter using cosine distance and metadata filters."""

    name = "chroma"

    def __init__(self, client: Any | None = None, collection: Any | None = None) -> None:
        settings = get_settings()
        self.collection_name = settings.chroma_collection
        if collection is not None:
            self.client = client
            self.collection = collection
            return
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def build_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata_filter:
            return None
        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        ids = [record["id"] for record in records]
        embeddings = [record["vector"] for record in records]
        documents = [record.get("text", "") for record in records]
        metadatas = [record.get("metadata", {}) or {"_empty_metadata": True} for record in records]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        kwargs: dict[str, Any] = {"query_embeddings": [query_vector], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
        native_filter = self.build_filter(metadata_filter)
        if native_filter:
            kwargs["where"] = native_filter
        response = self.collection.query(**kwargs)
        ids = (response.get("ids") or [[]])[0]
        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results = []
        for index, identifier in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) and distances[index] is not None else 0.0
            metadata = metas[index] if index < len(metas) and metas[index] else {}
            metadata = {key: value for key, value in metadata.items() if key != "_empty_metadata"}
            results.append({
                "id": identifier,
                "score": 1.0 - distance,
                "text": docs[index] if index < len(docs) else "",
                "metadata": metadata,
            })
        return results

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def clear(self) -> None:
        response = self.collection.get(include=[])
        ids = response.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def close(self) -> None:
        # Chroma's PersistentClient has no mandatory close operation.
        return None
