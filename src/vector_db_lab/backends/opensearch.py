from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from ..config import get_settings, get_bool_setting


class OpenSearchAdapter:
    """OpenSearch k-NN adapter using cosine similarity and equality metadata filters."""

    name = "opensearch"

    def __init__(self, client: Any | None = None, ensure_index: bool = True) -> None:
        settings = get_settings()
        self.index_name = settings.opensearch_index
        self.dimension = settings.vector_dimension
        if client is None:
            from opensearchpy import OpenSearch

            host = settings.opensearch_host
            parsed = urlparse(host)
            username = os.getenv("OPENSEARCH_USERNAME")
            password = os.getenv("OPENSEARCH_PASSWORD")
            auth = (username, password) if username and password else None
            verify_certs = get_bool_setting("OPENSEARCH_VERIFY_CERTS", default=False)
            is_https = parsed.scheme == "https"
            self.client = OpenSearch(
                hosts=[{"host": parsed.hostname or "localhost", "port": parsed.port or (443 if is_https else 9200)}],
                http_auth=auth,
                use_ssl=is_https,
                verify_certs=verify_certs,
                ssl_show_warn=verify_certs,
                timeout=30,
            )
        else:
            self.client = client
        if ensure_index:
            self._ensure_index()

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "dynamic_templates": [
                    {"metadata_strings": {"path_match": "metadata.*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                ],
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "space_type": "cosinesimil",
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "dynamic": True},
                },
            },
        }
        self.client.indices.create(index=self.index_name, body=body)

    @staticmethod
    def build_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata_filter:
            return None
        return {"bool": {"filter": [{"term": {f"metadata.{key}": value}} for key, value in metadata_filter.items()]}}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            body = {
                "vector": record["vector"],
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            }
            self.client.index(index=self.index_name, id=record["id"], body=body, refresh=False)
        if records:
            self.client.indices.refresh(index=self.index_name)

    def search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        knn: dict[str, Any] = {"vector": query_vector, "k": top_k}
        native_filter = self.build_filter(metadata_filter)
        if native_filter:
            knn["filter"] = native_filter
        response = self.client.search(
            index=self.index_name,
            body={"size": top_k, "query": {"knn": {"vector": knn}}},
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            {
                "id": hit.get("_id", ""),
                "score": float(hit.get("_score", 0.0)),
                "text": hit.get("_source", {}).get("text", ""),
                "metadata": hit.get("_source", {}).get("metadata", {}),
            }
            for hit in hits
        ]

    def delete(self, ids: list[str]) -> None:
        for identifier in ids:
            self.client.delete(index=self.index_name, id=identifier, ignore=[404], refresh=False)
        if ids:
            self.client.indices.refresh(index=self.index_name)

    def clear(self) -> None:
        self.client.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}}, refresh=True, conflicts="proceed")

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
