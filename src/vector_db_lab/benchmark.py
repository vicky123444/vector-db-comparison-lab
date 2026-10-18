from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .dataset import stable_query_terms
from .embeddings import cosine_similarity, deterministic_vector
from .metrics import latency_percentiles, recall_at_k, throughput


def _matching_documents(documents: list[dict[str, Any]], metadata_filter: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metadata_filter:
        return documents
    return [doc for doc in documents if all(doc.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())]


def run_benchmark(
    adapter: Any,
    documents: list[dict[str, Any]],
    queries: list[str] | None = None,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a small reproducible retrieval benchmark against one adapter.

    Vectors are deterministic synthetic vectors, not semantic embeddings. Recall@k
    is computed against exact cosine search over this same vector set; it measures
    index retrieval behavior only, not real-world semantic answer quality.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    queries = queries or stable_query_terms()
    adapter.clear()
    records = [
        {
            "id": document["id"],
            "vector": deterministic_vector(document["text"]),
            "metadata": document.get("metadata", {}),
            "text": document.get("text", ""),
        }
        for document in documents
    ]
    index_started = perf_counter()
    adapter.upsert(records)
    index_seconds = perf_counter() - index_started

    eligible_docs = _matching_documents(documents, metadata_filter)
    vector_by_id = {record["id"]: record["vector"] for record in records}
    timings: list[float] = []
    recall_values: list[float] = []
    for query in queries:
        query_vector = deterministic_vector(query)
        exact_results = sorted(
            eligible_docs,
            key=lambda doc: cosine_similarity(query_vector, vector_by_id[doc["id"]]),
            reverse=True,
        )[:top_k]
        relevant_ids = {document["id"] for document in exact_results}

        started = perf_counter()
        results = adapter.search(query_vector, top_k=top_k, metadata_filter=metadata_filter)
        timings.append(perf_counter() - started)
        recalled_ids = [item["id"] for item in results]
        recall_values.append(recall_at_k(relevant_ids, recalled_ids, k=top_k))

    total_query_seconds = sum(timings)
    return {
        "backend": getattr(adapter, "name", adapter.__class__.__name__),
        "dataset_records": len(documents),
        "filtered_records": len(eligible_docs),
        "queries": len(queries),
        "top_k": top_k,
        "metadata_filter": metadata_filter or {},
        "indexing_time_ms": round(index_seconds * 1000, 3),
        "query_latency_ms": latency_percentiles([timing * 1000 for timing in timings]),
        "throughput_qps": round(throughput(timings, total_query_seconds), 3),
        "recall_at_k_exact_cosine_baseline": round(sum(recall_values) / max(len(recall_values), 1), 4),
        "vector_note": "Deterministic synthetic vectors; not semantic embeddings.",
    }


def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
