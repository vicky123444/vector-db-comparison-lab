from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def generate_documents(count: int = 200, seed: int = 42) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be zero or greater")
    random.seed(seed)
    docs: list[dict[str, Any]] = []
    topics = [
        "vector search",
        "rag pipeline",
        "postgresql",
        "opensearch",
        "pinecone",
        "chroma",
        "metadata filtering",
        "semantic retrieval",
        "benchmarking",
        "embedding quality",
    ]
    categories = ["analytics", "engineering", "research", "products", "ops"]
    for index in range(count):
        topic = topics[index % len(topics)]
        category = categories[index % len(categories)]
        doc = {
            "id": f"doc-{index:04d}",
            "text": f"{topic} and {category} discussion for dataset entry {index}",
            "metadata": {
                "category": category,
                "language": "en",
                "source": f"synthetic-{index % 5}",
                "tenant": f"tenant-{index % 3}",
                "published": index % 2 == 0,
            },
        }
        docs.append(doc)
    return docs


def write_dataset(documents: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, sort_keys=True) + "\n")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def stable_query_terms() -> list[str]:
    return [
        "semantic search",
        "vector retrieval",
        "metadata filter",
        "embedding pipeline",
        "database indexing",
    ]
