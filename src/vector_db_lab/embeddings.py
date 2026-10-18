from __future__ import annotations

import hashlib
import math
from typing import Iterable

from .config import get_settings


def deterministic_vector(text: str, dimension: int | None = None, salt: str = "vector-db-comparison") -> list[float]:
    """Generate a repeatable synthetic vector for testing database mechanics.

    This is deliberately not an embedding model and must not be used to judge
    semantic quality. Use the same stored vectors and exact-search baseline to
    compare database behavior independently of an embedding provider.
    """
    size = dimension or get_settings().vector_dimension
    values: list[float] = []
    for index in range(size):
        token = f"{salt}:{text}:{index}".encode("utf-8")
        digest = hashlib.sha256(token).hexdigest()
        numeric = int(digest[:8], 16) / float(0xFFFFFFFF)
        values.append((numeric * 2.0) - 1.0)
    return values


def build_vectors(documents: Iterable[dict], dimension: int | None = None) -> list[list[float]]:
    return [deterministic_vector(item.get("text", ""), dimension=dimension) for item in documents]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimensions do not match")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
