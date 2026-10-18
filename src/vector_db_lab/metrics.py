from __future__ import annotations

import statistics
from typing import Iterable


def latency_percentiles(samples: Iterable[float], percentiles: tuple[float, ...] = (50.0, 95.0)) -> dict[str, float]:
    values = sorted(samples)
    if not values:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    result: dict[str, float] = {}
    for percentile in percentiles:
        if percentile <= 0:
            result[f"p{int(percentile)}"] = values[0]
        elif percentile >= 100:
            result[f"p{int(percentile)}"] = values[-1]
        else:
            index = (len(values) - 1) * (percentile / 100.0)
            lower = int(index)
            upper = min(lower + 1, len(values) - 1)
            weight = index - lower
            result[f"p{int(percentile)}"] = values[lower] * (1 - weight) + values[upper] * weight
    return result


def throughput(samples: Iterable[float], seconds: float) -> float:
    values = list(samples)
    if not values or seconds <= 0:
        return 0.0
    return len(values) / seconds


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int | None = None) -> float:
    if not relevant_ids:
        return 0.0
    limit = k if k is not None else len(retrieved_ids)
    retrieved_top = retrieved_ids[:limit]
    hit_count = len(relevant_ids.intersection(retrieved_top))
    return hit_count / len(relevant_ids)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return statistics.fmean(items)
