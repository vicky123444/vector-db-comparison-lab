from vector_db_lab.metrics import latency_percentiles, recall_at_k, throughput
from vector_db_lab.benchmark import run_benchmark
from vector_db_lab.backends.memory import MemoryAdapter
from vector_db_lab.dataset import generate_documents


def test_latency_percentiles_smoke():
    values = [0.01, 0.02, 0.05, 0.07, 0.10]
    percentiles = latency_percentiles(values)
    assert set(percentiles) == {"p50", "p95"}
    assert percentiles["p50"] >= 0
    assert percentiles["p95"] >= percentiles["p50"]


def test_recall_at_k_counts_hits():
    relevant = {"a", "b", "c"}
    retrieved = ["b", "x", "a"]
    assert recall_at_k(relevant, retrieved, k=3) == 2 / 3


def test_throughput_handles_empty():
    assert throughput([], 2.0) == 0.0


def test_benchmark_uses_filter_and_reports_measured_operation_fields():
    docs = generate_documents(count=30)
    result = run_benchmark(
        MemoryAdapter(),
        docs,
        queries=["semantic search"],
        top_k=3,
        metadata_filter={"category": "analytics", "published": True},
    )
    assert result["dataset_records"] == 30
    assert result["filtered_records"] == 3
    assert result["indexing_time_ms"] >= 0
    assert result["query_latency_ms"]["p50"] >= 0
    assert 0 <= result["recall_at_k_exact_cosine_baseline"] <= 1
