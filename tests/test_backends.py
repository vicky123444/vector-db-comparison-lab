from vector_db_lab.backends import ChromaAdapter, OpenSearchAdapter, PGVectorAdapter, PineconeAdapter
from vector_db_lab.backends.memory import MemoryAdapter


def _assert_filter_contract(adapter):
    adapter.upsert([
        {"id": "a", "vector": [1.0, 0.0, 0.0], "metadata": {"tenant": "a", "published": True}, "text": "alpha"},
        {"id": "b", "vector": [0.0, 1.0, 0.0], "metadata": {"tenant": "b", "published": True}, "text": "bravo"},
    ])
    result = adapter.search([1.0, 0.0, 0.0], top_k=2, metadata_filter={"tenant": "a"})
    assert [item["id"] for item in result] == ["a"]
    adapter.clear()
    adapter.close()


def test_backend_classes_are_available():
    assert [cls.name for cls in [OpenSearchAdapter, PGVectorAdapter, ChromaAdapter, PineconeAdapter]] == [
        "opensearch", "pgvector", "chroma", "pinecone"
    ]


def test_in_memory_test_double_obeys_common_contract():
    _assert_filter_contract(MemoryAdapter())


def test_backend_native_filter_converters():
    assert OpenSearchAdapter.build_filter({"category": "analytics"}) == {
        "bool": {"filter": [{"term": {"metadata.category": "analytics"}}]}
    }
    assert ChromaAdapter.build_filter({"category": "analytics"}) == {"category": {"$eq": "analytics"}}
    assert PineconeAdapter.build_filter({"category": "analytics"}) == {"category": {"$eq": "analytics"}}
