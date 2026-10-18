from vector_db_lab.backends.memory import MemoryAdapter


def test_filter_contract_applies_all_equality_fields():
    adapter = MemoryAdapter()
    adapter.upsert([
        {"id": "a", "vector": [0.1, 0.2, 0.3], "metadata": {"category": "analytics", "published": True}, "text": "analytics document"},
        {"id": "b", "vector": [0.9, 0.8, 0.7], "metadata": {"category": "research", "published": True}, "text": "research document"},
        {"id": "c", "vector": [0.2, 0.1, 0.3], "metadata": {"category": "analytics", "published": False}, "text": "draft analytics document"},
    ])
    result = adapter.search([0.1, 0.2, 0.3], top_k=5, metadata_filter={"category": "analytics", "published": True})
    assert [item["id"] for item in result] == ["a"]
