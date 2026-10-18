from vector_db_lab.dataset import generate_documents, stable_query_terms


def test_generate_documents_has_stable_ids():
    docs = generate_documents(count=25, seed=42)
    assert len(docs) == 25
    assert docs[0]["id"] == "doc-0000"
    assert docs[-1]["id"] == "doc-0024"


def test_stable_query_terms_are_present():
    queries = stable_query_terms()
    assert len(queries) == 5
    assert "semantic search" in queries
