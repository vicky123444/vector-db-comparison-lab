# Backend implementation and benchmark status

This repository contains client-backed adapters for OpenSearch, PostgreSQL + pgvector, Chroma, and Pinecone. The in-memory `MemoryAdapter` exists only for fast offline unit tests.

The benchmark creates deterministic synthetic vectors. These are intentionally not semantic embeddings. Recall is computed against an exact cosine-ranking baseline over the exact same synthetic vectors and is only a metric for retrieval behavior under this lab's conditions.

A passing GitHub Actions run verifies the offline unit tests and imports; it does not prove a live remote service was reachable. To make a genuine comparison, configure each backend separately, use disposable resources, run repeated warm-up and measured tests, and save the exact configuration with results. Do not compare latency output from unlike hardware/service plans as a universal ranking.

The benchmark clears the configured lab collection/index/table before upserting records. Never point it at production data. The included local Docker Compose configuration disables OpenSearch security and uses demonstration credentials; never expose it to public networks.

The test suite is not a security audit or a production-readiness certification.
