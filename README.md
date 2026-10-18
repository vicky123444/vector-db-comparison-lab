# Vector DB Comparison Lab

A Python benchmark lab comparing **OpenSearch, PostgreSQL + pgvector, Chroma, and Pinecone** through a shared adapter contract. It uses the same records, vectors, query set, and metadata conditions to reduce apples-to-oranges comparisons.

> **Important methodology note:** the default vectors are deterministic synthetic vectors, not learned semantic embeddings. This isolates database storage/retrieval mechanics for a reproducible lab; it does **not** measure semantic search quality. The benchmark compares each backend's nearest-neighbor results to an exact cosine-search baseline over those same vectors. Do not publish the output as a general ranking of database speed without documenting your environment, dataset size, index settings, and repeated runs.

## What you can explore

- Upsert records with IDs, text, vectors, and scalar metadata.
- Run nearest-neighbor search using cosine similarity.
- Filter results by metadata such as `category` and `published`.
- Measure indexing duration, p50/p95 query latency, queries per second, and recall@k against an exact baseline.
- Swap storage adapters without changing the benchmark runner.
- Run fast, offline unit tests without live database credentials.

## Backend adapters

| Backend | Adapter | What you need |
|---|---|---|
| OpenSearch | `OpenSearchAdapter` | OpenSearch endpoint and a user allowed to create/query the lab index |
| PostgreSQL + pgvector | `PGVectorAdapter` | PostgreSQL with the `vector` extension available |
| Chroma | `ChromaAdapter` | Local persistent Chroma directory; no separate server required for this setup |
| Pinecone | `PineconeAdapter` | API key and an existing dense-vector index matching `VECTOR_DIMENSION` and cosine metric |

The lab's `MemoryAdapter` is only a test double used by unit tests. It is not one of the four benchmark backends.

## Project structure

```text
vector-db-comparison-lab/
├── src/vector_db_lab/
│   ├── backends/              # OpenSearch, pgvector, Chroma, Pinecone + Protocol
│   ├── benchmark.py           # Timing, exact cosine baseline and result export
│   ├── cli.py                 # Dataset generation and benchmark CLI
│   ├── config.py              # Environment settings
│   ├── dataset.py             # Deterministic synthetic corpus and JSONL I/O
│   ├── embeddings.py           # Deterministic vectors (not semantic embeddings)
│   ├── metrics.py             # p50/p95, throughput and recall@k
│   └── models.py              # Shared record/result models
├── datasets/sample_documents.jsonl
├── configs/benchmark.yaml
├── sql/init.sql
├── docker-compose.yml
├── tests/
├── .env.example
├── requirements.txt
├── pyproject.toml
├── .github/workflows/python-tests.yml
├── LICENSE
└── README.md
```

## Prerequisites

- Python 3.11+
- Docker Desktop for local OpenSearch and PostgreSQL services, if testing those backends
- An API key and a previously created, matching Pinecone index, if testing Pinecone

## 1. Set up Python

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and the package:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2. Configure local services

Copy `.env.example` to `.env` and adjust settings for your environment. Defaults are intended only for local learning, not a publicly exposed or production instance.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

### Start OpenSearch and PostgreSQL + pgvector

```bash
docker compose up -d opensearch postgres
```

The Compose file disables OpenSearch's security plugin for local development and uses sample database credentials. **Do not expose these services to the internet or reuse these credentials in production.** The `VECTOR_DIMENSION` value must match OpenSearch mappings, the pgvector column, Chroma records, and the Pinecone index.

If you change `VECTOR_DIMENSION`, recreate the local PostgreSQL database/table and OpenSearch index before benchmarking. Changing `.env` does not automatically migrate existing vector schemas.

### Chroma

Chroma stores its local collection under `CHROMA_PERSIST_DIRECTORY`. The adapter can run without Docker.

### Pinecone

Create a Pinecone dense-vector index with the configured dimension and **cosine** metric. Set `PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_NAMESPACE` in `.env`. The adapter deliberately does not create a remote index automatically.

## 3. Generate a repeatable dataset

```bash
python -m vector_db_lab.cli generate-dataset --count 200 --output datasets/sample_documents.jsonl
```

The corpus uses synthetic sentences and scalar metadata (`category`, `language`, `source`, `tenant`, `published`). It is intentionally small and reproducible.

## 4. Run a benchmark

First configure and start the backend you want to test, then run one backend at a time:

```bash
python -m vector_db_lab.cli benchmark --backend chroma --top-k 5
python -m vector_db_lab.cli benchmark --backend opensearch --top-k 5
python -m vector_db_lab.cli benchmark --backend pgvector --top-k 5
python -m vector_db_lab.cli benchmark --backend pinecone --top-k 5
```

Add an equality metadata filter:

```bash
python -m vector_db_lab.cli benchmark --backend chroma --top-k 5 --metadata-filter '{"category":"analytics","published":true}'
```

Filter keys and values must be supported by all selected backend adapters; this starter benchmark handles equality conditions only.

The runner clears the selected lab collection/index/table before inserting the benchmark records. **Use a dedicated, disposable index/table/namespace only. Never point this lab at production data.**

Results are written to `results/benchmark.json` (ignored by Git). Output includes dataset size, filter, indexing duration, p50/p95 query latency, observed queries per second, and recall@k against exact cosine ranking over the synthetic vectors.

## Interpret results correctly

- **Indexing time:** wall-clock time used for this run's upsert operation.
- **p50/p95 latency:** percentiles of individual search calls from the recorded run. Use warm-up and many measured queries for stronger conclusions.
- **QPS:** completed test queries divided by total measured query time. This sequential harness is not a concurrent load test.
- **Recall@k:** overlap between a backend's top-k IDs and exact cosine top-k IDs computed over the same synthetic vectors.
- **Scores are not directly comparable:** backend score and distance conventions differ. Compare returned IDs and normalized metrics rather than raw scores.
- **No semantic-embedding conclusion:** deterministic hash vectors are not meaningful language representations. For a second experiment, use one fixed real embedding model and reuse the same vector values in every backend.

## Choosing a backend (high-level guidance)

- **OpenSearch:** consider when vector retrieval needs to coexist with keyword search, filters, or broader search-platform operations.
- **pgvector:** consider when PostgreSQL is central to the application and vectors belong alongside relational data.
- **Chroma:** consider for local RAG experiments, prototyping, and straightforward collection-based retrieval.
- **Pinecone:** consider when you prefer a managed vector service and accept its pricing and provider-specific operating model.

These are architectural starting points, not a universal ranking. Validate workload-specific latency, filtered recall, data volume, operations, and cost.

## Security checklist

- Never commit `.env`, credentials, private data, vector dumps, or benchmark results containing confidential metadata.
- Restrict database endpoints to trusted local/private networks.
- Use a dedicated lab index/table/namespace; the benchmark clears data.
- Use least-privilege credentials and TLS/authentication when services are not local.
- Document and account for any data sent to hosted services.

## References

- [OpenSearch vector search documentation](https://docs.opensearch.org/latest/vector-search/)
- [pgvector project documentation](https://github.com/pgvector/pgvector)
- [Chroma documentation](https://docs.trychroma.com/)
- [Pinecone Python SDK documentation](https://sdk.pinecone.io/python/)

## License

MIT. See [LICENSE](LICENSE).
