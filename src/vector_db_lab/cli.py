from __future__ import annotations

import argparse
import importlib
import json

from .benchmark import run_benchmark, write_results
from .dataset import generate_documents, load_dataset, write_dataset


BACKEND_MAP = {
    "opensearch": ("vector_db_lab.backends.opensearch", "OpenSearchAdapter"),
    "pgvector": ("vector_db_lab.backends.pgvector", "PGVectorAdapter"),
    "chroma": ("vector_db_lab.backends.chroma", "ChromaAdapter"),
    "pinecone": ("vector_db_lab.backends.pinecone", "PineconeAdapter"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vector database comparison lab")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate-dataset", help="Generate a synthetic dataset")
    generate_parser.add_argument("--count", type=int, default=200)
    generate_parser.add_argument("--output", type=str, default="datasets/sample_documents.jsonl")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark an actual configured backend")
    benchmark_parser.add_argument("--backend", choices=sorted(BACKEND_MAP), required=True)
    benchmark_parser.add_argument("--dataset", type=str, default="datasets/sample_documents.jsonl")
    benchmark_parser.add_argument("--top-k", type=int, default=5)
    benchmark_parser.add_argument("--metadata-filter", type=str, default=None, help="JSON equality filter for metadata fields")
    benchmark_parser.add_argument("--output", type=str, default="results/benchmark.json")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-dataset":
        documents = generate_documents(count=args.count)
        write_dataset(documents, args.output)
        print(f"Wrote {len(documents)} documents to {args.output}")
        return

    if args.command == "benchmark":
        module_name, class_name = BACKEND_MAP[args.backend]
        module = importlib.import_module(module_name)
        backend_cls = getattr(module, class_name)
        documents = load_dataset(args.dataset)
        metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            parser.error("--metadata-filter must be a JSON object")
        backend = backend_cls()
        try:
            result = run_benchmark(backend, documents, top_k=args.top_k, metadata_filter=metadata_filter)
            result["backend"] = args.backend
            write_results(result, args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            backend.close()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
