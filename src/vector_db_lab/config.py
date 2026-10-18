from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: Any | None = None) -> Any:
    return os.getenv(name, default)


def get_int_setting(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def get_bool_setting(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_path_setting(name: str, default: str | None = None) -> Path | None:
    raw = os.getenv(name, default)
    return Path(raw) if raw is not None else None


@dataclass(frozen=True)
class Settings:
    vector_dimension: int = 8
    opensearch_host: str = "http://localhost:9200"
    opensearch_index: str = "vector-db-comparison-lab"
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/vector_lab"
    pgvector_table: str = "documents"
    chroma_persist_directory: str = "./.chroma"
    chroma_collection: str = "vector_db_comparison_lab"
    pinecone_index: str = "demo-index"
    pinecone_namespace: str = "vector-lab"


def get_settings() -> Settings:
    return Settings(
        vector_dimension=get_int_setting("VECTOR_DIMENSION", 8),
        opensearch_host=str(get_setting("OPENSEARCH_HOST", "http://localhost:9200")),
        opensearch_index=str(get_setting("OPENSEARCH_INDEX", "vector-db-comparison-lab")),
        pgvector_dsn=str(get_setting("PGVECTOR_DSN", "postgresql://postgres:postgres@localhost:5432/vector_lab")),
        pgvector_table=str(get_setting("PGVECTOR_TABLE", "documents")),
        chroma_persist_directory=str(get_setting("CHROMA_PERSIST_DIRECTORY", "./.chroma")),
        chroma_collection=str(get_setting("CHROMA_COLLECTION", "vector_db_comparison_lab")),
        pinecone_index=str(get_setting("PINECONE_INDEX", "demo-index")),
        pinecone_namespace=str(get_setting("PINECONE_NAMESPACE", "vector-lab")),
    )


def ensure_directories() -> None:
    Path(get_settings().chroma_persist_directory).mkdir(parents=True, exist_ok=True)
