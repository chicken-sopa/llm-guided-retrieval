"""In-memory vector index over the cached chunk embeddings.

The corpus is ~116 articles, so there's no need for a database: build_index.py
caches each article's embedding into convention_chunks.json, and this module
loads those vectors into a NumPy matrix and does exact cosine top-k in memory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def chunk_article_id(chunk: dict, normalize_article_label) -> str | None:
    """Map a chunk's (protocol, article_number) to a normalized gold-aligned id."""
    protocol = str(chunk.get("protocol", "")).strip()
    number = str(chunk.get("article_number", "")).strip()
    if not number:
        return None
    if protocol.lower() in ("", "convention"):
        label = f"Article {number}"
    else:
        match = re.search(r"(\d+)", protocol)
        label = f"Protocol {match.group(1)} Article {number}" if match else f"Article {number}"
    return normalize_article_label(label)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class VectorIndex:
    """Loads embedded chunks and answers exact cosine top-k queries in memory."""

    def __init__(self, records: list[dict], matrix: np.ndarray):
        self.records = records
        self.matrix = _normalize_rows(matrix.astype(np.float32))

    @property
    def size(self) -> int:
        return len(self.records)

    @classmethod
    def from_chunks_json(cls, chunks_path: str | Path, model_name: str, normalize_article_label) -> "VectorIndex":
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        records: list[dict] = []
        vectors: list[list[float]] = []
        for chunk in chunks:
            embedding = chunk.get("embeddings")
            if not embedding or chunk.get("embedding_model") != model_name:
                continue
            article_id = chunk_article_id(chunk, normalize_article_label)
            if article_id is None:
                continue
            records.append(
                {
                    "article_id": article_id,
                    "part": chunk.get("protocol", ""),
                    "title": chunk.get("title", ""),
                    "content": chunk.get("text", ""),
                }
            )
            vectors.append(embedding)

        if not records:
            raise SystemExit(
                f"No chunks in {chunks_path} have embeddings for model '{model_name}'. "
                "Run build_index.py first."
            )
        return cls(records, np.asarray(vectors, dtype=np.float32))

    def search(self, query_vec: list[float], k: int) -> list[dict[str, Any]]:
        query = np.asarray(query_vec, dtype=np.float32)
        query = query / np.clip(np.linalg.norm(query), 1e-12, None)
        sims = self.matrix @ query
        top = np.argsort(-sims)[:k]
        return [{**self.records[i], "similarity": float(sims[i])} for i in top]
