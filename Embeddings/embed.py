"""Embedding client: calls the vLLM OpenAI-compatible /v1/embeddings endpoint.

Applies the per-model query/document prefix (critical for retrieval quality —
e.g. Qwen3-Embedding wants an instruction prefix on queries but raw documents).
"""
from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from config import ModelCfg


def _client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def embed_texts(
    texts: Sequence[str],
    is_query: bool,
    cfg: ModelCfg,
    base_url: str,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts. Returns vectors in the same order as `texts`."""
    prefix = cfg.query_prefix if is_query else cfg.doc_prefix
    items = [f"{prefix}{text}" for text in texts]

    client = _client(base_url)
    vectors: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        resp = client.embeddings.create(model=cfg.hf_name, input=chunk)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def probe_dim(cfg: ModelCfg, base_url: str) -> int:
    """Return the model's embedding dimension by embedding a probe string."""
    vector = embed_texts(["dimension probe"], is_query=False, cfg=cfg, base_url=base_url)
    return len(vector[0])
