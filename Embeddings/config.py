"""Central configuration for the embedding-search baseline.

HOW TO USE: imported by build_index.py and eval_embedding_search.py — not run
directly. Edit MODELS / ACTIVE_MODEL here (or set the EMBED_MODEL / EMBED_BASE_URL
/ TOP_K env vars) to change which embedding model and endpoint are used.

Single model now, registry-ready: `MODELS` is keyed by slug so adding more
models later (multi-model sweep) is just adding entries — no other code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency).

    Reads Embeddings/.env if present and sets vars that are not already in the
    environment. Lets host-run scripts pick up DB creds / model settings.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


@dataclass
class ModelCfg:
    slug: str
    hf_name: str
    query_prefix: str = ""
    doc_prefix: str = ""


# ---- Model registry (one entry now; add more for a multi-model sweep) ----
MODELS: dict[str, ModelCfg] = {
    "qwen3-emb-8b": ModelCfg(
        slug="qwen3-emb-8b",
        hf_name=_env("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B"),
        # Qwen3-Embedding convention: instruction prefix on queries, raw docs.
        query_prefix=(
            "Instruct: Given the facts of a legal case, retrieve the European "
            "Convention on Human Rights articles most relevant to it.\nQuery: "
        ),
        doc_prefix="",
    ),
}

ACTIVE_MODEL = _env("ACTIVE_MODEL", "qwen3-emb-8b")

EMBED_BASE_URL = _env("EMBED_BASE_URL", "http://localhost:8100/v1")
TOP_K = int(_env("TOP_K", "3"))
_min_score = _env("PREDICTION_MIN_SCORE")
PREDICTION_MIN_SCORE = (
    float(_min_score) if _min_score not in (None, "", "none", "None") else None
)


def get_model(slug: str | None = None) -> ModelCfg:
    slug = slug or ACTIVE_MODEL
    if slug not in MODELS:
        raise KeyError(f"Unknown model '{slug}'. Known: {list(MODELS)}")
    return MODELS[slug]
