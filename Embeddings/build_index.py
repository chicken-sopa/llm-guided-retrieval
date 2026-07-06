"""Build the index: embed convention_chunks.json and cache the vectors in it.

HOW TO USE (needs a vLLM embedding server running — see serve_and_index.sh):
    python Embeddings/build_index.py               # embed missing chunks + cache
    python Embeddings/build_index.py --overwrite   # recompute all embeddings
Then evaluate with src/llm_rl_playground/eval_embedding_search.py.

Source: echr/convention_chunks.json — a list of chunks, each with
`protocol`, `article_number`, `title`, `text`.

Embeddings are cached back into that JSON under the `embeddings` key (with
`embedding_model` recording which model produced them). That json IS the index —
eval_embedding_search.py loads it and does in-memory cosine top-k (no database).

On rerun, chunks that already have embeddings for the active model are reused;
pass --overwrite to recompute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_paths() -> None:
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))  # Embeddings/  (config, embed)


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_chunks(path: Path, chunks: list[dict]) -> None:
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def has_cached_embedding(chunk: dict, model_name: str) -> bool:
    return bool(chunk.get("embeddings")) and chunk.get("embedding_model") == model_name


def embed_chunks(chunks: list[dict], cfg, base_url: str, overwrite: bool) -> bool:
    """Embed chunk text into chunk['embeddings'] (cache). Returns True if changed.

    A chunk is (re)embedded when --overwrite is set, when it has no embeddings,
    or when its cached embeddings were produced by a different model.
    """
    import embed

    todo = [
        i
        for i, chunk in enumerate(chunks)
        if overwrite or not has_cached_embedding(chunk, cfg.hf_name)
    ]
    if not todo:
        return False

    vectors = embed.embed_texts(
        [chunks[i]["text"] for i in todo],
        is_query=False,
        cfg=cfg,
        base_url=base_url,
    )
    for i, vector in zip(todo, vectors):
        chunks[i]["embeddings"] = vector
        chunks[i]["embedding_model"] = cfg.hf_name
    return True


def main() -> None:
    _add_paths()
    import config

    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Embed convention_chunks.json and cache vectors in it.")
    parser.add_argument("--model", default=None, help="Model slug from config.MODELS (default: ACTIVE_MODEL).")
    parser.add_argument("--chunks", default=str(here.parent / "echr" / "convention_chunks.json"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute embeddings even if already cached.")
    args = parser.parse_args()

    cfg = config.get_model(args.model)
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} not found.")
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks in {chunks_path}.")

    if all(has_cached_embedding(chunk, cfg.hf_name) for chunk in chunks) and not args.overwrite:
        dim = len(chunks[0]["embeddings"])
        print(f"All {len(chunks)} chunks already have embeddings for {cfg.hf_name} (dim={dim}); nothing to do.")
        return

    print(f"Embedding {len(chunks)} chunks for {cfg.hf_name} (overwrite={args.overwrite})...")
    if embed_chunks(chunks, cfg, config.EMBED_BASE_URL, overwrite=args.overwrite):
        save_chunks(chunks_path, chunks)
        dim = len(chunks[0]["embeddings"])
        print(f"Cached embeddings (dim={dim}) into {chunks_path}")
        print("Index ready. Run src/llm_rl_playground/eval_embedding_search.py to evaluate.")


if __name__ == "__main__":
    main()
