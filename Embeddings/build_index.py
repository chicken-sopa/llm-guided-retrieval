"""Build the embedding index from convention_chunks.json.

Source: parsing/echr/convention_chunks.json — a list of chunks, each with
`protocol`, `article_number`, `title`, `text`.

Embeddings are CACHED back into that JSON under the `embeddings` key (with
`embedding_model` recording which model produced them). On rerun, chunks that
already have embeddings for the active model are reused; pass --overwrite to
recompute. Then all chunks are upserted into pgvector.

Run order:
    1. (have parsing/echr/convention_chunks.json ready)
    2. build_index.py        # embed (cache into json) -> pgvector
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _add_paths() -> None:
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))              # Embeddings/  (config, embed, db)
    sys.path.insert(0, str(here.parents[1] / "src"))  # repo src/ (ecthr_evaluation)


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_chunks(path: Path, chunks: list[dict]) -> None:
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


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


def has_cached_embedding(chunk: dict, model_name: str) -> bool:
    return bool(chunk.get("embeddings")) and chunk.get("embedding_model") == model_name


def embed_chunks(chunks: list[dict], cfg, base_url: str, overwrite: bool) -> bool:
    """Embed chunk text into chunk['embeddings'] (cache). Returns True if anything changed.

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
    import db
    from llm_rl_playground.ecthr_evaluation import normalize_article_label

    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Embed convention_chunks.json into pgvector (cached in the json).")
    parser.add_argument("--model", default=None, help="Model slug from config.MODELS (default: ACTIVE_MODEL).")
    parser.add_argument("--chunks", default=str(here.parent / "parsing" / "echr" / "convention_chunks.json"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute embeddings even if already cached.")
    parser.add_argument("--no-db", action="store_true", help="Only compute/cache embeddings; skip the pgvector upsert.")
    args = parser.parse_args()

    cfg = config.get_model(args.model)
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} not found.")
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks in {chunks_path}.")

    # Embed only what's needed; cache back into the json.
    if all(has_cached_embedding(chunk, cfg.hf_name) for chunk in chunks) and not args.overwrite:
        print(f"All {len(chunks)} chunks already have embeddings for {cfg.hf_name}; using cache.")
    else:
        print(f"Embedding chunks for {cfg.hf_name} (overwrite={args.overwrite})...")
        if embed_chunks(chunks, cfg, config.EMBED_BASE_URL, overwrite=args.overwrite):
            save_chunks(chunks_path, chunks)
            print(f"Cached embeddings into {chunks_path}")

    dim = len(chunks[0]["embeddings"])

    if args.no_db:
        print(f"--no-db set; embeddings cached, skipping pgvector upsert. dim={dim}")
        return

    rows = []
    skipped = 0
    for chunk in chunks:
        article_id = chunk_article_id(chunk, normalize_article_label)
        if article_id is None:
            skipped += 1
            continue
        protocol = str(chunk.get("protocol", "")).strip()
        part = "convention" if protocol.lower() in ("", "convention") else protocol.lower().replace(" ", "_")
        rows.append(
            {
                "article_id": article_id,
                "part": part,
                "title": chunk.get("title", ""),
                "content": chunk.get("text", ""),
                "embedding": chunk["embeddings"],
            }
        )
    if skipped:
        print(f"Warning: skipped {skipped} chunk(s) with no resolvable article_id.")

    conn = db.connect(config.dsn())
    db.ensure_table(conn, cfg.table, dim)
    db.upsert_documents(conn, cfg.table, rows)
    total = db.count_rows(conn, cfg.table)
    conn.close()
    print(f"Upserted {len(rows)} rows into {cfg.table} (table now has {total}). dim={dim}")


if __name__ == "__main__":
    main()
