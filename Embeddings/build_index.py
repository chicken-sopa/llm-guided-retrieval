"""Build the embedding index: parsed articles -> embeddings -> pgvector.

Run order:
    1. parsing/parse_convention.py   (produces data/convention_articles.jsonl)
    2. build_index.py                (this file)

The embedding dimension is probed from the model unless ModelCfg.dim is set, so
the table is created with whatever dimension the chosen model produces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_paths() -> None:
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))            # Embeddings/  (config, embed, db)
    sys.path.insert(0, str(here.parents[1] / "src"))  # repo src/ (if needed)


def main() -> None:
    _add_paths()
    import config
    import db
    import embed

    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Embed parsed Convention articles into pgvector.")
    parser.add_argument("--model", default=None, help="Model slug from config.MODELS (default: ACTIVE_MODEL).")
    parser.add_argument("--jsonl", default=str(here.parent / "parsing" / "echr" / "convention_articles.jsonl"))
    args = parser.parse_args()

    cfg = config.get_model(args.model)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"{jsonl_path} not found — run parsing/parse_convention.py first.")
    articles = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not articles:
        raise SystemExit(f"No articles in {jsonl_path}.")

    dim = cfg.dim or embed.probe_dim(cfg, config.EMBED_BASE_URL)
    print(f"Model: {cfg.hf_name}  |  dim: {dim}  |  articles: {len(articles)}")

    print("Embedding documents...")
    vectors = embed.embed_texts(
        [article["content"] for article in articles],
        is_query=False,
        cfg=cfg,
        base_url=config.EMBED_BASE_URL,
    )

    conn = db.connect(config.dsn())
    db.ensure_table(conn, cfg.table, dim)
    rows = [{**article, "embedding": vector} for article, vector in zip(articles, vectors)]
    db.upsert_documents(conn, cfg.table, rows)
    total = db.count_rows(conn, cfg.table)
    conn.close()

    print(f"Upserted {len(rows)} rows into {cfg.table} (table now has {total} rows).")


if __name__ == "__main__":
    main()
