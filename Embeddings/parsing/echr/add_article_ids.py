"""Stamp a canonical `id` onto every ECHR chunk.

The id is produced by the SAME normalizer the evaluation uses for gold labels
(`normalize_article_label`), so leaf ids and gold ids live in one ID space:
    Convention  Article N   -> "article_N"
    Protocol P  Article N    -> "protocol_P_article_N"

This is what lets the eval read a leaf's article identity from node.id instead of
regex-scraping its prose (which loses the protocol and is format-fragile).

Run after (re)generating the chunk files; it edits them in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_rl_playground.ecthr_evaluation import normalize_article_label  # noqa: E402

# Files to stamp. The embedded file is the one the ablation actually consumes;
# the plain file is kept in sync so a re-embed stays consistent. (Both now live in
# the dataset folder; the former third copy was a byte-identical duplicate.)
DATASET = REPO_ROOT / "corpora/ECHR/convention"
TARGETS = [
    DATASET / "chunks_with_embeddings.json",
    DATASET / "chunks.json",
]


def canonical_id(chunk: dict) -> str:
    """Build the normalized article id for one chunk."""
    protocol = str(chunk.get("protocol", "")).strip()
    article = chunk["article_number"]
    if protocol.lower().startswith("protocol"):
        # "Protocol 16" -> "Protocol 16 Article 5"
        source = f"{protocol} Article {article}"
    else:
        # Convention -> "Article 5"
        source = f"Article {article}"

    article_id = normalize_article_label(source)
    if not article_id:
        raise ValueError(f"Could not normalize id for chunk: protocol={protocol!r} article={article!r}")
    return article_id


def stamp(path: Path) -> None:
    if not path.exists():
        print(f"skip (not found): {path}")
        return

    chunks = json.loads(path.read_text(encoding="utf-8"))
    ids = [canonical_id(c) for c in chunks]

    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(f"Duplicate ids in {path.name}: {sorted(duplicates)}")

    for chunk, article_id in zip(chunks, ids):
        # `id` is the field infer_record_id() checks first, so the builder will
        # use it as the leaf node id automatically (no --id-field needed).
        chunk["id"] = article_id

    path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"stamped {len(chunks)} ids -> {path}")


def main() -> None:
    for path in TARGETS:
        stamp(path)


if __name__ == "__main__":
    main()
