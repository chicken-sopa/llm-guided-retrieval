"""Evaluate plain top-k embedding retrieval on ECtHR cases.

HOW TO USE (embed server must be up + build_index.py must have run first):
    python src/llm_rl_playground/eval_embedding_search.py --label embed-topk \
        --n-cases 100 --top-k 10
Output: src/llm_rl_playground/outputs/ecthr-embed-eval/<label>_ecthr_summary.csv

Baseline for the LATTICE comparison. For each case: embed the facts, do an exact
cosine top-k over the Convention articles, treat the retrieved article ids as the
predicted alleged-violation articles, and score against the gold labels.

Scoring REUSES ecthr_evaluation.py (same gold extraction and metrics as the
LATTICE eval) so the comparison is apples-to-apples. Output schema matches the
LATTICE per-run summary. The embedding ops live in the Embeddings/ folder
(config, embed, db); this script imports them.

Run order:
    docker compose up -d db vllm-embed        # from repo root
    python Embeddings/build_index.py          # embed corpus -> pgvector
    python src/llm_rl_playground/eval_embedding_search.py --label embed-topk-qwen3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    sys.path.insert(0, str(repo_root / "src"))         # ecthr_evaluation
    sys.path.insert(0, str(repo_root / "Embeddings"))  # config, embed, db


def build_query(example: dict, max_chars: int = 8000) -> str:
    facts = example.get("facts", [])
    if isinstance(facts, str):
        facts = [facts]
    text = "\n".join(str(fact).strip() for fact in facts if str(fact).strip())
    return text[:max_chars]


def main() -> None:
    _add_paths()
    import pandas as pd

    import config
    import embed
    from store import VectorIndex
    from llm_rl_playground.ecthr_evaluation import (
        example_gold_articles,
        get_label_names,
        load_ecthr_dataset,
        normalize_article_label,
        score_prediction,
        summarize_ecthr_cases,
    )

    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Top-k embedding retrieval baseline on ECtHR.")
    parser.add_argument("--model", default=None, help="Model slug from config.MODELS (default: ACTIVE_MODEL).")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.PREDICTION_MIN_SCORE,
                        help="Cosine-similarity threshold on retrieved articles (default: none).")
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--output-dir", default=str(here.parent / "outputs" / "ecthr-embed-eval"))
    parser.add_argument("--chunks", default=None, help="Path to the embedded chunks json (default: Embeddings/echr/convention_chunks.json).")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    cfg = config.get_model(args.model)
    label = args.label or f"embed-topk-{cfg.slug}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = args.chunks or (here.parents[2] / "Embeddings" / "echr" / "convention_chunks.json")
    index = VectorIndex.from_chunks_json(chunks_path, cfg.hf_name, normalize_article_label)
    print(f"Loaded in-memory index: {index.size} articles embedded with {cfg.hf_name}.")

    dataset = load_ecthr_dataset(split=args.eval_split, config=args.ecthr_config)
    label_names = get_label_names(dataset)

    n = min(args.n_cases, len(dataset) - args.start)
    rows = []
    start_time = time.time()
    for i in range(args.start, args.start + n):
        example = dataset[i]
        gold = example_gold_articles(example, label_names)
        query_text = build_query(example)
        query_vec = embed.embed_texts([query_text], is_query=True, cfg=cfg, base_url=config.EMBED_BASE_URL)[0]

        hits = index.search(query_vec, args.top_k)
        if args.min_score is not None:
            hits = [hit for hit in hits if hit["similarity"] >= args.min_score]
        predicted = {hit["article_id"] for hit in hits}

        scores = score_prediction(gold, predicted)
        rows.append(
            {
                "index": i,
                "gold": sorted(gold),
                "predicted": sorted(predicted),
                "top_hit": hits[0]["article_id"] if hits else None,
                **scores,
            }
        )
    elapsed = time.time() - start_time

    df = pd.DataFrame(rows)
    df.to_json(output_dir / f"{label}_ecthr_eval_rows.json", orient="records", indent=2)

    summary = summarize_ecthr_cases(df)
    summary.insert(0, "elapsed_seconds", round(elapsed, 2))
    summary.insert(0, "top_k", args.top_k)
    summary.insert(0, "model", cfg.hf_name)
    summary.insert(0, "method", "embedding_topk")
    summary.insert(0, "run", label)
    summary.to_csv(output_dir / f"{label}_ecthr_summary.csv", index=False)
    summary.to_json(output_dir / f"{label}_ecthr_summary.json", orient="records", indent=2)

    print("\n================ Embedding Top-k Eval Summary ================")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {output_dir / f'{label}_ecthr_summary.csv'}")


if __name__ == "__main__":
    main()
