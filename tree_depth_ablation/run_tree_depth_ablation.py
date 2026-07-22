"""Tree-depth ablation for LATTICE on the ECtHR task.

WHAT THIS DOES
    For each fanout setting in --max-children:
      1. builds a semantic tree over the ECHR article chunks,
      2. measures the tree that actually came out (depth / nodes / leaves),
      3. runs the LATTICE ECtHR eval on it with num_iters = depth + 1,
      4. records the metrics plus the cost of that run.
    Every run is appended to ONE csv so the depths can be compared directly.

WHY num_iters = depth + 1
    LATTICE expands one tree level per iteration, so reaching the leaf articles
    in a depth-D tree needs at least D iterations (+1 for the leaf-level scoring
    pass). A fixed iteration budget across depths would make deep trees unable to
    reach any article at all, so the budget has to scale with depth. Because that
    also makes deeper trees cost more, the output table reports cost columns
    (llm calls / elapsed) next to quality — the comparison is quality-vs-cost.

NOTE ON DEPTH
    The tree builder has no --max-depth: it is bottom-up clustering with a
    fanout cap (--max-children), so depth is EMERGENT. We sweep max_children and
    MEASURE the resulting depth. Smaller fanout => deeper tree.

REQUIREMENTS
    - The chunks file must already contain embeddings (key "embeddings"). The
      bottom-up builder clusters on vectors and does NOT compute them. Embed the
      corpus once and the vectors are reused for every tree in the sweep:
          python Embeddings/build_index.py --chunks src/echr/convention_chunks.json
    - A vLLM server must already be serving --model. The script refuses to start
      until /health responds and the model is listed (both the tree builder and
      LATTICE call the LLM).

USAGE
    python tree_depth_ablation/run_tree_depth_ablation.py \
        --model "$LATTICE_VLLM_MODEL" --base-url "$VLLM_BASE_URL" \
        --max-children 16 10 4 3 2 --n-cases 50

    Trees are cached in tree_depth_ablation/trees/, per-config eval outputs in
    tree_depth_ablation/output/, and the aggregated result in
    tree_depth_ablation/tree_depth_ablation.csv.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Outputs live next to this script (survives renaming the folder).
SCRIPT_DIR = Path(__file__).resolve().parent


def _add_paths() -> None:
    # Unpickling a tree may need SemanticNode importable from src/.
    sys.path.insert(0, str(REPO_ROOT / "src"))


def require_embedded_chunks(chunks_path: Path, embedding_field: str = "embeddings") -> None:
    """Abort early unless every chunk already carries an embedding.

    The tree builder assumes embeddings are present (it clusters on them), so we
    verify up front rather than letting it fail per-record inside a subprocess.
    """
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    missing = [i for i, c in enumerate(chunks) if not c.get(embedding_field)]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(chunks)} chunks in {chunks_path} have no "
            f"'{embedding_field}'. Embed them once first, e.g.:\n"
            f"    python Embeddings/build_index.py --chunks {chunks_path}\n"
            "(embeddings are cached in the file and reused for every tree)."
        )


# --------------------------------------------------------------------------
# vLLM gate: nothing runs until the server is actually serving the model.
# --------------------------------------------------------------------------
def wait_for_vllm(base_url: str, model: str, timeout: int = 900, interval: int = 10) -> None:
    """Block until vLLM answers /health AND lists `model`. Raise on timeout."""
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                healthy = resp.status == 200
        except Exception:
            healthy = False

        if healthy:
            # /health can be up before the model is registered — verify the id too.
            try:
                with urllib.request.urlopen(models_url, timeout=5) as resp:
                    served = [m["id"] for m in json.loads(resp.read())["data"]]
                if model in served:
                    print(f"[vllm] online, serving {model}")
                    return
                print(f"[vllm] healthy but '{model}' not served yet. Available: {served}")
            except Exception as exc:
                print(f"[vllm] /models not readable yet ({exc})")

        print(f"[vllm] waiting for {health_url} ...")
        time.sleep(interval)

    raise SystemExit(f"vLLM did not become ready within {timeout}s at {base_url}")


# --------------------------------------------------------------------------
# Tree measurement: depth is emergent, so read it off the built tree.
# --------------------------------------------------------------------------
def _children(node):
    """Children accessor that works for both dict-serialized and object trees."""
    if isinstance(node, dict):
        return node.get("child") or []
    return getattr(node, "child", None) or []


def tree_stats(tree_path: Path) -> dict:
    """Return depth / node count / leaf count of a built tree."""
    _add_paths()
    with tree_path.open("rb") as f:
        root = pickle.load(f)

    nodes = leaves = 0
    max_depth = 0

    def walk(node, depth):
        nonlocal nodes, leaves, max_depth
        nodes += 1
        max_depth = max(max_depth, depth)
        kids = _children(node)
        if not kids:
            leaves += 1
            return
        for kid in kids:
            walk(kid, depth + 1)

    walk(root, 0)
    return {"depth": max_depth, "num_nodes": nodes, "num_leaves": leaves}


# --------------------------------------------------------------------------
# Step 1: build one tree per fanout setting (cached — building costs LLM calls).
# --------------------------------------------------------------------------
def build_tree(max_children: int, args, tree_path: Path) -> float:
    if tree_path.exists() and not args.rebuild:
        print(f"[tree] reusing cached {tree_path.name}")
        return 0.0

    cmd = [
        sys.executable, str(REPO_ROOT / "src/tree_construction/build_llm_bottom_up_tree.py"),
        "--input", str(args.chunks),
        "--output", str(tree_path),
        "--max-children", str(max_children),
        # Reuse the embeddings already cached in convention_chunks.json so the
        # ablation does not re-embed the corpus for every fanout setting.
        "--text-field", "text",
        "--embedding-field", "embeddings",
        "--llm-api-backend", "vllm",
        "--llm", args.model,
        "--vllm-base-url", args.base_url,
        "--summary-cache", str(args.tree_dir / f"summary_cache_mc{max_children}.json"),
    ]
    print(f"[tree] building max_children={max_children} -> {tree_path.name}")
    start = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - start


# --------------------------------------------------------------------------
# Step 2: run the LATTICE ECtHR eval against that tree.
# --------------------------------------------------------------------------
def run_lattice(tree_path: Path, num_iters: int, label: str, args) -> Path:
    cmd = [
        sys.executable, str(REPO_ROOT / "src/llm_rl_playground/test_ecthr_models.py"),
        "--backend", "vllm",
        "--model", args.model,
        "--base-url", args.base_url,
        "--tree-path", str(tree_path),
        "--num-iters", str(num_iters),      # = measured depth + 1
        "--n-cases", str(args.n_cases),
        "--eval-split", args.eval_split,
        "--label", label,
        "--output-dir", str(args.run_dir),
    ]
    print(f"[lattice] {label}: num_iters={num_iters}, n_cases={args.n_cases}")
    subprocess.run(cmd, check=True)
    return args.run_dir / f"{label}_ecthr_summary.csv"


def coverage_from_rows(rows_path: Path) -> float | None:
    """Fraction of cases that produced at least one predicted article.

    This is the validity check: traversal MUST reach leaf articles. A coverage of
    ~0 means the iteration budget never got to the leaves, so that row is not a
    real 'deep trees are bad' result — it is an invalid run.
    """
    try:
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        if not rows:
            return None
        keys = [k for k, v in rows[0].items() if "predict" in k.lower() and isinstance(v, list)]
        if not keys:
            return None
        key = keys[0]
        return sum(1 for r in rows if r.get(key)) / len(rows)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LATTICE tree-depth ablation on ECtHR.")
    parser.add_argument("--max-children", type=int, nargs="+", default=[16, 10, 4, 3, 2],
                        help="Fanout caps to sweep. Smaller => deeper tree.")
    parser.add_argument("--chunks", default=str(REPO_ROOT / "src/echr/convention_chunks.json"))
    parser.add_argument("--model", required=True, help="Model id as served by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--n-cases", type=int, default=50)
    parser.add_argument("--eval-split", default="validation")
    # Trees and outputs live inside this script's folder.
    parser.add_argument("--tree-dir", type=Path, default=SCRIPT_DIR / "trees")
    parser.add_argument("--run-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "tree_depth_ablation.csv")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild trees even if cached.")
    parser.add_argument("--vllm-timeout", type=int, default=900)
    args = parser.parse_args()

    import pandas as pd

    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The bottom-up builder clusters on vectors, so the chunks MUST already carry
    # embeddings (embed them once with build_index; they are reused for every
    # tree in the sweep). Fail early and clearly instead of erroring per-record
    # deep inside the builder subprocess.
    require_embedded_chunks(Path(args.chunks))

    # Gate the whole sweep on vLLM being up (tree building needs it too).
    wait_for_vllm(args.base_url, args.model, timeout=args.vllm_timeout)

    records = []
    for max_children in args.max_children:
        tree_path = args.tree_dir / f"tree-mc{max_children}.pkl"
        record = {"max_children": max_children}
        try:
            # 1. build (cached) and 2. measure the tree that resulted
            record["build_seconds"] = round(build_tree(max_children, args, tree_path), 1)
            stats = tree_stats(tree_path)
            record.update(stats)

            # 3. iterations follow the MEASURED depth
            num_iters = stats["depth"] + 1
            record["num_iters"] = num_iters
            label = f"mc{max_children}-depth{stats['depth']}"
            record["run"] = label

            # vLLM can die mid-sweep; re-check before each (long) eval.
            wait_for_vllm(args.base_url, args.model, timeout=args.vllm_timeout)
            summary_path = run_lattice(tree_path, num_iters, label, args)

            # 4. pull the metrics + the validity check
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
            summary.pop("run", None)  # keep our own label
            record.update(summary)
            record["coverage"] = coverage_from_rows(args.run_dir / f"{label}_ecthr_eval_rows.json")
            record["status"] = "ok"
        except Exception as exc:
            # Never abort the sweep: record the failure and continue.
            record["status"] = f"failed: {exc}"
            print(f"[error] max_children={max_children}: {exc}", file=sys.stderr)

        records.append(record)
        # Write after every config so a crash never loses completed runs.
        pd.DataFrame(records).to_csv(args.out, index=False)

    df = pd.DataFrame(records).sort_values("depth", na_position="last")
    df.to_csv(args.out, index=False)
    print("\n================ Tree-depth ablation ================")
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out}")
    print("Note: rows with coverage ~0 never reached leaf articles — treat as invalid.")


if __name__ == "__main__":
    main()
