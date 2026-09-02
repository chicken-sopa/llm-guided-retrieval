"""Build a LATTICE semantic tree for one corpus, and keep its manifest honest.

A corpus lives in ONE self-contained folder that you can copy to another project:

    corpora/<DATASET>/<SUBSET>/
        chunks.json                  the corpus (source of truth)
        chunks_with_embeddings.json  optional: same chunks with cached vectors
        manifest.json                what is here, and how each tree was built
        trees/
            tree-<version>.pkl       what LatticeRetriever loads
            tree-<version>.json      readable copy

`LatticeRetriever.from_hp` resolves `corpora/{DATASET}/{SUBSET}/trees/tree-{TREE_VERSION}.pkl`
by default, so a tree built here is loadable by name with no path wiring.

--------------------------------------------------------------------------------
Adding a new corpus (the whole workflow)
--------------------------------------------------------------------------------
    mkdir -p corpora/MyCorpus/main
    cp my_chunks.json corpora/MyCorpus/main/chunks.json

    python build_tree.py --dataset MyCorpus --subset main \
        --text-field text --id-field id --max-children 10

Then retrieve against it:

    hp = HyperParams.from_args("--dataset MyCorpus --subset main "
                               "--tree_version bottom-up-llm "
                               "--llm_api_backend openai --llm gpt-4.1")
    retriever = LatticeRetriever.from_hp(hp)

--------------------------------------------------------------------------------
Other uses
--------------------------------------------------------------------------------
    python build_tree.py --list                      # what corpora exist
    python build_tree.py --dataset ECHR --subset convention --refresh-manifest
    python build_tree.py --dataset ECHR --subset convention \
        --tree-version bottom-up-mc4 --max-children 4   # a second tree, same corpus

Any flag this script does not define is passed through to the underlying builder
(`src/tree_construction/build_llm_bottom_up_tree.py`) untouched -- so `--llm`,
`--llm-api-backend`, `--max-leaves`, `--embedding-field`, `--summary-cache` and
friends all work here.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
CORPORA_DIR = REPO_ROOT / "corpora"
BUILDER = REPO_ROOT / "src" / "tree_construction" / "build_llm_bottom_up_tree.py"


# --------------------------------------------------------------------------
# Reading what is on disk
# --------------------------------------------------------------------------
def corpus_dir(dataset: str, subset: str) -> Path:
    return CORPORA_DIR / dataset / subset


def read_chunks(path: Path) -> tuple[str, list[dict]]:
    """Return (format, chunks). Both known layouts are accepted:
    a bare JSON list, or {"document": ..., "chunks": [...]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "list", data
    return "wrapped", data.get("chunks", [])


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_stats(path: Path) -> dict[str, int]:
    """Leaf count and depth of a serialized SemanticNode tree (children live under 'child')."""
    node = pickle.loads(path.read_bytes())

    def leaves(n: dict) -> int:
        kids = n.get("child")
        return 1 if not kids else sum(leaves(k) for k in kids)

    def depth(n: dict) -> int:
        kids = n.get("child")
        return 1 if not kids else 1 + max(depth(k) for k in kids)

    return {"leaves": leaves(node), "depth": depth(node)}


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------
def write_manifest(dataset: str, subset: str, new_build: dict[str, Any] | None = None) -> Path:
    """Rewrite manifest.json from what is actually on disk.

    Build parameters already recorded for a tree are PRESERVED -- rescanning must never
    silently drop provenance for a tree built earlier. `new_build`, when given, is
    {"path": "trees/x.pkl", "build": {...}} for the tree this run just produced.
    """
    root = corpus_dir(dataset, subset)
    manifest_path = root / "manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_builds = {t["path"]: t.get("build", {}) for t in previous.get("trees", [])}
    if new_build:
        prior_builds[new_build["path"]] = new_build["build"]

    chunks_path = root / "chunks.json"
    fmt, chunks = read_chunks(chunks_path)
    embedded = root / "chunks_with_embeddings.json"

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "subset": subset,
        "note": previous.get("note", ""),
        "generated": datetime.date.today().isoformat(),
        "chunks": {
            "path": "chunks.json",
            "format": fmt,
            "count": len(chunks),
            "md5": file_md5(chunks_path),
            "fields": sorted(chunks[0].keys()) if chunks else [],
            "id_field": previous.get("chunks", {}).get("id_field"),
            "text_field": previous.get("chunks", {}).get("text_field"),
            "embedding_field": previous.get("chunks", {}).get("embedding_field"),
            "embedded_copy": "chunks_with_embeddings.json" if embedded.exists() else None,
        },
        "trees": [],
    }

    for pkl_path in sorted((root / "trees").glob("*.pkl")):
        rel = f"trees/{pkl_path.name}"
        stats = tree_stats(pkl_path)
        version = pkl_path.stem[len("tree-"):] if pkl_path.stem.startswith("tree-") else pkl_path.stem
        manifest["trees"].append({
            "path": rel,
            "tree_version": version,
            "leaves": stats["leaves"],
            "depth": stats["depth"],
            "size_bytes": pkl_path.stat().st_size,
            "build": prior_builds.get(rel, {}),
            # A tree whose leaf count matches the corpus was built from THIS chunks file
            # in full. A mismatch means a --max-leaves cap or a different revision -- say
            # so rather than implying a provenance that was never verified.
            "provenance": ("verified: leaf count == chunk count"
                           if stats["leaves"] == len(chunks)
                           else f"UNVERIFIED: {stats['leaves']} leaves != {len(chunks)} chunks"),
        })

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return manifest_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def list_corpora() -> None:
    if not CORPORA_DIR.exists():
        print(f"no corpora directory at {CORPORA_DIR}")
        return
    for manifest_path in sorted(CORPORA_DIR.glob("*/*/manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"{m['dataset']}/{m['subset']}  ({m['chunks']['count']} chunks, "
              f"{m['chunks']['format']} format)")
        for t in m["trees"]:
            flag = "" if t["provenance"].startswith("verified") else "  [!]"
            print(f"    --tree_version {t['tree_version']:<20s} "
                  f"leaves={t['leaves']:<6d} depth={t['depth']:<3d}{flag}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a LATTICE semantic tree into corpora/<DATASET>/<SUBSET>/trees/.",
        epilog="Unrecognized flags are forwarded to build_llm_bottom_up_tree.py.")
    parser.add_argument("--dataset", help="Corpus family, e.g. ECHR / EU / BRIGHT.")
    parser.add_argument("--subset", help="Corpus within the family, e.g. convention.")
    parser.add_argument("--tree-version", default="bottom-up-llm",
                        help="Names the output tree-<version>.pkl and is what you pass as "
                             "--tree_version at retrieval time. Default: bottom-up-llm")
    parser.add_argument("--input", help="Chunks file. Default: the corpus's "
                                        "chunks_with_embeddings.json if present, else chunks.json.")
    parser.add_argument("--list", action="store_true", help="List corpora and their trees, then exit.")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="Rescan the corpus folder and rewrite manifest.json without building.")
    args, passthrough = parser.parse_known_args(argv)

    if args.list:
        list_corpora()
        return

    if not args.dataset or not args.subset:
        parser.error("--dataset and --subset are required (or use --list)")

    root = corpus_dir(args.dataset, args.subset)
    if not root.exists():
        parser.error(f"no such corpus: {root}\n"
                     f"Create it with:  mkdir -p {root} && cp <your chunks> {root / 'chunks.json'}")
    if not (root / "chunks.json").exists():
        parser.error(f"missing {root / 'chunks.json'} -- every corpus needs its chunks file")

    if args.refresh_manifest:
        print(f"manifest: {write_manifest(args.dataset, args.subset)}")
        return

    # Prefer the embedded copy: the bottom-up builder clusters on vectors, and reusing
    # cached ones avoids re-embedding the corpus for every tree.
    if args.input:
        input_path = Path(args.input)
    else:
        embedded = root / "chunks_with_embeddings.json"
        input_path = embedded if embedded.exists() else root / "chunks.json"

    out_pkl = root / "trees" / f"tree-{args.tree_version}.pkl"
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    builder_argv = [
        str(BUILDER),
        "--input", str(input_path),
        "--dataset", args.dataset,
        "--subset", args.subset,
        "--output", str(out_pkl),
        "--json-output", str(out_pkl.with_suffix(".json")),
        *passthrough,
    ]

    print(f"[build] corpus  {args.dataset}/{args.subset}")
    print(f"[build] input   {input_path.relative_to(REPO_ROOT)}")
    print(f"[build] output  {out_pkl.relative_to(REPO_ROOT)}")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tree_construction import build_llm_bottom_up_tree as builder  # noqa: E402

    saved_argv = sys.argv
    try:
        sys.argv = builder_argv
        builder.main()
    finally:
        sys.argv = saved_argv

    manifest_path = write_manifest(
        args.dataset, args.subset,
        new_build={"path": f"trees/{out_pkl.name}",
                   "build": {"input": os.path.relpath(input_path, root).replace("\\", "/"),
                             "args": passthrough,
                             "built": datetime.date.today().isoformat()}},
    )
    print(f"[build] manifest {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
