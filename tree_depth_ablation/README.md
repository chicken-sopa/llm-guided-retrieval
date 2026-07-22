# tree_depth_ablation/

Tree-depth ablation for LATTICE on the ECtHR task.

```
tree_depth_ablation/
├── run_tree_depth_ablation.py   # the driver
├── trees/                       # one built tree per fanout (cached)
├── output/                      # per-config LATTICE eval outputs
└── tree_depth_ablation.csv      # ALL configs in one table (the result)
```

Corpus: `Embeddings/echr/convention_chunks.json` (ECHR article chunks **with
cached embeddings** — the builder clusters on those vectors).

## What it does

For each `--max-children` value it builds a tree over the ECHR article chunks,
**measures the depth that resulted**, runs the LATTICE ECtHR eval with
`num_iters = depth + 1`, and appends everything to one CSV.

## Depth is emergent, not requested

The tree builder has no `--max-depth` — it is bottom-up clustering with a fanout
cap. Smaller `--max-children` produces a deeper tree, so the sweep varies fanout
and the script reads the real depth off each built tree.

| `--max-children` | approx. depth (116 articles) |
|---|---|
| 16 | ~2 |
| 10 | ~3 |
| 4  | ~4 |
| 3  | ~5 |
| 2  | ~7 |

## Why `num_iters = depth + 1`

LATTICE expands one level per iteration, so a depth-D tree needs at least D
iterations to reach the leaf articles (+1 for the leaf scoring pass). A fixed
budget across depths would leave deep trees unable to reach any article. Because
that makes deeper trees cost more, the table reports **cost next to quality**
(`build_seconds`, `elapsed_seconds`, `num_iters`) — read it as a quality/cost
curve, not quality alone.

## Validity check

`coverage` = fraction of cases that produced at least one predicted article.
**A row with coverage ≈ 0 never reached the leaves** — that is an invalid run
(budget too small), not evidence that deep trees are bad. Discard those.

## Run

Prerequisites:
1. **Embed the corpus once** — the bottom-up builder clusters on vectors and does
   not compute them, so the chunks must carry embeddings. They are cached in
   `Embeddings/echr/convention_chunks.json` and reused for every tree in the sweep:
   ```bash
   python Embeddings/build_index.py
   ```
   (The script checks this up front and aborts with instructions if missing.)
2. **Start a vLLM server** — the script refuses to begin until `/health` responds
   *and* the model is listed (tree building calls the LLM too).

```bash
python tree_depth_ablation/run_tree_depth_ablation.py \
  --model "$LATTICE_VLLM_MODEL" --base-url "$VLLM_BASE_URL" \
  --max-children 16 10 4 3 2 --n-cases 50
```

Useful flags: `--rebuild` (ignore cached trees), `--n-cases` (sweep small, then
re-run the best depths at 200), `--out` (result path).

Trees are cached in `trees/`, so re-running only repeats the evals. The CSV is
rewritten after every config, so a crash never loses completed runs.
