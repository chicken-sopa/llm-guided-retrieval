# Embedding-search baseline (vs LATTICE) on ECtHR

Plain **top-k embedding retrieval** over the European Convention on Human Rights
articles, evaluated on the ECtHR cases dataset with the **same scoring** as the
LATTICE eval — so the two are directly comparable.

```
Embeddings/
├── docker-compose.yml          # pgvector + vLLM embedding server
├── .env.example                # copy to .env
├── requirements.txt            # host-side python deps
├── sql/init.sql                # CREATE EXTENSION vector
├── data/                       # put Convention_ENG.pdf here; parser writes the jsonl
├── parsing/parse_convention.py # PDF -> one chunk per article
├── config.py                   # model registry + DB DSN (single model now)
├── embed.py                    # vLLM /v1/embeddings client
├── db.py                       # pgvector connect / upsert / exact top-k search
├── build_index.py              # articles -> embeddings -> pgvector
└── eval_embedding_search.py    # cases -> embed query -> search -> score
```

## Design notes

- **Postgres + pgvector is the vector DB** (one container, not two).
- **Exact search, no ANN index** — the Convention is ~60 articles, so brute-force
  cosine is instant and sidesteps pgvector's ~2000-dim index cap (any embedding
  dim works).
- **Embedding dim follows the model** — probed at build time; the per-model table
  is created with that dimension.
- **Scoring is reused** from `src/llm_rl_playground/ecthr_evaluation.py` (gold
  extraction + precision/recall/F1), so this baseline and LATTICE are scored
  identically. Output schema matches the LATTICE per-run summary.
- **Single model now, multi-model later**: `config.MODELS` is a registry keyed by
  slug, each with its own table. Add entries + run with `--model <slug>` to sweep.

## Setup

```bash
cp .env.example .env                       # adjust if needed
pip install -r requirements.txt            # host deps
# download the source PDF into data/
curl -L https://www.echr.coe.int/Documents/Convention_ENG.pdf -o data/Convention_ENG.pdf
```

## Run order

```bash
# 1. start the stores (Postgres + embedding server)
docker compose up -d db vllm-embed

# 2. parse the Convention PDF into per-article chunks
python parsing/parse_convention.py
#    -> data/convention_articles.jsonl

# 3. embed the articles into pgvector
PYTHONPATH=../src python build_index.py

# 4. run the top-k retrieval baseline on ECtHR (reuses LATTICE scoring)
PYTHONPATH=../src python eval_embedding_search.py --label embed-topk-qwen3 \
    --n-cases 100 --top-k 10
#    -> outputs/embed-topk-qwen3_ecthr_summary.{csv,json}
```

## Comparing against LATTICE

Run the LATTICE ECtHR eval (`src/llm_rl_playground/test_ecthr_models.py`) and this
baseline with the **same** `--top-k`, `--n-cases`, and `--eval-split`. Both write
a summary with `mean_recall / mean_precision / mean_f1 / any_gold_found`, so you
can place the rows side by side.

## Caveats

- **Corpus parity**: LATTICE's EU eval uses a tree built from
  `src/tree_construction/EU_conventions_example/Convention_ENG_chunks.json`.
  Validate that this parser's article set matches it, or note any difference —
  otherwise the comparison mixes retrieval method with corpus.
- **PDF parsing is heuristic**: check the parsed article count/ids after step 2.
- **GPU**: the embedding server reserves one GPU. Run the index build, then (if
  VRAM is tight) stop `vllm-embed` before launching a LATTICE generation cluster.
