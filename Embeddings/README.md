# Embeddings/

Embedding operations for the dense-retrieval baseline: **create** embeddings,
**store** them in pgvector, and **search** them.

| File | Role |
|------|------|
| `embed.py` | embedding client (vLLM `/v1/embeddings`); query vs document prefix |
| `db.py` | pgvector connect / upsert / exact top-k search |
| `build_index.py` | embed `echr/convention_chunks.json` → cache in the json → upsert to pgvector |
| `config.py` | model registry + DB DSN |
| `echr/convention_chunks.json` | corpus (one chunk per article; cached embeddings stored here) |

Infrastructure (`docker-compose.yml`, `sql/`) lives at the repo root. The ECtHR
evaluation that *uses* these embeddings is
`src/llm_rl_playground/eval_embedding_search.py`.

## Run

```bash
# from repo root: start Postgres + the embedding server
docker compose up -d db vllm-embed

# embed the corpus into pgvector (cached back into the chunks json)
python Embeddings/build_index.py          # --overwrite to recompute, --no-db to only cache

# evaluate the top-k baseline on ECtHR (reuses ecthr_evaluation scoring)
python src/llm_rl_playground/eval_embedding_search.py --label embed-topk-qwen3 --n-cases 100 --top-k 10
```

Config defaults (Postgres `echr/echr/echr`, embed server `localhost:8100`) match
`docker-compose.yml`, so no `.env` is needed unless you override them.
