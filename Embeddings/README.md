# Embeddings/

Embedding operations for the dense-retrieval baseline: **create** embeddings and
**search** them in memory. No database, no Docker — the corpus is ~116 articles,
so exact cosine top-k with NumPy is instant.

| File | Role |
|------|------|
| `embed.py` | embedding client (vLLM `/v1/embeddings`); query vs document prefix |
| `store.py` | in-memory vector index: load embedded chunks → NumPy cosine top-k |
| `build_index.py` | embed `echr/convention_chunks.json` and cache the vectors in it |
| `config.py` | model registry (query/doc prefixes) |
| `serve_and_index.sh` | start the vLLM embed server + build the index |
| `echr/convention_chunks.json` | corpus (one chunk per article; cached embeddings live here) |

The ECtHR evaluation that *uses* these embeddings is
`src/llm_rl_playground/eval_embedding_search.py`.

## How it works (no DB)

`build_index.py` writes each article's embedding into `convention_chunks.json`
under the `embeddings` key — **that json is the index**. The eval loads it into a
NumPy matrix and does cosine top-k in memory. The only running service is the
vLLM embedding server (used to embed documents at build time and queries at eval
time).

## Run

```bash
# 1. start the embed server + build the index (one script)
CUDA_VISIBLE_DEVICES=0 ./Embeddings/serve_and_index.sh
#    (leaves the embed server running; --overwrite to recompute embeddings)

# 2. evaluate the top-k baseline on ECtHR (reuses ecthr_evaluation scoring)
python src/llm_rl_playground/eval_embedding_search.py --label embed-topk --n-cases 100 --top-k 10

# stop the embed server when done
kill $(cat Embeddings/vllm_embed.pid)
```

Or start the server yourself and run the steps manually:
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Embedding-8B --task embed --port 8100 &
python Embeddings/build_index.py
python src/llm_rl_playground/eval_embedding_search.py --label embed-topk
```
