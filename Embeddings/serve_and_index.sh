#!/bin/bash
# Start the vLLM embedding server (background) and build the index.
#
# The embedding server stays running afterwards because the ECtHR eval also
# needs it (to embed each case query). Stop it later with:
#   kill $(cat Embeddings/vllm_embed.pid)
#
# Usage:
#   ./serve_and_index.sh [--overwrite]
# Env:
#   EMBED_MODEL   (default Qwen/Qwen3-Embedding-8B)
#   EMBED_PORT    (default 8100)
#   CUDA_VISIBLE_DEVICES  (which GPU to use, e.g. 0)

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
PORT="${EMBED_PORT:-8100}"
LOG="$HERE/vllm_embed.log"

# 1. Start the embedding server if it isn't already up.
if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "Embedding server already running on port ${PORT}."
else
    echo "Starting vLLM embedding server: ${MODEL} on port ${PORT} ..."
    # --runner pooling replaces the deprecated --task embed (vLLM >= ~0.11);
    # for an embedding model vLLM also auto-detects this via --runner auto.
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --runner pooling \
        --port "$PORT" \
        --host 0.0.0.0 \
        > "$LOG" 2>&1 &
    echo $! > "$HERE/vllm_embed.pid"
    echo "PID $(cat "$HERE/vllm_embed.pid"), log: $LOG"

    echo -n "Waiting for the model to load"
    for _ in $(seq 1 1200); do
        if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo " — up."
            break
        fi
        echo -n "."
        sleep 5
    done
    if ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo
        echo "Server did not become healthy in time. Check ${LOG}" >&2
        exit 1
    fi
fi

# 2. Build the index (embed chunks, cache into convention_chunks.json).
echo "Building index..."
python "$HERE/build_index.py" "$@"

echo
echo "Done. The embedding server is still running (needed by the eval)."
echo "Evaluate with:"
echo "  python src/llm_rl_playground/eval_embedding_search.py --label embed-topk --n-cases 100 --top-k 10"
