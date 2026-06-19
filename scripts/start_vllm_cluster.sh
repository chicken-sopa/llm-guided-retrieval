#!/bin/bash
# Script to start vLLM servers in data parallel or tensor parallel mode
#
# Usage:
#   ./start_vllm_cluster.sh [OPTIONS]
#
# Options (all optional; unspecified ones use the default):
#   --model NAME              HF model id to serve            (default: Qwen/Qwen3-VL-8B-Instruct)
#   --vllm-mode data|tensor   Parallelism mode                (default: data)
#   --gpu-ids LIST            Comma-separated GPU ids         (default: 0,1,2,3)
#   --num-gpus N              Number of GPUs to use           (default: count of --gpu-ids)
#   --base-port PORT          First server port               (default: 8000)
#   --gpu-mem-util FLOAT      GPU memory utilization 0-1      (default: 0.90)
#   --max-model-len N         Max context length              (default: 16384)
#   --max-num-seqs N          Max concurrent seqs per server  (default: 32)
#   --enforce-eager auto|true|false      Skip CUDA graph capture (default: auto)
#   --disable-custom-ar auto|true|false  Use NCCL all-reduce     (default: auto)
#   -h, --help                Show this help and exit
#
# Examples:
#   ./start_vllm_cluster.sh
#   ./start_vllm_cluster.sh --model "Qwen/Qwen3-VL-8B-Instruct" --vllm-mode data
#   ./start_vllm_cluster.sh --model "Qwen/Qwen3.6-27B-FP8" --vllm-mode tensor --gpu-ids 0,1 --num-gpus 2
#   ./start_vllm_cluster.sh --model "Qwen/Qwen3.6-27B-FP8" --vllm-mode data --gpu-ids 0,1 --num-gpus 2  --gpu-mem-util 0.95 --max-model-len 8192 --max-num-seqs 128
#   ./start_vllm_cluster.sh --model "Qwen/Qwen3.6-27B-FP8" --vllm-mode tensor --gpu-ids 0,1 --disable-custom-ar false
#
# enforce-eager (auto|true|false, default "auto"):
#   auto  - enabled in tensor mode (recommended for large models)
#   true  - always skip CUDA graph compilation
#   false - allow CUDA graph compilation (may hang if compilation takes >60s)
#
# disable-custom-ar (auto|true|false, default "auto"):
#   auto  - enabled in tensor mode (recommended; avoids Blackwell hangs)
#   true  - always disable custom all-reduce (use NCCL all-reduce)
#   false - keep vLLM's custom all-reduce (may hang on some GPUs/topologies)
#
# Modes:
#   data   - Run multiple servers (one per GPU) for maximum throughput
#   tensor - Run single server with model split across GPUs for large models

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_usage() {
    sed -n '2,39p' "$0" | sed 's/^#\s\?//'
}

# ---- Defaults ----
MODEL="Qwen/Qwen3-VL-8B-Instruct"
MODE="data"                 # "data" or "tensor"
BASE_PORT=8000
GPU_MEM_UTIL=0.90
GPU_IDS_RAW="0,1,2,3"        # comma-separated GPU ids, e.g. "0,1,3,4"
NUM_GPUS=""                 # empty => derive from GPU_IDS count
MAX_MODEL_LEN=16384
# Max concurrent sequences per server. Lower values shrink the CUDA-graph
# capture / activation peak, which prevents OOM when weights nearly fill the GPU
# (e.g. FP8-27B on a 48GB card).
MAX_NUM_SEQS=32
# Whether to skip CUDA graph compilation (avoids EngineCore IPC timeout on large models)
ENFORCE_EAGER="auto"
# Whether to disable vLLM's custom all-reduce kernel (falls back to NCCL).
# Custom all-reduce can deadlock on some GPUs/topologies (e.g. Blackwell sm_100),
# causing tensor parallel to hang silently after weight loading.
DISABLE_CUSTOM_AR="auto"

# ---- Parse named flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL="$2"; shift 2 ;;
        --vllm-mode|--mode)  MODE="$2"; shift 2 ;;
        --gpu-ids)           GPU_IDS_RAW="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        --base-port)         BASE_PORT="$2"; shift 2 ;;
        --gpu-mem-util)      GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs)      MAX_NUM_SEQS="$2"; shift 2 ;;
        --enforce-eager)     ENFORCE_EAGER="$2"; shift 2 ;;
        --disable-custom-ar) DISABLE_CUSTOM_AR="$2"; shift 2 ;;
        -h|--help)           print_usage; exit 0 ;;
        *)
            echo -e "${RED}Error: unknown option '$1'${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1 ;;
    esac
done

# Derive GPU array / count
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"

if [[ "$NUM_GPUS" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Warning: NUM_GPUS ($NUM_GPUS) doesn't match number of GPU_IDS passed (${#GPU_IDS[@]}): ${GPU_IDS[*]}" >&2
fi

# Validate mode
if [[ "$MODE" != "data" && "$MODE" != "tensor" ]]; then
    echo -e "${RED}Error: --vllm-mode must be 'data' or 'tensor', got: $MODE${NC}"
    echo "Run '$0 --help' for usage."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Save mode to file for stop script
echo "$MODE" > logs/cluster_mode.txt

if [[ "$MODE" == "tensor" ]]; then
    # ============================================
    # TENSOR PARALLEL MODE
    # ============================================
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Tensor Parallel Server${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${GPU_IDS[@]}"
    echo -e "Tensor Parallel Size: ${NUM_GPUS}"
    echo -e "Port: ${BASE_PORT}"
    echo -e "${BLUE}================================================${NC}\n"

    # Build GPU list as comma-separated string
    GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
    LOG_FILE="logs/vllm_tensor_port${BASE_PORT}.log"

    echo -e "${GREEN}Starting tensor parallel server on GPUs: $GPU_LIST${NC}"

    # For large models in tensor parallel mode, CUDA graph compilation can take
    # several minutes, causing vLLM's EngineCore IPC watchdog to time out and
    # report "No available shared memory broadcast block found in 60 seconds".
    # --enforce-eager bypasses graph compilation entirely (slight throughput cost).
    EAGER_FLAG=""
    if [[ "$ENFORCE_EAGER" == "auto" || "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
        EAGER_FLAG="--enforce-eager"
        echo -e "${YELLOW}Note: enforce-eager enabled — CUDA graph compilation skipped.${NC}"
    fi

    # vLLM's custom all-reduce kernel uses CUDA IPC peer access. On some GPUs
    # (notably Blackwell sm_100) it deadlocks during the first all-reduce, making
    # tensor parallel hang silently right after weight loading. Falling back to
    # NCCL avoids this with negligible throughput cost for beam-search workloads.
    CUSTOM_AR_FLAG=""
    if [[ "$DISABLE_CUSTOM_AR" == "auto" || "$DISABLE_CUSTOM_AR" == "1" || "$DISABLE_CUSTOM_AR" == "true" ]]; then
        CUSTOM_AR_FLAG="--disable-custom-all-reduce"
        echo -e "${YELLOW}Note: custom all-reduce disabled — using NCCL all-reduce.${NC}"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_LIST nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $BASE_PORT \
        --host 0.0.0.0 \
        --tensor-parallel-size $NUM_GPUS \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        $EAGER_FLAG \
        $CUSTOM_AR_FLAG \
        > $LOG_FILE 2>&1 &

    PID=$!
    echo $PID > "logs/vllm_tensor.pid"
    echo -e "${GREEN}Server started with PID: ${PID}${NC}"
    echo -e "Log file: ${LOG_FILE}\n"

    echo -e "\nWaiting for server to initialize..."
    sleep 15

    # Check server health
    echo -e "\n${BLUE}Checking server health...${NC}"
    if curl -s http://localhost:${BASE_PORT}/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Tensor parallel server (port ${BASE_PORT}): HEALTHY"
    else
        echo -e "${YELLOW}⚠${NC} Tensor parallel server (port ${BASE_PORT}): Still initializing..."
    fi

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Tensor Parallel Server Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\nUsage in Python:"
    echo -e '  client = VllmAPI('
    echo -e '      model_name="'$MODEL'",'
    echo -e '      base_url="http://localhost:'$BASE_PORT'/v1"'
    echo -e '  )'
    echo -e "\nNote: Tensor parallelism provides lower throughput but enables larger models."
    echo -e "For maximum throughput with smaller models, use data parallel mode.\n"

else
    # ============================================
    # DATA PARALLEL MODE
    # ============================================
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Data Parallel Cluster${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${GPU_IDS[@]}"
    echo -e "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
    echo -e "${BLUE}================================================${NC}\n"

    # Start servers
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        LOG_FILE="logs/vllm_gpu${i}_port${PORT}.log"

        echo -e "${GREEN}[GPU $GPU_ID]${NC} Starting vLLM server on port ${PORT}..."

        CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
            --model $MODEL \
            --port $PORT \
            --host 0.0.0.0 \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --disable-log-requests \
            --max-model-len $MAX_MODEL_LEN \
            --max-num-seqs $MAX_NUM_SEQS \
            > $LOG_FILE 2>&1 &

        PID=$!
        echo $PID > "logs/vllm_gpu${i}.pid"
        echo -e "${GREEN}[GPU $GPU_ID]${NC} Server started with PID: ${PID}"
        echo -e "        Log file: ${LOG_FILE}\n"

        # Give each server time to initialize
        sleep 3
    done

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}All vLLM servers started successfully!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "URLs:"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        echo -e "  GPU $GPU_ID: http://localhost:${PORT}/v1"
    done

    echo -e "\nWaiting for servers to fully initialize..."
    sleep 10

    # Check server health
    echo -e "\n${BLUE}Checking server health...${NC}"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        GPU_ID=${GPU_IDS[$i]}
        PORT=$((BASE_PORT + i))
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} GPU $GPU_ID (port ${PORT}): HEALTHY"
        else
            echo -e "${YELLOW}⚠${NC} GPU $GPU_ID (port ${PORT}): Still initializing..."
        fi
    done

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Data Parallel Cluster Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    # Build comma-separated base_url list across all started servers
    BASE_URLS=""
    for i in $(seq 0 $((NUM_GPUS-1))); do
        PORT=$((BASE_PORT + i))
        BASE_URLS+="http://localhost:${PORT}/v1"
        if [[ $i -lt $((NUM_GPUS-1)) ]]; then BASE_URLS+=","; fi
    done
    echo -e "\nUsage in Python:"
    echo -e '  client = VllmAPI('
    echo -e '      model_name="'$MODEL'",'
    echo -e '      base_url="'$BASE_URLS'"'
    echo -e '  )'
    echo -e "\nTo test: python test_vllm_cluster.py"
fi

echo -e "\nTo stop: ./stop_vllm_cluster.sh"
echo -e "View logs: tail -f logs/vllm*.log\n"
