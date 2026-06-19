#!/bin/bash
# Script to start vLLM servers in data parallel or tensor parallel mode
#
# Usage:
#   ./start_vllm_cluster.sh [MODEL] [MODE] [GPU_IDS] [NUM_GPUS] [ENFORCE_EAGER] [DISABLE_CUSTOM_AR]
#
# Examples:
#   ./start_vllm_cluster.sh                                                # Data parallel with default model
#   ./start_vllm_cluster.sh "Qwen/Qwen3-VL-8B-Instruct" data              # Data parallel
#   ./start_vllm_cluster.sh "meta-llama/Llama-2-70b-hf" tensor            # Tensor parallel
#   ./start_vllm_cluster.sh "Qwen/Qwen3.6-27B-FP8" tensor 0,1 2          # Tensor parallel, GPUs 0 and 1
#   ./start_vllm_cluster.sh "Qwen/Qwen3.6-27B-FP8" tensor 0,1 2 true     # Same, force eager mode
#   ./start_vllm_cluster.sh "Qwen/Qwen3.6-27B-FP8" tensor 0,1 2 auto false # Same, keep custom all-reduce
#
# ENFORCE_EAGER (5th arg, default "auto"):
#   auto  - always enabled in tensor mode (recommended for large models)
#   true  - always skip CUDA graph compilation
#   false - allow CUDA graph compilation (may hang if compilation takes >60s)
#
# DISABLE_CUSTOM_AR (6th arg, default "auto"):
#   auto  - always enabled in tensor mode (recommended; avoids Blackwell hangs)
#   true  - always disable custom all-reduce (use NCCL all-reduce)
#   false - keep vLLM's custom all-reduce (may hang on some GPUs/topologies)
#
# Modes:
#   data   - Run multiple servers (one per GPU) for maximum throughput
#   tensor - Run single server with model split across GPUs for large models

set -e

# Configuration
MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MODE="${2:-data}"  # "data" or "tensor"
BASE_PORT=8000
GPU_MEM_UTIL=0.90
# Specify GPU IDs to use (comma-separated, e.g. "0,1,3,4")
GPU_IDS_RAW="${3:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
NUM_GPUS="${4:-${#GPU_IDS[@]}}"

if [[ "$NUM_GPUS" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Warning: NUM_GPUS ($NUM_GPUS) doesn't match number of GPU_IDS passed (${#GPU_IDS[@]}): ${GPU_IDS[*]}" >&2
fi





MAX_MODEL_LEN=16384
# Whether to skip CUDA graph compilation (avoids EngineCore IPC timeout on large models)
ENFORCE_EAGER="${5:-auto}"
# Whether to disable vLLM's custom all-reduce kernel (falls back to NCCL).
# Custom all-reduce can deadlock on some GPUs/topologies (e.g. Blackwell sm_100),
# causing tensor parallel to hang silently after weight loading.
DISABLE_CUSTOM_AR="${6:-auto}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Validate mode
if [[ "$MODE" != "data" && "$MODE" != "tensor" ]]; then
    echo -e "${RED}Error: MODE must be 'data' or 'tensor', got: $MODE${NC}"
    echo "Usage: $0 [MODEL] [MODE]"
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
    echo -e "\nUsage in Python:"
    echo -e '  client = VllmAPI('
    echo -e '      model_name="'$MODEL'",'
    echo -e '      base_url="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1"'
    echo -e '  )'
    echo -e "\nTo test: python test_vllm_cluster.py"
fi

echo -e "\nTo stop: ./stop_vllm_cluster.sh"
echo -e "View logs: tail -f logs/vllm*.log\n"
