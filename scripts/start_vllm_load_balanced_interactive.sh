#!/usr/bin/env bash
# Interactive vLLM launcher for LATTICE load-balanced inference.
#
# Starts one full vLLM OpenAI-compatible server per selected GPU, using
# consecutive ports. This is data parallel serving, not tensor parallelism.
#
# If a LoRA adapter is provided, every server loads the same adapter with
# --enable-lora and --lora-modules. Use the adapter name as --llm in LATTICE.

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

default_model="Qwen/Qwen2.5-1.5B-Instruct"
default_base_port="8000"
default_max_model_len="4096"
default_gpu_mem_util="0.90"
default_dtype="half"
default_adapter_name="ecthr_adapter"
default_max_lora_rank="64"
default_think_mode = "false"  

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Interactive vLLM Load-Balanced Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo "This starts one full model replica per GPU."
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${BLUE}Current GPUs:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits || true
    echo ""
fi

read -r -p "How many GPUs do you want to use? " num_gpus
if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo -e "${RED}GPU count must be a positive integer.${NC}"
    exit 1
fi

read -r -p "GPU IDs as comma list [0..$((num_gpus - 1))]: " gpu_ids_input
if [[ -z "${gpu_ids_input// }" ]]; then
    gpu_ids=()
    for ((i = 0; i < num_gpus; i++)); do
        gpu_ids+=("$i")
    done
else
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_input"
fi

if [[ "${#gpu_ids[@]}" -ne "$num_gpus" ]]; then
    echo -e "${RED}You asked for ${num_gpus} GPUs but provided ${#gpu_ids[@]} GPU IDs.${NC}"
    exit 1
fi

read -r -p "Base model [${default_model}]: " model
model="${model:-$default_model}"

read -r -p "Base port [${default_base_port}]: " base_port
base_port="${base_port:-$default_base_port}"

read -r -p "Max model length [${default_max_model_len}]: " max_model_len
max_model_len="${max_model_len:-$default_max_model_len}"

read -r -p "GPU memory utilization [${default_gpu_mem_util}]: " gpu_mem_util
gpu_mem_util="${gpu_mem_util:-$default_gpu_mem_util}"

read -r -p "dtype [${default_dtype}]: " dtype
dtype="${dtype:-$default_dtype}"

read -r -p "LoRA adapter path, empty for base model only: " adapter_path
adapter_name=""
max_lora_rank=""
if [[ -n "${adapter_path// }" ]]; then
    if [[ ! -d "$adapter_path" ]]; then
        echo -e "${RED}Adapter path does not exist or is not a directory: ${adapter_path}${NC}"
        exit 1
    fi
    read -r -p "Adapter model name exposed by vLLM [${default_adapter_name}]: " adapter_name
    adapter_name="${adapter_name:-$default_adapter_name}"
    read -r -p "Max LoRA rank [${default_max_lora_rank}]: " max_lora_rank
    max_lora_rank="${max_lora_rank:-$default_max_lora_rank}"
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo -e "${RED}Could not find 'vllm' on PATH. Activate your vLLM environment first.${NC}"
    echo "Example: source vllm_env/bin/activate"
    exit 1
fi

base_urls=()
pid_files=()

echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}Starting vLLM servers${NC}"
echo -e "${BLUE}================================================${NC}"
echo "Model: ${model}"
echo "GPU IDs: ${gpu_ids[*]}"
echo "Ports: ${base_port}-$((base_port + num_gpus - 1))"
if [[ -n "$adapter_path" ]]; then
    echo "Adapter: ${adapter_name}=${adapter_path}"
fi
echo ""

for ((i = 0; i < num_gpus; i++)); do
    gpu_id="$(echo "${gpu_ids[$i]}" | xargs)"
    port=$((base_port + i))
    log_file="logs/vllm_lb_gpu${gpu_id}_port${port}.log"
    pid_file="logs/vllm_lb_gpu${gpu_id}_port${port}.pid"

    cmd=(
        vllm serve "$model"
        --host 0.0.0.0
        --port "$port"
        --dtype "$dtype"
        --gpu-memory-utilization "$gpu_mem_util"
        --max-model-len "$max_model_len"
        --default-chat-template-kwargs '{"enable_thinking": '$default_think_mode'}'
    )

    if [[ -n "$adapter_path" ]]; then
        cmd+=(
            --enable-lora
            --max-lora-rank "$max_lora_rank"
            --lora-modules "${adapter_name}=${adapter_path}"
        )
    fi

    echo -e "${GREEN}[GPU ${gpu_id}]${NC} Starting server on port ${port}"
    CUDA_VISIBLE_DEVICES="$gpu_id" nohup "${cmd[@]}" > "$log_file" 2>&1 &
    pid=$!
    echo "$pid" > "$pid_file"
    pid_files+=("$pid_file")
    base_urls+=("http://localhost:${port}/v1")
    echo "  PID: ${pid}"
    echo "  Log: ${log_file}"
    sleep 2
done

base_url_csv="$(IFS=,; echo "${base_urls[*]}")"
served_model="$model"
if [[ -n "$adapter_name" ]]; then
    served_model="$adapter_name"
fi

cat > logs/vllm_load_balanced_env.sh <<EOF
export VLLM_BASE_URL="${base_url_csv}"
export LATTICE_VLLM_MODEL="${served_model}"
EOF

echo -e "\n${BLUE}Waiting a few seconds before health checks...${NC}"
sleep 10

healthy=0
for ((i = 0; i < num_gpus; i++)); do
    gpu_id="$(echo "${gpu_ids[$i]}" | xargs)"
    port=$((base_port + i))
    if curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC} GPU ${gpu_id}, port ${port}"
        healthy=$((healthy + 1))
    else
        echo -e "${YELLOW}WAIT${NC} GPU ${gpu_id}, port ${port} may still be starting"
    fi
done

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}Started ${num_gpus} vLLM server(s). Healthy now: ${healthy}/${num_gpus}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Use for ECtHR adapter-vs-base LATTICE comparison:"
echo "  source logs/vllm_load_balanced_env.sh"
echo "  python src/llm_rl_playground/train_ecthr_tree_traversal_lora_multi_gpu.py \\"
echo "    --skip-training \\"
echo "    --run-ecthr-batched-eval \\"
echo "    --compare-base-model \\"
echo "    --ecthr-eval-llm-api-backend vllm \\"
echo "    --ecthr-eval-vllm-base-url \"\${VLLM_BASE_URL}\" \\"
echo "    --ecthr-eval-model-id \"\${LATTICE_VLLM_MODEL}\" \\"
echo "    --ecthr-eval-base-model-id \"${model}\" \\"
echo "    --ecthr-eval-max-concurrent-calls $((num_gpus * 4)) \\"
echo "    --ecthr-eval-parse-max-concurrent-calls $((num_gpus * 8)) \\"
echo "    --ecthr-eval-n-cases 10 \\"
echo "    --ecthr-eval-num-iters 10 \\"
echo "    --ecthr-eval-top-k-leaves 10 \\"
echo "    --ecthr-eval-use-llm-selector"
echo ""
echo "Current VLLM_BASE_URL:"
echo "  ${base_url_csv}"
echo ""
if [[ -n "$adapter_name" ]]; then
    echo "Adapter requests must use model name:"
    echo "  ${adapter_name}"
    echo ""
fi
echo "View logs:"
echo "  tail -f logs/vllm_lb_*.log"
echo ""
echo "Stop these servers:"
for pid_file in "${pid_files[@]}"; do
    echo "  kill \$(cat ${pid_file})"
done
