#!/bin/bash
# Run the ECtHR LATTICE evaluation for ONE model on a chosen backend.
#
# Wraps src/llm_rl_playground/test_ecthr_models.py so you only specify the
# model and backend; everything else falls back to sensible defaults.
#
# Usage:
#   ./run_ecthr_test.sh --model NAME [OPTIONS]
#
# Options:
#   --model NAME        HF model id, or vLLM-served model/LoRA name   (required)
#   --backend B         local | vllm | openai | genai                 (default: vllm)
#   --lora PATH         LoRA/adapter dir. Only used by 'local' backend.
#                       (vLLM serves LoRA server-side — see note below.)
#   --base-url URL      vLLM endpoint(s), comma-separated for cluster
#                                          (default: http://localhost:8000/v1)
#   --label NAME        Run label / output prefix (default: derived from model)
#   --n-cases N         Number of ECtHR cases to evaluate             (default: 100)
#   --num-iters N       LATTICE traversal steps                       (default: 10)
#   -h, --help          Show this help and exit
#
# Any extra args after a literal '--' are passed straight through to
# test_ecthr_models.py, e.g.:
#   ./run_ecthr_test.sh --model Qwen/Qwen3.6-27B-FP8 -- --max-beam-size 4 --start 50
#
# Backends:
#   local  - load model (and optional --lora) in-process via HuggingFace
#   vllm   - call a running vLLM server at --base-url
#   openai - OpenAI Responses API (needs OPENAI_API_KEY)
#   genai  - Google Gemini API (needs GOOGLE_API_KEY)
#
# Note on vLLM + LoRA:
#   vLLM applies LoRA at the server level, not per-request from the eval.
#   To test a LoRA on vLLM, start the server with:
#     --enable-lora --lora-modules <name>=/path/to/adapter
#   then run this script with: --backend vllm --model <name>

set -e

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

print_usage() { sed -n '2,38p' "$0" | sed 's/^#\s\?//'; }

# ---- Defaults ----
MODEL=""
BACKEND="vllm"
LORA=""
BASE_URL="http://localhost:8000/v1"
LABEL=""
N_CASES=100
NUM_ITERS=10
PASSTHROUGH=()

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2 ;;
        --backend)   BACKEND="$2"; shift 2 ;;
        --lora)      LORA="$2"; shift 2 ;;
        --base-url)  BASE_URL="$2"; shift 2 ;;
        --label)     LABEL="$2"; shift 2 ;;
        --n-cases)   N_CASES="$2"; shift 2 ;;
        --num-iters) NUM_ITERS="$2"; shift 2 ;;
        -h|--help)   print_usage; exit 0 ;;
        --)          shift; PASSTHROUGH=("$@"); break ;;
        *)
            echo -e "${RED}Error: unknown option '$1'${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1 ;;
    esac
done

# ---- Validate ----
if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required.${NC}" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 1
fi

case "$BACKEND" in
    local|localModel|vllm|openai|genai) ;;
    *)
        echo -e "${RED}Error: --backend must be local, vllm, openai, or genai (got '$BACKEND').${NC}" >&2
        exit 1 ;;
esac

# ---- Derive label if not given ----
# Falls back to LATTICE_VLLM_LABEL (set by start_vllm_cluster.sh's env file) if
# present, otherwise a slug of the model name.
if [[ -z "$LABEL" ]]; then
    if [[ -n "$LATTICE_VLLM_LABEL" ]]; then
        LABEL="$LATTICE_VLLM_LABEL"
    else
        SLUG=$(echo "$MODEL" | sed 's#[^A-Za-z0-9._-]#-#g')
        if [[ -n "$LORA" ]]; then LABEL="${SLUG}-lora"; else LABEL="${SLUG}-base"; fi
    fi
fi

# ---- Build the --run spec string ----
RUN_SPEC="label=${LABEL},backend=${BACKEND},model=${MODEL}"

# LoRA handling differs by backend
if [[ -n "$LORA" ]]; then
    case "$BACKEND" in
        local|localModel)
            RUN_SPEC="${RUN_SPEC},adapter=${LORA}" ;;
        vllm)
            echo -e "${YELLOW}Warning: vLLM applies LoRA at the server, not from the eval.${NC}" >&2
            echo -e "${YELLOW}  --lora '${LORA}' will be IGNORED for the vllm backend.${NC}" >&2
            echo -e "${YELLOW}  Start vLLM with: --enable-lora --lora-modules <name>=${LORA}${NC}" >&2
            echo -e "${YELLOW}  then re-run with: --backend vllm --model <name>${NC}" >&2 ;;
        *)
            echo -e "${YELLOW}Warning: --lora is only supported for the 'local' backend; ignoring.${NC}" >&2 ;;
    esac
fi

# vLLM needs the endpoint URL(s)
if [[ "$BACKEND" == "vllm" ]]; then
    RUN_SPEC="${RUN_SPEC},base_url=${BASE_URL}"
fi

# ---- Locate repo + script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="${REPO_ROOT}/src/llm_rl_playground/test_ecthr_models.py"

echo -e "${GREEN}Running ECtHR eval${NC}"
echo -e "  Model:    ${MODEL}"
echo -e "  Backend:  ${BACKEND}"
echo -e "  LoRA:     ${LORA:-<none>}"
[[ "$BACKEND" == "vllm" ]] && echo -e "  Base URL: ${BASE_URL}"
echo -e "  Cases:    ${N_CASES}   Iters: ${NUM_ITERS}"
echo -e "  Run spec: ${RUN_SPEC}\n"

# ---- Execute ----
PYTHONPATH="${REPO_ROOT}/src" python "$TEST_SCRIPT" \
    --run "$RUN_SPEC" \
    --n-cases "$N_CASES" \
    --num-iters "$NUM_ITERS" \
    "${PASSTHROUGH[@]}"
