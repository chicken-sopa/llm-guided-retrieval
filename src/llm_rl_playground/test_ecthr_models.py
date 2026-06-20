from __future__ import annotations

"""Run the ECtHR LATTICE evaluation for a SINGLE model on a chosen backend.

Pick the LLM type with --backend (local | vllm | openai | genai), the model with
--model, and tune the run with the eval/api/traversal flags (see --help).

------------------------------------------------------------------------------
Setup: start a vLLM cluster and load the env vars it writes
------------------------------------------------------------------------------
The start script writes logs/vllm_load_balanced_env.sh exporting:
    VLLM_BASE_URL        all server URLs (comma-separated)
    LATTICE_VLLM_MODEL   the served model id
    LATTICE_VLLM_LABEL   a label for the run / output files

    ./scripts/start_vllm_cluster.sh --model "Qwen/Qwen3.6-27B-FP8" --vllm-mode data
    source logs/vllm_load_balanced_env.sh

--base-url falls back to $VLLM_BASE_URL and --label to $LATTICE_VLLM_LABEL, so
once the env file is sourced you only need --backend and --model.

------------------------------------------------------------------------------
Examples (all assume `source logs/vllm_load_balanced_env.sh` was run first)
------------------------------------------------------------------------------
  # vLLM base model — base_url + label come from the env vars
  python src/llm_rl_playground/test_ecthr_models.py \
      --backend vllm --model "$LATTICE_VLLM_MODEL" --base-url "$VLLM_BASE_URL" \
      --n-cases 100 --num-iters 10

  # vLLM, minimal — base_url ($VLLM_BASE_URL) and label ($LATTICE_VLLM_LABEL)
  # are picked up automatically when omitted
  python src/llm_rl_playground/test_ecthr_models.py \
      --backend vllm --model "$LATTICE_VLLM_MODEL"

  # vLLM with an explicit run label (overrides $LATTICE_VLLM_LABEL)
  python src/llm_rl_playground/test_ecthr_models.py \
      --backend vllm --model "$LATTICE_VLLM_MODEL" --base-url "$VLLM_BASE_URL" \
      --label "${LATTICE_VLLM_LABEL}-run2"

  # Local HuggingFace model with a LoRA adapter (no server needed)
  python src/llm_rl_playground/test_ecthr_models.py \
      --backend local --model "$LATTICE_VLLM_MODEL" \
      --lora src/llm_rl_playground/outputs/qwen2.5-1.5b-ecthr-tree-traversal-lora \
      --label "$LATTICE_VLLM_LABEL"

  # OpenAI / Gemini (set OPENAI_API_KEY / GOOGLE_API_KEY first)
  python src/llm_rl_playground/test_ecthr_models.py \
      --backend openai --model gpt-4.1 --label "$LATTICE_VLLM_LABEL"
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelRunSpec:
    label: str
    model_id: str
    backend: str = "vllm"
    adapter_path: str | None = None
    base_url: str | None = None


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "run"


def none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null", "false", "-"}:
        return None
    return value


def resolve_path(path: str | None, repo_root: Path) -> Path | None:
    path = none_if_blank(path)
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def normalize_backend(backend: str) -> str:
    aliases = {
        "local": "localModel",
        "localmodel": "localModel",
        "localModel": "localModel",
        "vllm": "vllm",
        "openai": "openai",
        "genai": "genai",
    }
    key = backend.strip()
    normalized = aliases.get(key, aliases.get(key.lower()))
    if normalized is None:
        raise argparse.ArgumentTypeError(
            f"Unknown backend {backend!r}. Choose local, vllm, openai, or genai."
        )
    return normalized


def build_run_spec(args: argparse.Namespace) -> ModelRunSpec:
    backend = normalize_backend(args.backend)
    adapter_path = none_if_blank(args.lora)
    base_url = none_if_blank(args.base_url)

    if backend == "vllm" and base_url is None:
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    if adapter_path and backend != "localModel":
        print(
            f"Warning: --lora is only applied for the local backend; "
            f"ignoring adapter for backend '{backend}'.",
            file=sys.stderr,
        )
        adapter_path = None

    label = none_if_blank(args.label) or os.getenv("LATTICE_VLLM_LABEL")
    if not label:
        label = slugify(args.model)
        label = f"{label}-lora" if adapter_path else f"{label}-base"

    return ModelRunSpec(
        label=slugify(label),
        model_id=args.model,
        backend=backend,
        adapter_path=adapter_path,
        base_url=base_url,
    )


def load_eval_semantic_tree(eval_tree_path: Path):
    import pickle

    from tree_objects import SemanticNode

    if eval_tree_path.suffix == ".pkl":
        tree_obj = pickle.loads(eval_tree_path.read_bytes())
    else:
        tree_obj = json.loads(eval_tree_path.read_text(encoding="utf-8"))
    return SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj


def make_eval_hyperparams(args: argparse.Namespace, tree_path: Path, run: ModelRunSpec):
    from hyperparams import HyperParams

    hp = HyperParams.from_args("--subset fiqa --tree_version eu_conventions_notebook")
    hp.TREE_PATH = str(tree_path)
    hp.DATASET = "EU"
    hp.LLM_API_BACKEND = run.backend
    hp.LLM = run.model_id
    hp.LLM_API_TIMEOUT = args.timeout
    hp.LLM_API_MAX_RETRIES = args.max_retries
    hp.LLM_MAX_CONCURRENT_CALLS = args.max_concurrent_calls or (1 if run.backend == "localModel" else 8)
    hp.LLM_API_STAGGERING_DELAY = args.staggering_delay
    hp.REASONING_IN_TRAVERSAL_PROMPT = -1
    hp.SUBSET = "fiqa"
    hp.MAX_BEAM_SIZE = args.max_beam_size
    hp.SEARCH_WITH_PATH_RELEVANCE = True
    hp.NUM_LEAF_CALIB = 0
    hp.RELEVANCE_CHAIN_FACTOR = args.relevance_chain_factor
    hp.MAX_PROMPT_PROTO_SIZE = args.max_prompt_proto_size
    hp.MAX_DOC_DESC_CHAR_LEN = args.max_doc_desc_char_len
    hp.SHOW_ERROR_LOGS = args.show_error_logs
    return hp


def make_llm_api_kwargs(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    from prompts import get_traversal_prompt_response_constraint

    kwargs: dict[str, Any] = {
        "max_concurrent_calls": args.max_concurrent_calls or (1 if backend == "localModel" else 8),
        "response_mime_type": "application/json",
        "response_schema": get_traversal_prompt_response_constraint(
            bool(args.reasoning_in_traversal_prompt)
        ),
        "staggering_delay": args.staggering_delay,
        "print_summary_report": False,
        "show_error_logs": args.show_error_logs,
        "parse_max_concurrent_calls": args.parse_max_concurrent_calls
        or args.max_concurrent_calls
        or (1 if backend == "localModel" else 8),
    }

    if backend == "vllm":
        kwargs.pop("response_mime_type", None)
        kwargs.pop("response_schema", None)
        kwargs["max_tokens"] = args.max_tokens
    elif backend == "openai":
        kwargs.pop("response_mime_type", None)
        kwargs["max_output_tokens"] = args.max_tokens
    elif backend == "genai":
        kwargs["max_output_tokens"] = args.max_tokens
    elif backend == "localModel":
        kwargs.pop("response_mime_type", None)
        kwargs["max_new_tokens"] = args.max_tokens

    return kwargs


def make_eval_api(args: argparse.Namespace, run: ModelRunSpec, adapter_path: Path | None, logger):
    from llm_apis import GenAIAPI, LocalModelAPI, OpenAIResponsesAPI, VllmAPI

    if run.backend == "genai":
        return GenAIAPI(run.model_id, logger=logger, timeout=args.timeout, max_retries=args.max_retries)
    if run.backend == "openai":
        return OpenAIResponsesAPI(run.model_id, logger=logger, timeout=args.timeout, max_retries=args.max_retries)
    if run.backend == "vllm":
        return VllmAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
            base_url=run.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            load_balance=True,
            enable_thinking=args.enable_thinking,
        )
    if run.backend == "localModel":
        return LocalModelAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
            adapter_path=None if adapter_path is None else str(adapter_path),
            use_4bit=args.local_use_4bit,
            enable_thinking=args.enable_thinking,
            serialize_requests=args.local_serialize_requests,
            log_api_calls=args.log_api_calls,
        )
    raise ValueError(f"Unknown backend: {run.backend}")


def unload_eval_api(api: Any) -> None:
    unload = getattr(api, "unload", None)
    if callable(unload):
        unload()
        return
    runtime = getattr(api, "runtime", None)
    if runtime is not None and hasattr(runtime, "unload"):
        runtime.unload()


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "sample"}


def save_results(output_dir: Path, run: ModelRunSpec, df, results: list[dict[str, Any]]) -> None:
    run_slug = slugify(run.label)
    df.to_json(output_dir / f"{run_slug}_ecthr_eval_rows.json", orient="records", indent=2)
    with (output_dir / f"{run_slug}_ecthr_eval_results.json").open("w", encoding="utf-8") as f:
        json.dump([serializable_result(result) for result in results], f, ensure_ascii=False, indent=2)


def run_one_model(
    args: argparse.Namespace,
    run: ModelRunSpec,
    *,
    repo_root: Path,
    tree_path: Path,
    semantic_root_node: Any,
    node_registry: list[Any],
    eval_dataset: Any,
    label_names: list[str] | None,
    output_dir: Path,
):
    from llm_rl_playground.ecthr_evaluation import EcthrTraversalEvaluator, summarize_ecthr_cases
    from utils import setup_logger

    adapter_path = resolve_path(run.adapter_path, repo_root)
    if run.backend == "localModel" and adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    log_path = output_dir / f"{slugify(run.label)}.log"
    logger = setup_logger(f"test_ecthr_models.{slugify(run.label)}", str(log_path))
    hp = make_eval_hyperparams(args, tree_path, run)
    llm_api_kwargs = make_llm_api_kwargs(args, run.backend)

    api = make_eval_api(args, run, adapter_path, logger)
    evaluator = EcthrTraversalEvaluator(
        semantic_root_node=semantic_root_node,
        node_registry=node_registry,
        hp=hp,
        logger=logger,
        llm_api=api,
        llm_api_kwargs=llm_api_kwargs,
        show_error_logs=args.show_error_logs,
    )

    start_time = time.time()
    try:
        df, results = evaluator.evaluate_ecthr_cases_batched(
            eval_dataset,
            label_names,
            n_cases=args.n_cases,
            num_iters=args.num_iters,
            top_k=args.top_k_leaves,
            start=args.start,
            detailed_logs=args.detailed_logs,
            prediction_min_score=args.prediction_min_score,
            max_predicted_articles=args.max_predicted_articles,
            use_llm_selector=args.use_llm_selector,
            print_cases=not args.quiet_cases,
            print_summary=True,
        )
        elapsed = time.time() - start_time
        save_results(output_dir, run, df, results)

        summary = summarize_ecthr_cases(df)
        summary.insert(0, "elapsed_seconds", round(elapsed, 2))
        summary.insert(0, "adapter_path", "" if adapter_path is None else str(adapter_path))
        summary.insert(0, "model_id", run.model_id)
        summary.insert(0, "backend", run.backend)
        summary.insert(0, "run", run.label)
        return summary
    finally:
        unload_eval_api(api)
        del evaluator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ECtHR LATTICE evaluation for a single model on a chosen backend."
    )

    # --- Model / backend selection ---
    parser.add_argument("--backend", default="vllm", help="local | vllm | openai | genai (default: vllm)")
    parser.add_argument("--model", required=True, help="HF model id, or vLLM-served model/adapter name.")
    parser.add_argument("--lora", "--adapter", dest="lora", default=None,
                        help="LoRA/adapter dir. Only applied for the local backend.")
    parser.add_argument("--base-url", default=None,
                        help="vLLM endpoint(s), comma-separated. Defaults to $VLLM_BASE_URL or localhost:8000.")
    parser.add_argument("--label", default=None,
                        help="Run label / output prefix. Defaults to $LATTICE_VLLM_LABEL or a slug of --model.")

    # --- Dataset / tree / output ---
    parser.add_argument("--tree-path", default="trees/EU/eu_conventions_notebook/eu_conventions_tree-bottom-up-llm.pkl")
    parser.add_argument("--ecthr-dataset", default="AUEB-NLP/ecthr_cases")
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--output-dir", default="src/llm_rl_playground/outputs/ecthr-eval")

    # --- Eval controls ---
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--top-k-leaves", type=int, default=10)
    parser.add_argument("--prediction-min-score", type=float, default=0.4)
    parser.add_argument("--max-predicted-articles", type=int, default=None)
    parser.add_argument("--use-llm-selector", action="store_true")

    # --- API controls ---
    parser.add_argument("--max-concurrent-calls", type=int, default=None)
    parser.add_argument("--parse-max-concurrent-calls", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--staggering-delay", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=384)

    # --- Local backend controls ---
    parser.add_argument("--local-use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-serialize-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-api-calls", action="store_true")

    # --- Traversal controls ---
    parser.add_argument("--max-beam-size", type=int, default=8)
    parser.add_argument("--relevance-chain-factor", type=float, default=0.5)
    parser.add_argument("--reasoning-in-traversal-prompt", type=int, default=-1)
    parser.add_argument("--max-prompt-proto-size", type=int, default=None)
    parser.add_argument("--max-doc-desc-char-len", type=int, default=None)

    # --- Logging ---
    parser.add_argument("--detailed-logs", action="store_true")
    parser.add_argument("--quiet-cases", action="store_true")
    parser.add_argument("--show-error-logs", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    ensure_src_on_path(repo_root)

    from llm_rl_playground.ecthr_evaluation import get_label_names, load_ecthr_dataset
    from utils import compute_node_registry

    output_dir = resolve_path(args.output_dir, repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = resolve_path(args.tree_path, repo_root)
    if tree_path is None or not tree_path.exists():
        raise FileNotFoundError(f"Evaluation tree does not exist: {tree_path}")

    run = build_run_spec(args)
    print(
        json.dumps(
            {
                "tree_path": str(tree_path),
                "output_dir": str(output_dir),
                "run": asdict(run),
                "eval_split": args.eval_split,
                "start": args.start,
                "n_cases": args.n_cases,
                "num_iters": args.num_iters,
                "top_k_leaves": args.top_k_leaves,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    semantic_root_node = load_eval_semantic_tree(tree_path)
    node_registry = compute_node_registry(semantic_root_node)
    eval_dataset = load_ecthr_dataset(split=args.eval_split, config=args.ecthr_config)
    label_names = get_label_names(eval_dataset)

    try:
        summary = run_one_model(
            args,
            run,
            repo_root=repo_root,
            tree_path=tree_path,
            semantic_root_node=semantic_root_node,
            node_registry=node_registry,
            eval_dataset=eval_dataset,
            label_names=label_names,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"Run failed: {run.label}: {exc}", file=sys.stderr)
        if args.show_error_logs:
            print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)

    summary_csv = output_dir / f"{slugify(run.label)}_ecthr_summary.csv"
    summary_json = output_dir / f"{slugify(run.label)}_ecthr_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary.to_json(summary_json, orient="records", indent=2)
    print("\n================ ECtHR Eval Summary ================")
    print(summary.to_string(index=False))
    print(f"Saved summary CSV to:  {summary_csv}")
    print(f"Saved summary JSON to: {summary_json}")


if __name__ == "__main__":
    main()
