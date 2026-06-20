from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
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
    backend: str = "localModel"
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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_run_spec(raw: str, default_backend: str = "localModel") -> ModelRunSpec:
    """Parse one --run entry.

    Supported formats:
      --run '{"label": "qwen", "model_id": "Qwen/...", "adapter_path": "..."}'
      --run label=qwen,backend=localModel,model=Qwen/...,adapter=src/...
    """
    raw = raw.strip()
    if not raw:
        raise argparse.ArgumentTypeError("--run cannot be empty")

    if raw.startswith("{"):
        payload = json.loads(raw)
    else:
        payload: dict[str, str] = {}
        for part in raw.split(","):
            if "=" not in part:
                raise argparse.ArgumentTypeError(
                    f"Invalid --run segment {part!r}. Use comma-separated key=value pairs."
                )
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip()

    label = payload.get("label") or payload.get("name")
    model_id = payload.get("model_id") or payload.get("model")
    backend = payload.get("backend", default_backend)
    adapter_path = payload.get("adapter_path") or payload.get("adapter")
    base_url = payload.get("base_url") or payload.get("vllm_base_url")

    if not model_id:
        raise argparse.ArgumentTypeError(f"--run is missing model/model_id: {raw}")
    if not label:
        label = slugify(model_id)
        if none_if_blank(adapter_path):
            label = f"{label}-adapter"

    return ModelRunSpec(
        label=slugify(label),
        model_id=model_id,
        backend=normalize_backend(backend),
        adapter_path=none_if_blank(adapter_path),
        base_url=none_if_blank(base_url),
    )


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
            f"Unknown backend {backend!r}. Choose localModel, vllm, openai, or genai."
        )
    return normalized


def load_run_specs_from_file(path: Path, default_backend: str) -> list[ModelRunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--runs-file must contain a JSON list")
    return [parse_run_spec(json.dumps(item), default_backend=default_backend) for item in payload]


def infer_adapter_base_model(adapter_dir: Path) -> str | None:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get("base_model_name_or_path")


def discover_local_adapter_runs(outputs_dir: Path) -> list[ModelRunSpec]:
    runs = []
    if not outputs_dir.exists():
        return runs

    for adapter_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        has_config = (adapter_dir / "adapter_config.json").exists()
        has_weights = (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()
        if not (has_config and has_weights):
            continue
        model_id = infer_adapter_base_model(adapter_dir)
        if not model_id:
            continue
        runs.append(
            ModelRunSpec(
                label=slugify(adapter_dir.name),
                model_id=model_id,
                backend="localModel",
                adapter_path=str(adapter_dir),
            )
        )
    return runs


def add_base_model_runs(runs: list[ModelRunSpec]) -> list[ModelRunSpec]:
    expanded = list(runs)
    seen = {(run.backend, run.model_id, none_if_blank(run.adapter_path), none_if_blank(run.base_url)) for run in expanded}

    for run in runs:
        if run.backend != "localModel" or not none_if_blank(run.adapter_path):
            continue
        key = (run.backend, run.model_id, None, none_if_blank(run.base_url))
        if key in seen:
            continue
        expanded.append(
            ModelRunSpec(
                label=slugify(f"{run.label}-base"),
                model_id=run.model_id,
                backend=run.backend,
                adapter_path=None,
                base_url=run.base_url,
            )
        )
        seen.add(key)
    return expanded


def load_eval_semantic_tree(eval_tree_path: Path):
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


def make_eval_api(
    args: argparse.Namespace,
    run: ModelRunSpec,
    adapter_path: Path | None,
    logger,
):
    from llm_apis import GenAIAPI, LocalModelAPI, OpenAIResponsesAPI, VllmAPI

    if run.backend == "genai":
        return GenAIAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    if run.backend == "openai":
        return OpenAIResponsesAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    if run.backend == "vllm":
        return VllmAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
            base_url=run.base_url or args.vllm_base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            load_balance=True,
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


def save_results(
    output_dir: Path,
    run: ModelRunSpec,
    df,
    results: list[dict[str, Any]],
) -> None:
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
        raise FileNotFoundError(f"Adapter path for run {run.label!r} does not exist: {adapter_path}")

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
        return summary, None
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
        description=(
            "Run ECtHR LATTICE evaluation for multiple local, adapter, vLLM, OpenAI, "
            "or Gemini model configurations and save one comparison table."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Model run config. Repeat for multiple runs. Format: "
            "label=name,backend=localModel,model=Qwen/...,adapter=src/... "
            "or a JSON object with label/model_id/backend/adapter_path/base_url."
        ),
    )
    parser.add_argument("--runs-file", default=None, help="JSON list of run configs.")
    parser.add_argument("--default-backend", default="localModel", type=normalize_backend)
    parser.add_argument(
        "--auto-discover-adapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When no --run is passed, evaluate adapter dirs under --adapter-outputs-dir. "
            "Off by default: with no adapter specified, no adapter is used."
        ),
    )
    parser.add_argument(
        "--include-base-models",
        action="store_true",
        help="For local adapter runs, also add the same base model without the adapter.",
    )
    parser.add_argument(
        "--adapter-outputs-dir",
        default="src/llm_rl_playground/outputs",
        help="Directory scanned by --auto-discover-adapters.",
    )

    parser.add_argument("--tree-path", default="trees/EU/eu_conventions_notebook/eu_conventions_tree-bottom-up-llm.pkl")
    parser.add_argument("--ecthr-dataset", default="AUEB-NLP/ecthr_cases")
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--output-dir", default="src/llm_rl_playground/outputs/ecthr-model-comparison")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--top-k-leaves", type=int, default=10)
    parser.add_argument("--prediction-min-score", type=float, default=0.4)
    parser.add_argument("--max-predicted-articles", type=int, default=None)
    parser.add_argument("--use-llm-selector", action="store_true")

    parser.add_argument("--max-concurrent-calls", type=int, default=None)
    parser.add_argument("--parse-max-concurrent-calls", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--staggering-delay", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--vllm-base-url", default=None)

    parser.add_argument("--local-use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-serialize-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-api-calls", action="store_true")

    parser.add_argument("--max-beam-size", type=int, default=8)
    parser.add_argument("--relevance-chain-factor", type=float, default=0.5)
    parser.add_argument("--reasoning-in-traversal-prompt", type=int, default=-1)
    parser.add_argument("--max-prompt-proto-size", type=int, default=None)
    parser.add_argument("--max-doc-desc-char-len", type=int, default=None)

    parser.add_argument("--detailed-logs", action="store_true")
    parser.add_argument("--quiet-cases", action="store_true")
    parser.add_argument("--show-error-logs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def collect_run_specs(args: argparse.Namespace, repo_root: Path) -> list[ModelRunSpec]:
    runs = [parse_run_spec(raw, default_backend=args.default_backend) for raw in args.run]

    if args.runs_file:
        runs.extend(
            load_run_specs_from_file(
                resolve_path(args.runs_file, repo_root) or Path(args.runs_file),
                default_backend=args.default_backend,
            )
        )

    if not runs and args.auto_discover_adapters:
        adapter_outputs_dir = resolve_path(args.adapter_outputs_dir, repo_root)
        assert adapter_outputs_dir is not None
        runs.extend(discover_local_adapter_runs(adapter_outputs_dir))

    if args.include_base_models:
        runs = add_base_model_runs(runs)

    if not runs:
        raise SystemExit(
            "No model runs configured. Pass one or more --run entries, a --runs-file, "
            "or keep --auto-discover-adapters enabled with adapters in --adapter-outputs-dir."
        )

    return runs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    ensure_src_on_path(repo_root)

    import pandas as pd

    from llm_rl_playground.ecthr_evaluation import get_label_names, load_ecthr_dataset
    from utils import compute_node_registry

    output_dir = resolve_path(args.output_dir, repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = resolve_path(args.tree_path, repo_root)
    if tree_path is None or not tree_path.exists():
        raise FileNotFoundError(f"Evaluation tree does not exist: {tree_path}")

    runs = collect_run_specs(args, repo_root)
    print(
        json.dumps(
            {
                "tree_path": str(tree_path),
                "output_dir": str(output_dir),
                "runs": [asdict(run) for run in runs],
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

    summaries = []
    failures = []
    for run_index, run in enumerate(runs, start=1):
        print(f"\n================ Run {run_index}/{len(runs)}: {run.label} ================")
        print(json.dumps(asdict(run), ensure_ascii=False, indent=2))
        try:
            summary, failure = run_one_model(
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
            if summary is not None:
                summaries.append(summary)
            if failure is not None:
                failures.append(failure)
        except Exception as exc:
            failure = {
                "run": run.label,
                "backend": run.backend,
                "model_id": run.model_id,
                "adapter_path": run.adapter_path or "",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"Run failed: {run.label}: {exc}")
            if args.show_error_logs:
                print(failure["traceback"])
            if args.fail_fast:
                raise

    if summaries:
        comparison_df = pd.concat(summaries, ignore_index=True)
        comparison_csv = output_dir / "ecthr_model_comparison.csv"
        comparison_json = output_dir / "ecthr_model_comparison.json"
        comparison_df.to_csv(comparison_csv, index=False)
        comparison_df.to_json(comparison_json, orient="records", indent=2)
        print("\n================ ECtHR Model Comparison ================")
        print(comparison_df.to_string(index=False))
        print(f"Saved comparison CSV to: {comparison_csv}")
        print(f"Saved comparison JSON to: {comparison_json}")

    if failures:
        failures_path = output_dir / "ecthr_model_comparison_failures.json"
        failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved failures to: {failures_path}")
        if not summaries:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
