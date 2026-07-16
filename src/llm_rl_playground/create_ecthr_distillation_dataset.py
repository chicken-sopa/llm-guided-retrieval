from __future__ import annotations

"""Build an ECtHR LATTICE traversal distillation dataset with ANY teacher backend.

==============================================================================
QUICK START (vLLM — the default backend)
==============================================================================
A teacher model runs the LATTICE tree traversal over ECtHR cases, and every
decision it makes is saved as a supervised example to later fine-tune a smaller
"student" model. The default teacher is a vLLM-served model. Start the cluster,
source the env file it writes, then run the script — model/URL/label are read
from the env vars, and the dataset is saved batch-by-batch so a crash never
loses finished work (add ``--resume`` to continue):

    ./scripts/start_vllm_cluster.sh --model "Qwen/Qwen3.6-27B-FP8" --vllm-mode data
    source logs/vllm_load_balanced_env.sh

    python src/llm_rl_playground/create_ecthr_distillation_dataset.py \
        --model "$LATTICE_VLLM_MODEL" --n-cases 200 --batch-size 10 --num-iters 10
==============================================================================

The teacher model can be served by any of the supported backends: ``vllm`` (default),
``openai``, ``genai``, or ``local``. Pick the backend with ``--backend`` and the model
with ``--model``. (This replaced an earlier OpenAI-only distillation script.)

The dataset is produced in checkpointed BATCHES: cases are processed ``--batch-size`` at a
time and each batch's trace/training/eval rows are appended to disk immediately. If the run
crashes (OOM, API outage, Ctrl-C, ...) the completed batches are already saved, and
re-running with ``--resume`` skips the cases that finished and continues from where it
stopped. Only after every batch lands are the consolidated typed dataset + summary written.

------------------------------------------------------------------------------
Other backends
------------------------------------------------------------------------------
    # OpenAI GPT teacher (set OPENAI_API_KEY)
    python src/llm_rl_playground/create_ecthr_distillation_dataset.py \
        --backend openai --model gpt-5.5 --n-cases 100 --batch-size 10

    # Gemini teacher (set GOOGLE_API_KEY)
    python src/llm_rl_playground/create_ecthr_distillation_dataset.py \
        --backend genai --model gemini-2.5-flash --n-cases 100 --batch-size 10

    # Local HuggingFace teacher with a LoRA adapter (no server needed)
    python src/llm_rl_playground/create_ecthr_distillation_dataset.py \
        --backend local --model Qwen/Qwen2.5-1.5B-Instruct --lora path/to/adapter

    # Resume an interrupted run (same --output-dir + --label)
    python src/llm_rl_playground/create_ecthr_distillation_dataset.py \
        --model "$LATTICE_VLLM_MODEL" --n-cases 200 --batch-size 10 --resume
"""

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def repo_root_from_script() -> Path:
    """Find the repository root so default tree/output paths work from any launch directory."""
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path() -> Path:
    """Make repo-local modules importable when the script is run directly with python."""
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "run"


def none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null", "false", "-"}:
        return None
    return value


def normalize_backend(backend: str) -> str:
    """Map the many spellings of a backend name onto the canonical HyperParams value."""
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


@dataclass(slots=True)
class TeacherRunSpec:
    """Everything needed to build the teacher LLM API and name its output files."""

    label: str
    model_id: str
    backend: str = "vllm"
    adapter_path: str | None = None
    base_url: str | None = None


def build_run_spec(args: argparse.Namespace) -> TeacherRunSpec:
    """Resolve backend/model/adapter/base_url/label, honoring the vLLM env-var conventions."""
    backend = normalize_backend(args.backend)
    adapter_path = none_if_blank(args.lora)
    base_url = none_if_blank(args.base_url)

    model_id = none_if_blank(args.model) or os.getenv("LATTICE_VLLM_MODEL")
    if not model_id:
        raise SystemExit(
            "No teacher model given. Pass --model or set $LATTICE_VLLM_MODEL "
            "(e.g. by sourcing logs/vllm_load_balanced_env.sh)."
        )

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
        label = slugify(model_id)
        label = f"{label}-lora" if adapter_path else f"{label}-base"

    return TeacherRunSpec(
        label=slugify(label),
        model_id=model_id,
        backend=backend,
        adapter_path=adapter_path,
        base_url=base_url,
    )


def load_semantic_tree(tree_path: Path):
    """Load the semantic tree in the object form expected by InferSample and the evaluator."""
    from tree_objects import SemanticNode

    if tree_path.suffix == ".pkl":
        tree_obj = pickle.loads(tree_path.read_bytes())
    else:
        tree_obj = json.loads(tree_path.read_text(encoding="utf-8"))
    return SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj


def make_hyperparams(args: argparse.Namespace, tree_path: Path, run: TeacherRunSpec):
    """Configure HyperParams for an ECtHR traversal run with the chosen teacher backend."""
    from hyperparams import HyperParams

    default_concurrency = 1 if run.backend == "localModel" else 8
    hp = HyperParams.from_args("--subset fiqa --tree_version eu_conventions_notebook")
    hp.TREE_PATH = str(tree_path)
    hp.DATASET = "EU"
    hp.LLM_API_BACKEND = run.backend
    hp.LLM = run.model_id
    hp.LLM_API_TIMEOUT = args.timeout
    hp.LLM_API_MAX_RETRIES = args.max_retries
    hp.LLM_MAX_CONCURRENT_CALLS = args.max_concurrent_calls or default_concurrency
    hp.LLM_API_STAGGERING_DELAY = args.staggering_delay
    hp.REASONING_IN_TRAVERSAL_PROMPT = args.reasoning_in_traversal_prompt
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
    """Build the per-call kwargs each backend expects, matching the ECtHR eval path.

    Guided/structured JSON decoding is kept on for the backends that support it so the teacher
    labels are clean. vLLM guided decoding (``guided_json``) is opt-out via ``--no-vllm-guided-json``.
    """
    from prompts import get_traversal_prompt_response_constraint

    default_concurrency = 1 if backend == "localModel" else 8
    schema = get_traversal_prompt_response_constraint(bool(args.reasoning_in_traversal_prompt))
    kwargs: dict[str, Any] = {
        "max_concurrent_calls": args.max_concurrent_calls or default_concurrency,
        "response_mime_type": "application/json",
        "response_schema": schema,
        "staggering_delay": args.staggering_delay,
        "print_summary_report": False,
        "show_error_logs": args.show_error_logs,
        "parse_max_concurrent_calls": (
            args.parse_max_concurrent_calls or args.max_concurrent_calls or default_concurrency
        ),
    }

    if backend == "vllm":
        kwargs.pop("response_mime_type", None)
        if not args.vllm_guided_json:
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


def make_teacher_api(args: argparse.Namespace, run: TeacherRunSpec, logger):
    """Instantiate the teacher LLM API for the chosen backend."""
    from llm_apis import GenAIAPI, LocalModelAPI, OpenAIResponsesAPI, VllmAPI

    if run.backend == "genai":
        return GenAIAPI(run.model_id, logger=logger, timeout=args.timeout, max_retries=args.max_retries)
    if run.backend == "openai":
        return OpenAIResponsesAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
            log_api_calls=args.log_api_calls,
        )
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
        adapter_path = none_if_blank(run.adapter_path)
        if adapter_path is not None:
            resolved = Path(adapter_path).expanduser()
            if not resolved.is_absolute():
                resolved = repo_root_from_script() / resolved
            if not resolved.exists():
                raise FileNotFoundError(f"Adapter path does not exist: {resolved}")
            adapter_path = str(resolved.resolve())
        return LocalModelAPI(
            run.model_id,
            logger=logger,
            timeout=args.timeout,
            max_retries=args.max_retries,
            adapter_path=adapter_path,
            use_4bit=args.local_use_4bit,
            enable_thinking=args.enable_thinking,
            serialize_requests=args.local_serialize_requests,
            log_api_calls=args.log_api_calls,
        )
    raise ValueError(f"Unknown backend: {run.backend}")


def unload_teacher_api(api: Any) -> None:
    """Release the teacher runtime (matters for the in-process local backend)."""
    unload = getattr(api, "unload", None)
    if callable(unload):
        unload()
        return
    runtime = getattr(api, "runtime", None)
    if runtime is not None and hasattr(runtime, "unload"):
        runtime.unload()


async def run_lattice_with_trace_async(
    evaluator: Any,
    samples: list[Any],
    case_indexes: list[int],
    *,
    num_iters: int,
    detailed_logs: bool = False,
) -> list[dict[str, Any]]:
    """Run LATTICE externally while recording each teacher decision.

    This mirrors EcthrTraversalEvaluator.run_lattice_iterations_for_samples_async, but keeps
    the recording logic outside the normal evaluator. Each iteration gathers the current
    prompts, asks the teacher model, stores prompt/response metadata, then updates the samples.
    """
    trace_rows: list[dict[str, Any]] = []

    for step in range(num_iters):
        print(f"\n--- Teacher distillation iteration {step + 1}/{num_iters} ({len(samples)} cases) ---")
        # LATTICE exposes prompts per sample; flatten them so the teacher can process one efficient batch.
        inputs_by_sample = [sample.get_step_prompts() for sample in samples]
        counts = [len(inputs) for inputs in inputs_by_sample]
        flat_inputs = [item for inputs in inputs_by_sample for item in inputs]
        if not flat_inputs:
            print("No prompts left to process.")
            break

        flat_prompts = [prompt for prompt, _ in flat_inputs]
        flat_slates = [slate for _, slate in flat_inputs]
        # Use the same LLM wrapper/schema parser as evaluation so the teacher rows match inference behavior.
        raw_responses = await evaluator.llm_api.run_batch(flat_prompts, **evaluator.get_llm_run_kwargs())
        flat_response_jsons = await evaluator.parse_traversal_responses_async(
            raw_responses,
            prompts=flat_prompts,
            step=step,
        )

        skipped_count = sum(response_json is None for response_json in flat_response_jsons)
        if skipped_count and evaluator.show_error_logs:
            evaluator.logger.warning(
                "Skipped %s/%s traversal response(s) because they were still badly formatted after repair.",
                skipped_count,
                len(flat_response_jsons),
            )

        offset = 0
        for sample_idx, (sample, count) in enumerate(zip(samples, counts)):
            # Slice the flat batch back into the prompts that belonged to this case/sample.
            sample_prompts = flat_prompts[offset : offset + count]
            sample_slates = flat_slates[offset : offset + count]
            sample_raw_responses = raw_responses[offset : offset + count]
            sample_response_jsons = flat_response_jsons[offset : offset + count]
            case_index = case_indexes[sample_idx]

            for prompt, slate, raw_response, parsed_response in zip(
                sample_prompts,
                sample_slates,
                sample_raw_responses,
                sample_response_jsons,
            ):
                trace_rows.append(
                    {
                        "case_index": case_index,
                        "step": step,
                        "prompt": prompt,
                        "slate": slate,
                        "raw_response": raw_response,
                        "parsed_response": parsed_response,
                        "valid": parsed_response is not None,
                        "query": sample.query,
                        "teacher_model": getattr(evaluator.llm_api, "model_name", None),
                    }
                )

            if count:
                # Updating the sample advances LATTICE so the next iteration records the teacher's next decisions.
                sample.update(sample_slates, sample_response_jsons)
                if detailed_logs:
                    evaluator.print_frontier(sample)
            offset += count

    return trace_rows


def make_training_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert valid teacher trace rows into chat-style supervised fine-tuning examples."""
    rows = []
    for trace in trace_rows:
        parsed_response = trace.get("parsed_response")
        if not isinstance(parsed_response, dict):
            continue

        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an ECtHR semantic-tree traversal model. Return only valid JSON.",
                    },
                    {"role": "user", "content": trace["prompt"]},
                    {"role": "assistant", "content": json.dumps(parsed_response, ensure_ascii=False)},
                ],
                "metadata": {
                    "case_index": trace.get("case_index"),
                    "step": trace.get("step"),
                    "slate": trace.get("slate"),
                    "teacher_model": trace.get("teacher_model"),
                    "query": trace.get("query"),
                },
            }
        )
    return rows


def make_typed_traversal_dataset(training_rows: list[dict[str, Any]]):
    """Wrap chat training rows in the TraversalDataset dataclasses used by the local trainer."""
    from llm_rl_playground.ecthr_training_utils import (
        ChatMessage,
        TraversalDataset,
        TraversalDatasetStats,
        TraversalTrainingRow,
    )

    typed_rows = [
        TraversalTrainingRow(messages=[ChatMessage.from_dict(message) for message in row["messages"]])
        for row in training_rows
    ]
    valid_case_indexes = {
        row.get("metadata", {}).get("case_index")
        for row in training_rows
        if row.get("metadata", {}).get("case_index") is not None
    }
    return TraversalDataset(
        rows=typed_rows,
        stats=TraversalDatasetStats(
            cases_used=len(valid_case_indexes),
            skipped_cases=0,
            rows=len(typed_rows),
        ),
    )


def build_results_from_samples(
    *,
    selected: list[dict[str, Any]],
    queries: list[str],
    samples: list[Any],
    case_indexes: list[int],
    label_names: list[str] | None,
    top_k: int,
    prediction_min_score: float | None,
    max_predicted_articles: int | None,
) -> list[dict[str, Any]]:
    """Score the completed traced samples without rerunning LATTICE.

    The distillation loop already mutated each sample with teacher decisions, so this function
    only extracts top leaf predictions and computes the same metrics as evaluation. Returns one
    JSON-serializable eval-row dict per case (the columns summarize_ecthr_cases expects).
    """
    from llm_rl_playground.ecthr_evaluation import (
        example_gold_articles,
        predicted_articles_from_sample,
        score_prediction,
    )

    eval_rows = []
    for example, query, sample, case_idx in zip(selected, queries, samples, case_indexes):
        predicted, top_rows = predicted_articles_from_sample(
            sample,
            k=top_k,
            min_score=prediction_min_score,
            max_articles=max_predicted_articles,
        )
        gold = example_gold_articles(example, label_names)
        metrics = score_prediction(gold, predicted)
        eval_rows.append(
            {
                "case_index": case_idx,
                "gold": sorted(gold),
                "predicted": sorted(predicted),
                "lattice_predicted": sorted(predicted),
                "gold_removed_by_selector": 0,
                "gold_removed_by_selector_articles": [],
                "any_gold_found": metrics["any_gold_found"],
                "all_gold_found": metrics["all_gold_found"],
                "exact_set_match": metrics["exact_set_match"],
                "true_positive": metrics["true_positive"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "top_rows": top_rows,
                "query": query,
            }
        )

        print(f"\n================ ECtHR case {case_idx} ================")
        print("Gold:     ", eval_rows[-1]["gold"])
        print("Predicted: ", eval_rows[-1]["predicted"])
        print(
            f"Correct(all gold found) : {eval_rows[-1]['all_gold_found']} | "
            f"Recall: {eval_rows[-1]['recall']:.2f} | Precision: {eval_rows[-1]['precision']:.2f} | "
            f"F1: {eval_rows[-1]['f1']:.2f}"
        )

    return eval_rows


# --------------------------------------------------------------------------------------
# Checkpointed, batched output
# --------------------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "case_index",
    "gold",
    "predicted",
    "lattice_predicted",
    "gold_removed_by_selector",
    "gold_removed_by_selector_articles",
    "any_gold_found",
    "all_gold_found",
    "exact_set_match",
    "true_positive",
    "precision",
    "recall",
    "f1",
]


class DistillationOutputStore:
    """Append-only, crash-safe store for the distillation artifacts.

    Trace rows, chat training rows, and per-case eval rows are appended to JSONL files after
    every batch, so an interrupted run never loses completed batches. A checkpoint JSON tracks
    which case indexes are done, enabling ``--resume``. The consolidated typed dataset + summary
    JSON are (re)built from the JSONL files once every batch has landed.
    """

    def __init__(self, output_dir: Path, label: str):
        self.output_dir = output_dir
        self.label = label
        self.trace_path = output_dir / f"{label}_teacher_trace_rows.jsonl"
        self.training_jsonl_path = output_dir / f"{label}_teacher_training_rows.jsonl"
        self.eval_rows_path = output_dir / f"{label}_teacher_eval_rows.jsonl"
        self.checkpoint_path = output_dir / f"{label}_checkpoint.json"
        self.typed_dataset_path = output_dir / f"{label}_teacher_traversal_dataset.json"
        self.summary_path = output_dir / f"{label}_teacher_summary.json"
        self.checkpoint: dict[str, Any] = {}

    def start(self, *, resume: bool, config: dict[str, Any]) -> set[int]:
        """Prepare the files and return the set of already-completed case indexes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        completed: set[int] = set()

        if resume and self.checkpoint_path.exists():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            completed = set(self.checkpoint.get("completed_case_indexes", []))
            self.checkpoint["config"] = config
            self.checkpoint["resumed"] = True
            print(f"Resuming: {len(completed)} case(s) already recorded; they will be skipped.")
        else:
            # Fresh run: truncate any previous partial artifacts so appends start clean.
            for path in (self.trace_path, self.training_jsonl_path, self.eval_rows_path):
                path.write_text("", encoding="utf-8")
            self.checkpoint = {
                "config": config,
                "completed_case_indexes": [],
                "batches_completed": 0,
                "trace_rows": 0,
                "training_rows": 0,
                "eval_rows": 0,
                "resumed": False,
            }
            self._write_checkpoint()

        return completed

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _write_checkpoint(self) -> None:
        # Write to a temp file then atomically replace so a crash can't corrupt the checkpoint.
        tmp_path = self.checkpoint_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.checkpoint, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(self.checkpoint_path)

    def commit_batch(
        self,
        *,
        batch_index: int,
        case_indexes: list[int],
        trace_rows: list[dict[str, Any]],
        training_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]],
    ) -> None:
        """Persist one batch and advance the checkpoint. Called after each batch finishes."""
        self._append_jsonl(self.trace_path, trace_rows)
        self._append_jsonl(self.training_jsonl_path, training_rows)
        self._append_jsonl(self.eval_rows_path, eval_rows)

        completed = set(self.checkpoint.get("completed_case_indexes", []))
        completed.update(case_indexes)
        self.checkpoint["completed_case_indexes"] = sorted(completed)
        self.checkpoint["batches_completed"] = self.checkpoint.get("batches_completed", 0) + 1
        self.checkpoint["trace_rows"] = self.checkpoint.get("trace_rows", 0) + len(trace_rows)
        self.checkpoint["training_rows"] = self.checkpoint.get("training_rows", 0) + len(training_rows)
        self.checkpoint["eval_rows"] = self.checkpoint.get("eval_rows", 0) + len(eval_rows)
        self._write_checkpoint()

        print(
            f"[batch {batch_index}] committed cases {case_indexes[0]}..{case_indexes[-1]} "
            f"(+{len(trace_rows)} trace, +{len(training_rows)} training rows). "
            f"Totals: {self.checkpoint['trace_rows']} trace / "
            f"{self.checkpoint['training_rows']} training rows."
        )

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def finalize(self, *, run: TeacherRunSpec, config: dict[str, Any]) -> dict[str, Any]:
        """Build the consolidated typed dataset + summary from the accumulated JSONL files."""
        import pandas as pd
        from llm_rl_playground.ecthr_evaluation import summarize_ecthr_cases

        training_rows = self._read_jsonl(self.training_jsonl_path)
        eval_rows = self._read_jsonl(self.eval_rows_path)
        trace_rows_count = self.checkpoint.get("trace_rows", 0)

        typed_dataset = make_typed_traversal_dataset(training_rows)
        typed_dataset.to_json_path(self.typed_dataset_path)

        df = pd.DataFrame([{col: row.get(col) for col in _SUMMARY_COLUMNS} for row in eval_rows])
        metrics = summarize_ecthr_cases(df).to_dict(orient="records") if not df.empty else []

        summary = {
            "backend": run.backend,
            "model": run.model_id,
            "label": run.label,
            "config": config,
            "trace_rows": trace_rows_count,
            "training_rows": len(training_rows),
            "eval_rows": len(eval_rows),
            "cases_completed": len(self.checkpoint.get("completed_case_indexes", [])),
            "batches_completed": self.checkpoint.get("batches_completed", 0),
            "metrics": metrics,
            "paths": {
                "trace": str(self.trace_path),
                "training_jsonl": str(self.training_jsonl_path),
                "eval_rows": str(self.eval_rows_path),
                "typed_dataset": str(self.typed_dataset_path),
                "checkpoint": str(self.checkpoint_path),
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = repo_root_from_script()
    default_tree_path = repo_root / "trees" / "EU" / "eu_conventions_notebook" / "eu_conventions_tree-bottom-up-llm.pkl"
    default_output_dir = repo_root / "src" / "llm_rl_playground" / "outputs" / "ecthr-distillation"

    parser = argparse.ArgumentParser(
        description=(
            "Run ECtHR LATTICE traversal with a teacher model on any backend and save the "
            "prompt/response traces as a checkpointed, batched supervised distillation dataset."
        )
    )

    # --- Teacher model / backend selection ---
    parser.add_argument("--backend", default="vllm", help="vllm | openai | genai | local (default: vllm)")
    parser.add_argument("--model", default=None, help="Model id. Defaults to $LATTICE_VLLM_MODEL for vllm.")
    parser.add_argument("--lora", "--adapter", dest="lora", default=None,
                        help="LoRA/adapter dir. Only applied for the local backend.")
    parser.add_argument("--base-url", default=None,
                        help="vLLM endpoint(s), comma-separated. Defaults to $VLLM_BASE_URL or localhost:8000.")
    parser.add_argument("--label", default=None,
                        help="Run label / output prefix. Defaults to $LATTICE_VLLM_LABEL or a slug of --model.")

    # --- Dataset / tree / output ---
    parser.add_argument("--tree-path", default=str(default_tree_path))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--split", default="train")

    # --- Batching / checkpointing ---
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Cases per checkpointed batch. Each batch is flushed to disk before the next starts.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases already recorded in the checkpoint under this output-dir/label.")

    # --- Traversal / scoring controls ---
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--top-k-leaves", type=int, default=4)
    parser.add_argument("--prediction-min-score", type=float, default=0.7)
    parser.add_argument("--max-predicted-articles", type=int, default=None)
    parser.add_argument("--max-beam-size", type=int, default=8)
    parser.add_argument("--relevance-chain-factor", type=float, default=0.5)
    parser.add_argument("--reasoning-in-traversal-prompt", type=int, default=-1)
    parser.add_argument("--max-prompt-proto-size", type=int, default=None)
    parser.add_argument("--max-doc-desc-char-len", type=int, default=None)

    # --- API controls ---
    parser.add_argument("--max-concurrent-calls", type=int, default=32)
    parser.add_argument("--parse-max-concurrent-calls", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--staggering-delay", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=384,
                        help="Teacher output token budget (mapped to each backend's own arg name).")
    parser.add_argument("--vllm-guided-json", action=argparse.BooleanOptionalAction, default=True,
                        help="For vLLM, constrain the teacher with guided_json decoding (default: on).")

    # --- Local backend controls ---
    parser.add_argument("--local-use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-serialize-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-api-calls", action="store_true",
                        help="Also write a verbose teacher prompt/response log next to the main log file.")

    # --- Logging ---
    parser.add_argument("--detailed-logs", action="store_true")
    parser.add_argument("--show-error-logs", action="store_true")
    return parser


def main() -> None:
    """Build the teacher, run traced LATTICE in checkpointed batches, and save the dataset."""
    ensure_src_on_path()
    args = build_arg_parser().parse_args()

    from llm_rl_playground.ecthr_evaluation import (
        EcthrTraversalEvaluator,
        facts_to_case_prompt,
        get_label_names,
        load_ecthr_dataset,
        summarize_ecthr_cases,
    )
    from tree_construction.build_llm_bottom_up_tree import run_coro_sync
    from utils import compute_node_registry, setup_logger

    run = build_run_spec(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = Path(args.tree_path).expanduser().resolve()
    if not tree_path.exists():
        raise FileNotFoundError(f"Semantic tree does not exist: {tree_path}")
    semantic_root_node = load_semantic_tree(tree_path)
    node_registry = compute_node_registry(semantic_root_node)

    logger = setup_logger(
        f"create_ecthr_distillation_dataset.{run.label}",
        str(output_dir / f"{run.label}_create_ecthr_distillation_dataset.log"),
    )

    hp = make_hyperparams(args, tree_path, run)
    llm_api_kwargs = make_llm_api_kwargs(args, run.backend)

    config = {
        "backend": run.backend,
        "model": run.model_id,
        "label": run.label,
        "tree_path": str(tree_path),
        "split": args.split,
        "ecthr_config": args.ecthr_config,
        "start": args.start,
        "n_cases": args.n_cases,
        "batch_size": args.batch_size,
        "num_iters": args.num_iters,
        "top_k_leaves": args.top_k_leaves,
        "prediction_min_score": args.prediction_min_score,
        "max_tokens": args.max_tokens,
    }
    print(json.dumps({"run": asdict(run), "output_dir": str(output_dir), "config": config},
                     ensure_ascii=False, indent=2))

    dataset = load_ecthr_dataset(split=args.split, config=args.ecthr_config)
    label_names = get_label_names(dataset)

    end = min(args.start + args.n_cases, len(dataset))
    all_case_indexes = list(range(args.start, end))

    store = DistillationOutputStore(output_dir, run.label)
    already_done = store.start(resume=args.resume, config=config)
    pending = [idx for idx in all_case_indexes if idx not in already_done]
    if not pending:
        print("All requested cases are already recorded; nothing to do. Finalizing outputs.")
        summary = store.finalize(run=run, config=config)
        _print_final_summary(summary, summarize_ecthr_cases, store)
        return

    # The teacher API is built once and reused across every batch.
    teacher_api = make_teacher_api(args, run, logger)
    evaluator = EcthrTraversalEvaluator(
        semantic_root_node=semantic_root_node,
        node_registry=node_registry,
        hp=hp,
        logger=logger,
        llm_api=teacher_api,
        llm_api_kwargs=llm_api_kwargs,
        show_error_logs=args.show_error_logs,
    )

    batch_size = max(1, args.batch_size)
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"\nProcessing {len(pending)} pending case(s) in {len(batches)} batch(es) of up to {batch_size}.")

    try:
        for batch_index, case_indexes in enumerate(batches, start=1):
            print(f"\n########## Batch {batch_index}/{len(batches)}: cases {case_indexes} ##########")
            selected = [dataset[idx] for idx in case_indexes]
            queries: list[str] = []
            samples: list[Any] = []
            for example in selected:
                facts = example.get("text") or example.get("facts") or []
                query = facts_to_case_prompt(facts)
                queries.append(query)
                samples.append(evaluator.make_sample(query))

            trace_rows = run_coro_sync(
                run_lattice_with_trace_async(
                    evaluator,
                    samples,
                    case_indexes,
                    num_iters=args.num_iters,
                    detailed_logs=args.detailed_logs,
                )
            )
            training_rows = make_training_rows(trace_rows)
            eval_rows = build_results_from_samples(
                selected=selected,
                queries=queries,
                samples=samples,
                case_indexes=case_indexes,
                label_names=label_names,
                top_k=args.top_k_leaves,
                prediction_min_score=args.prediction_min_score,
                max_predicted_articles=args.max_predicted_articles,
            )
            store.commit_batch(
                batch_index=batch_index,
                case_indexes=case_indexes,
                trace_rows=trace_rows,
                training_rows=training_rows,
                eval_rows=eval_rows,
            )
    finally:
        unload_teacher_api(teacher_api)

    summary = store.finalize(run=run, config=config)
    _print_final_summary(summary, summarize_ecthr_cases, store)


def _print_final_summary(summary: dict[str, Any], summarize_ecthr_cases, store: DistillationOutputStore) -> None:
    print("\n================ Distillation dataset ================")
    print(
        {
            "backend": summary["backend"],
            "model": summary["model"],
            "cases_completed": summary["cases_completed"],
            "trace_rows": summary["trace_rows"],
            "training_rows": summary["training_rows"],
        }
    )
    if summary.get("metrics"):
        import pandas as pd

        print("\n================ Summary ================")
        print(pd.DataFrame(summary["metrics"]).to_string(index=False))
    print(f"\nSaved full teacher trace rows to:   {store.trace_path}")
    print(f"Saved supervised training JSONL to: {store.training_jsonl_path}")
    print(f"Saved typed TraversalDataset JSON:  {store.typed_dataset_path}")
    print(f"Saved run summary JSON to:          {store.summary_path}")
    print(f"Checkpoint (for --resume):          {store.checkpoint_path}")


if __name__ == "__main__":
    main()
