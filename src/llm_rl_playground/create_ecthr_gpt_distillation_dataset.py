from __future__ import annotations

import argparse
import json
import pickle
import sys
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


def parse_args() -> argparse.Namespace:
    """Collect the teacher model, ECtHR slice, traversal, and output options for one distillation run."""
    repo_root = repo_root_from_script()
    default_tree_path = repo_root / "trees" / "EU" / "eu_conventions_notebook" / "eu_conventions_tree-bottom-up-llm.pkl"
    default_output_dir = repo_root / "src" / "llm_rl_playground" / "outputs" / "ecthr-gpt-distillation"

    parser = argparse.ArgumentParser(
        description=(
            "Run ECtHR LATTICE traversal with an OpenAI GPT teacher and save the prompt/response "
            "traces as a supervised distillation dataset."
        )
    )
    parser.add_argument("--teacher-model", default="gpt-5.5")
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--tree-path", default=str(default_tree_path))
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--top-k-leaves", type=int, default=10)
    parser.add_argument("--prediction-min-score", type=float, default=0.4)
    parser.add_argument("--max-predicted-articles", type=int, default=None)
    parser.add_argument("--max-concurrent-calls", type=int, default=8)
    parser.add_argument("--parse-max-concurrent-calls", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--show-error-logs", action="store_true")
    parser.add_argument(
        "--log-api-calls",
        action="store_true",
        help="Also write a verbose OpenAI prompt/response log next to the main log file.",
    )
    return parser.parse_args()


def load_semantic_tree(tree_path: Path):
    """Load the semantic tree in the object form expected by InferSample and the evaluator."""
    from tree_objects import SemanticNode

    if tree_path.suffix == ".pkl":
        tree_obj = pickle.loads(tree_path.read_bytes())
    else:
        tree_obj = json.loads(tree_path.read_text(encoding="utf-8"))
    return SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write large trace/training outputs as JSONL so they can be streamed or inspected line by line."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


async def run_lattice_with_trace_async(
    evaluator: Any,
    samples: list[Any],
    case_indexes: list[int],
    *,
    num_iters: int,
    detailed_logs: bool = False,
) -> list[dict[str, Any]]:
    """Run LATTICE externally while recording each GPT teacher decision.

    This mirrors EcthrTraversalEvaluator.run_lattice_iterations_for_samples_async, but keeps
    the recording logic outside the normal evaluator. Each iteration gathers the current
    prompts, asks the teacher model, stores prompt/response metadata, then updates the samples.
    """
    trace_rows: list[dict[str, Any]] = []

    for step in range(num_iters):
        print(f"\n--- Teacher distillation iteration {step + 1}/{num_iters} ({len(samples)} cases) ---")
        # LATTICE exposes prompts per sample; flatten them so GPT can process one efficient batch.
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
    label_names: list[str] | None,
    start: int,
    top_k: int,
    prediction_min_score: float | None,
    max_predicted_articles: int | None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Score the completed traced samples without rerunning LATTICE.

    The distillation loop already mutated each sample with GPT teacher decisions, so this
    function only extracts top leaf predictions and computes the same metrics as evaluation.
    """
    import pandas as pd
    from llm_rl_playground.ecthr_evaluation import (
        example_gold_articles,
        predicted_articles_from_sample,
        score_prediction,
    )

    results = []
    for local_idx, (example, query, sample) in enumerate(zip(selected, queries, samples)):
        case_idx = start + local_idx
        predicted, top_rows = predicted_articles_from_sample(
            sample,
            k=top_k,
            min_score=prediction_min_score,
            max_articles=max_predicted_articles,
        )
        gold = example_gold_articles(example, label_names)
        metrics = score_prediction(gold, predicted)
        results.append(
            {
                "case_index": case_idx,
                "query": query,
                "gold": sorted(gold),
                "predicted": sorted(predicted),
                "lattice_predicted": sorted(predicted),
                "gold_removed_by_selector": 0,
                "gold_removed_by_selector_articles": [],
                "top_rows": top_rows,
                "selector_result": None,
                "sample": sample,
                **metrics,
            }
        )

        print(f"\n================ ECtHR case {case_idx} ================")
        print("Gold:     ", results[-1]["gold"])
        print("Predicted: ", results[-1]["predicted"])
        print(
            f"Correct(all gold found) : {results[-1]['all_gold_found']} | "
            f"Recall: {results[-1]['recall']:.2f} | Precision: {results[-1]['precision']:.2f} | "
            f"F1: {results[-1]['f1']:.2f}"
        )

    df = pd.DataFrame(
        [
            {
                "case_index": result["case_index"],
                "gold": result["gold"],
                "predicted": result["predicted"],
                "lattice_predicted": result["lattice_predicted"],
                "gold_removed_by_selector": result["gold_removed_by_selector"],
                "gold_removed_by_selector_articles": result["gold_removed_by_selector_articles"],
                "any_gold_found": result["any_gold_found"],
                "all_gold_found": result["all_gold_found"],
                "exact_set_match": result["exact_set_match"],
                "true_positive": result["true_positive"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
            for result in results
        ]
    )
    return df, results


def main() -> None:
    """Build the GPT teacher, run traced LATTICE, and save both traces and training datasets."""
    ensure_src_on_path()
    args = parse_args()

    from hyperparams import HyperParams
    from llm_apis import OpenAIResponsesAPI
    from llm_rl_playground.ecthr_evaluation import (
        EcthrTraversalEvaluator,
        facts_to_case_prompt,
        get_label_names,
        load_ecthr_dataset,
        summarize_ecthr_cases,
    )
    from prompts import get_traversal_prompt_response_constraint
    from tree_construction.build_llm_bottom_up_tree import run_coro_sync
    from utils import compute_node_registry, setup_logger

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = Path(args.tree_path).expanduser().resolve()
    semantic_root_node = load_semantic_tree(tree_path)
    node_registry = compute_node_registry(semantic_root_node)

    logger = setup_logger(
        "create_ecthr_gpt_distillation_dataset",
        str(output_dir / "create_ecthr_gpt_distillation_dataset.log"),
    )

    hp = HyperParams.from_args("--subset fiqa --tree_version eu_conventions_notebook")
    hp.TREE_PATH = str(tree_path)
    hp.DATASET = "EU"
    hp.LLM_API_BACKEND = "openai"
    hp.LLM = args.teacher_model
    hp.LLM_API_TIMEOUT = 120
    hp.LLM_API_MAX_RETRIES = 4
    hp.LLM_MAX_CONCURRENT_CALLS = args.max_concurrent_calls
    hp.LLM_API_STAGGERING_DELAY = 0.05
    hp.REASONING_IN_TRAVERSAL_PROMPT = -1
    hp.SUBSET = "fiqa"
    hp.MAX_BEAM_SIZE = 8
    hp.SEARCH_WITH_PATH_RELEVANCE = True
    hp.NUM_LEAF_CALIB = 0
    hp.RELEVANCE_CHAIN_FACTOR = 0.5
    hp.MAX_PROMPT_PROTO_SIZE = None
    hp.MAX_DOC_DESC_CHAR_LEN = None
    hp.SHOW_ERROR_LOGS = args.show_error_logs

    # The GPT teacher is only used to label traversal decisions; local student training happens later.
    llm_api = OpenAIResponsesAPI(
        args.teacher_model,
        logger=logger,
        timeout=hp.LLM_API_TIMEOUT,
        max_retries=hp.LLM_API_MAX_RETRIES,
        log_api_calls=args.log_api_calls,
    )
    llm_api_kwargs = {
        "max_concurrent_calls": hp.LLM_MAX_CONCURRENT_CALLS,
        "response_schema": get_traversal_prompt_response_constraint(bool(hp.REASONING_IN_TRAVERSAL_PROMPT)),
        "staggering_delay": hp.LLM_API_STAGGERING_DELAY,
        "print_summary_report": False,
        "show_error_logs": args.show_error_logs,
        "parse_max_concurrent_calls": args.parse_max_concurrent_calls or hp.LLM_MAX_CONCURRENT_CALLS,
        "max_output_tokens": args.max_output_tokens,
    }

    evaluator = EcthrTraversalEvaluator(
        semantic_root_node=semantic_root_node,
        node_registry=node_registry,
        hp=hp,
        logger=logger,
        llm_api=llm_api,
        llm_api_kwargs=llm_api_kwargs,
        show_error_logs=args.show_error_logs,
    )

    dataset = load_ecthr_dataset(split=args.split, config=args.ecthr_config)
    label_names = get_label_names(dataset)
    selected = list(dataset.select(range(args.start, min(args.start + args.n_cases, len(dataset)))))
    # Build queries and samples once so the external recorder loop owns all traversal side effects.
    queries = []
    samples = []
    for example in selected:
        facts = example.get("text") or example.get("facts") or []
        query = facts_to_case_prompt(facts)
        queries.append(query)
        samples.append(evaluator.make_sample(query))
    case_indexes = list(range(args.start, args.start + len(selected)))

    trace_rows = run_coro_sync(
        run_lattice_with_trace_async(
            evaluator,
            samples,
            case_indexes,
            num_iters=args.num_iters,
            detailed_logs=False,
        )
    )
    df, results = build_results_from_samples(
        selected=selected,
        queries=queries,
        samples=samples,
        label_names=label_names,
        start=args.start,
        top_k=args.top_k_leaves,
        prediction_min_score=args.prediction_min_score,
        max_predicted_articles=args.max_predicted_articles,
    )
    print("\n================ Summary ================")
    print(summarize_ecthr_cases(df).to_string(index=False))

    # Keep full traces for auditing, but train only on rows with parseable structured teacher JSON.
    training_rows = make_training_rows(trace_rows)
    typed_dataset = make_typed_traversal_dataset(training_rows)

    trace_path = output_dir / "gpt_teacher_lattice_trace_rows.jsonl"
    training_jsonl_path = output_dir / "gpt_teacher_training_rows.jsonl"
    typed_dataset_path = output_dir / "gpt_teacher_traversal_dataset.json"
    eval_rows_path = output_dir / "gpt_teacher_eval_rows.json"
    eval_results_path = output_dir / "gpt_teacher_eval_results.json"
    summary_path = output_dir / "gpt_teacher_summary.json"

    write_jsonl(trace_path, trace_rows)
    write_jsonl(training_jsonl_path, training_rows)
    typed_dataset.to_json_path(typed_dataset_path)
    df.to_json(eval_rows_path, orient="records", indent=2)
    eval_results_path.write_text(
        json.dumps(
            [{key: value for key, value in result.items() if key != "sample"} for result in results],
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "teacher_model": args.teacher_model,
                "split": args.split,
                "start": args.start,
                "n_cases": args.n_cases,
                "num_iters": args.num_iters,
                "trace_rows": len(trace_rows),
                "training_rows": len(training_rows),
                "valid_trace_rows": sum(1 for row in trace_rows if row.get("valid")),
                "metrics": summarize_ecthr_cases(df).to_dict(orient="records"),
                "paths": {
                    "trace": str(trace_path),
                    "training_jsonl": str(training_jsonl_path),
                    "typed_dataset": str(typed_dataset_path),
                    "eval_rows": str(eval_rows_path),
                    "eval_results": str(eval_results_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n================ Distillation dataset ================")
    print({"trace_rows": len(trace_rows), "training_rows": len(training_rows)})
    print(f"Saved full GPT trace rows to: {trace_path}")
    print(f"Saved supervised training JSONL to: {training_jsonl_path}")
    print(f"Saved typed TraversalDataset JSON to: {typed_dataset_path}")


if __name__ == "__main__":
    main()
