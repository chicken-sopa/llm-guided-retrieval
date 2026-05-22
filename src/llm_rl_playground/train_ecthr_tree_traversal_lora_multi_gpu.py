from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


TREE_ARTICLE_RE = re.compile(
    r"(?:Protocol\s+(?P<protocol>\d+)\s+)?Art\.\s*(?P<article>\d+[A-Za-z]?)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU LoRA/QLoRA training for ECtHR semantic-tree traversal."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--tree-path",
        default=None,
        help="Path to EU semantic tree JSON. Defaults to repo trees/EU/eu_conventions_notebook/eu_conventions_tree-bottom-up-llm.json.",
    )
    parser.add_argument("--output-dir", default="outputs/qwen2.5-1.5b-ecthr-tree-traversal-lora")
    parser.add_argument("--ecthr-dataset", default="AUEB-NLP/ecthr_cases")
    parser.add_argument("--ecthr-config", default="alleged-violation-prediction")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--max-train-cases", type=int, default=1500)
    parser.add_argument("--max-eval-cases", type=int, default=200)
    parser.add_argument("--max-examples-per-case", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=8000)
    parser.add_argument("--max-eval-examples", type=int, default=1000)
    parser.add_argument("--max-fact-chars", type=int, default=9000)
    parser.add_argument("--max-child-desc-chars", type=int, default=1100)
    parser.add_argument("--max-path-chars", type=int, default=1800)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--include-single-child-nodes", action="store_true")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-steps", type=int, default=0, help="If > 0, save every N steps; otherwise save per epoch.")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-ecthr-batched-eval",
        action="store_true",
        help="After training, run the ECtHR LATTICE batched evaluation with the saved adapter on rank 0.",
    )
    parser.add_argument(
        "--compare-base-model",
        action="store_true",
        help="When post-training ECtHR eval is enabled, also evaluate the base model with no adapter.",
    )
    parser.add_argument(
        "--ecthr-eval-tree-path",
        default=None,
        help="Tree path for LATTICE evaluation. Defaults to the repo .pkl EU tree, falling back to --tree-path.",
    )
    parser.add_argument("--ecthr-eval-n-cases", type=int, default=5)
    parser.add_argument("--ecthr-eval-start", type=int, default=0)
    parser.add_argument("--ecthr-eval-num-iters", type=int, default=6)
    parser.add_argument("--ecthr-eval-top-k-leaves", type=int, default=10)
    parser.add_argument("--ecthr-eval-prediction-min-score", type=float, default=0.4)
    parser.add_argument("--ecthr-eval-max-predicted-articles", type=int, default=None)
    parser.add_argument(
        "--ecthr-eval-use-llm-selector",
        action="store_true",
        help="Use the final LLM article selector during post-training ECtHR evaluation.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def clean_space(value) -> str:
    return " ".join(str(value or "").split())


def normalize_article_label(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None

    s = s.replace("no violation", "")
    s = s.replace("non violation", "")
    s = s.replace("non-violation", "")
    s = s.replace(".", "")
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"^echr_", "", s)
    s = re.sub(r"^convention_", "", s)

    if re.fullmatch(r"\d+[a-z]?", s):
        return f"article_{s}"

    match = re.fullmatch(r"p(\d+)_(\d+)", s)
    if match:
        return f"protocol_{int(match.group(1))}_article_{int(match.group(2))}"

    match = re.search(r"article_?(\d+[a-z]?)_(?:of_)?protocol_?(\d+)", s)
    if match:
        return f"protocol_{int(match.group(2))}_article_{match.group(1)}"

    match = re.search(r"protocol_?(\d+)_article_?(\d+[a-z]?)", s)
    if match:
        return f"protocol_{int(match.group(1))}_article_{match.group(2)}"

    match = re.search(r"article_?(\d+[a-z]?)", s)
    if match:
        return f"article_{match.group(1)}"

    return None


def article_id_to_display(article_id: str) -> str:
    match = re.fullmatch(r"article_(\d+[a-z]?)", article_id)
    if match:
        return f"Article {match.group(1).upper()}"

    match = re.fullmatch(r"protocol_(\d+)_article_(\d+[a-z]?)", article_id)
    if match:
        return f"Protocol {match.group(1)}, Article {match.group(2).upper()}"

    return article_id


def extract_articles_from_tree_text(text: str) -> set[str]:
    articles = set()
    for match in TREE_ARTICLE_RE.finditer(text or ""):
        article = match.group("article").lower()
        protocol = match.group("protocol")
        if protocol:
            articles.add(f"protocol_{int(protocol)}_article_{article}")
        else:
            articles.add(f"article_{article}")
    return articles


def example_gold_articles(example: dict, label_column: str = "labels") -> list[str]:
    raw_labels = example.get(label_column, [])
    if raw_labels is None:
        return []
    if isinstance(raw_labels, (int, str)):
        raw_labels = [raw_labels]

    normalized = []
    for raw_label in raw_labels:
        article_id = normalize_article_label(raw_label)
        if article_id and article_id not in normalized:
            normalized.append(article_id)
    return sorted(normalized)


def facts_to_text(example: dict, max_chars: int) -> str:
    facts = example.get("text") or example.get("facts") or []
    if isinstance(facts, str):
        fact_text = facts
    else:
        fact_text = "\n".join(f"- {clean_space(fact)}" for fact in facts if clean_space(fact))
    if len(fact_text) > max_chars:
        fact_text = fact_text[:max_chars] + "\n- [facts truncated]"
    return fact_text


def node_label(node: dict) -> str:
    meta = node.get("metadata") or {}
    payload = meta.get("summary_payload") or {}
    return clean_space(payload.get("label") or meta.get("label") or node.get("id") or "node")


def load_tree(tree_path: Path) -> dict:
    with tree_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotate_subtree_articles(node: dict) -> set[str]:
    children = node.get("child") or []
    if not children:
        articles = extract_articles_from_tree_text(node.get("desc", ""))
    else:
        articles = set()
        for child in children:
            articles |= annotate_subtree_articles(child)
    node["_subtree_articles"] = sorted(articles)
    return articles


def format_child_options(children: list[dict], max_child_desc_chars: int) -> str:
    parts = []
    for idx, child in enumerate(children):
        desc = clean_space(child.get("desc"))[:max_child_desc_chars]
        parts.append(f"[{idx}]. {desc}")
    return "\n\n".join(parts)


def format_current_path(path_labels: list[str], max_path_chars: int) -> str:
    text = " -> ".join(path_labels)
    if len(text) > max_path_chars:
        text = text[-max_path_chars:]
    return text or "ROOT"


def build_traversal_prompt(
    facts: str,
    path_labels: list[str],
    children: list[dict],
    *,
    max_child_desc_chars: int,
    max_path_chars: int,
) -> str:
    valid_ids = ", ".join(str(i) for i in range(len(children)))
    return f"""You are an intelligent search agent navigating a hierarchical semantic tree of European Convention law.

Given the facts of an ECtHR case, choose the child nodes most likely to lead to the Convention or Protocol articles alleged by the applicant.

Relevance definition:
A child node is relevant when its subtree is a strong legal path toward an article that could be alleged from the case facts. Penalize nodes that are only background, procedural context, or weakly related.

Case facts:
{facts}

Current tree path:
{format_current_path(path_labels, max_path_chars)}

Candidate child nodes:
Valid candidate IDs for this request: {valid_ids}.

{format_child_options(children, max_child_desc_chars)}

Return one clean JSON object with exactly these keys: reasoning, ranking, relevance_scores.
The ranking must include only valid candidate IDs, ordered from most to least relevant.
The relevance_scores field must be an array of [candidate_id, score] pairs with scores from 0 to 100."""


def make_traversal_answer(positive_ids: list[int], children: list[dict], gold_articles: set[str]) -> dict:
    positive_set = set(positive_ids)
    ranking = list(positive_ids) + [idx for idx in range(len(children)) if idx not in positive_set]
    scores = [[idx, 95 if idx in positive_set else 15] for idx in ranking]
    selected_labels = [node_label(children[idx]) for idx in positive_ids]
    gold_display = [article_id_to_display(article_id) for article_id in sorted(gold_articles)]
    return {
        "reasoning": (
            "The top-ranked child nodes are on oracle paths from the case facts to the alleged article labels. "
            f"Selected child labels: {selected_labels}. Alleged labels: {gold_display}."
        ),
        "ranking": ranking,
        "relevance_scores": scores,
    }


def collect_case_traversal_examples(
    node: dict,
    *,
    facts: str,
    gold_articles: set[str],
    path_labels: list[str],
    rows: list[dict],
    include_single_child_nodes: bool,
    max_child_desc_chars: int,
    max_path_chars: int,
) -> None:
    children = node.get("child") or []
    if not children:
        return

    positive_ids = [
        idx
        for idx, child in enumerate(children)
        if set(child.get("_subtree_articles") or []) & gold_articles
    ]
    if not positive_ids:
        return

    if include_single_child_nodes or len(children) > 1:
        prompt = build_traversal_prompt(
            facts=facts,
            path_labels=path_labels,
            children=children,
            max_child_desc_chars=max_child_desc_chars,
            max_path_chars=max_path_chars,
        )
        answer = make_traversal_answer(positive_ids, children, gold_articles)
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an ECtHR semantic-tree traversal model. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                ]
            }
        )

    next_path = path_labels + [node_label(node)]
    for idx in positive_ids:
        collect_case_traversal_examples(
            children[idx],
            facts=facts,
            gold_articles=gold_articles,
            path_labels=next_path,
            rows=rows,
            include_single_child_nodes=include_single_child_nodes,
            max_child_desc_chars=max_child_desc_chars,
            max_path_chars=max_path_chars,
        )


def make_examples_for_case(example: dict, tree: dict, all_tree_articles: set[str], args: argparse.Namespace, rng: random.Random) -> list[dict]:
    gold_articles = set(example_gold_articles(example))
    gold_articles &= all_tree_articles
    if not gold_articles:
        return []

    rows = []
    collect_case_traversal_examples(
        tree,
        facts=facts_to_text(example, args.max_fact_chars),
        gold_articles=gold_articles,
        path_labels=["ROOT"],
        rows=rows,
        include_single_child_nodes=args.include_single_child_nodes,
        max_child_desc_chars=args.max_child_desc_chars,
        max_path_chars=args.max_path_chars,
    )
    rng.shuffle(rows)
    return rows[: args.max_examples_per_case]


def build_traversal_rows(
    dataset,
    tree: dict,
    all_tree_articles: set[str],
    *,
    max_cases: int,
    max_rows: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    indexes = list(range(len(dataset)))
    rng.shuffle(indexes)

    rows = []
    cases_used = 0
    skipped_cases = 0
    for idx in indexes[:max_cases]:
        case_rows = make_examples_for_case(dataset[idx], tree, all_tree_articles, args, rng)
        if not case_rows:
            skipped_cases += 1
            continue
        rows.extend(case_rows)
        cases_used += 1
        if len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
    return rows, {"cases_used": cases_used, "skipped_cases": skipped_cases, "rows": len(rows)}


def tokenize_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer, max_length: int, desc: str) -> Dataset:
    def preprocess(example: dict) -> dict:
        messages = example["messages"]
        prompt_messages = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    tokenized = raw_dataset.map(
        preprocess,
        remove_columns=raw_dataset.column_names,
        desc=desc,
    )
    return tokenized.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc=f"{desc}: dropping fully masked examples",
    )


def resolve_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_eval_semantic_tree(eval_tree_path: Path):
    from tree_objects import SemanticNode

    if eval_tree_path.suffix == ".pkl":
        tree_obj = pickle.loads(eval_tree_path.read_bytes())
    else:
        tree_obj = json.loads(eval_tree_path.read_text(encoding="utf-8"))
    return SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj


def run_post_training_ecthr_batched_eval(args: argparse.Namespace, output_dir: Path, repo_root: Path, train_tree_path: Path) -> None:
    """Evaluate the saved traversal adapter with the reusable ECtHR batched evaluator."""
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import pandas as pd

    from hyperparams import HyperParams
    from llm_apis import LocalModelAPI
    from llm_rl_playground.ecthr_evaluation import (
        EcthrTraversalEvaluator,
        get_label_names,
        load_ecthr_dataset,
        summarize_ecthr_cases,
    )
    from prompts import get_traversal_prompt_response_constraint
    from utils import compute_node_registry, setup_logger

    default_eval_tree_path = repo_root / "trees" / "EU" / "eu_conventions_notebook" / "eu_conventions_tree-bottom-up-llm.pkl"
    eval_tree_path = Path(args.ecthr_eval_tree_path) if args.ecthr_eval_tree_path else default_eval_tree_path
    if not eval_tree_path.exists():
        eval_tree_path = train_tree_path

    semantic_root_node = load_eval_semantic_tree(eval_tree_path)
    node_registry = compute_node_registry(semantic_root_node)

    eval_hp = HyperParams.from_args("--subset fiqa --tree_version eu_conventions_notebook")
    eval_hp.TREE_PATH = str(eval_tree_path)
    eval_hp.DATASET = "EU"
    eval_hp.LLM_API_BACKEND = "localModel"
    eval_hp.LLM = args.model_id
    eval_hp.LLM_API_TIMEOUT = 120
    eval_hp.LLM_API_MAX_RETRIES = 4
    eval_hp.LLM_MAX_CONCURRENT_CALLS = 1
    eval_hp.LLM_API_STAGGERING_DELAY = 0.05
    eval_hp.REASONING_IN_TRAVERSAL_PROMPT = -1
    eval_hp.SUBSET = "fiqa"
    eval_hp.MAX_BEAM_SIZE = 8
    eval_hp.SEARCH_WITH_PATH_RELEVANCE = True
    eval_hp.NUM_LEAF_CALIB = 0
    eval_hp.RELEVANCE_CHAIN_FACTOR = 0.5
    eval_hp.MAX_PROMPT_PROTO_SIZE = None
    eval_hp.MAX_DOC_DESC_CHAR_LEN = None

    eval_logger = setup_logger(
        "train_ecthr_tree_traversal_lora_multi_gpu_eval",
        str(output_dir / "ecthr_batched_eval.log"),
    )
    eval_llm_api_kwargs = {
        "max_concurrent_calls": eval_hp.LLM_MAX_CONCURRENT_CALLS,
        "response_mime_type": "application/json",
        "response_schema": get_traversal_prompt_response_constraint(bool(eval_hp.REASONING_IN_TRAVERSAL_PROMPT)),
        "staggering_delay": eval_hp.LLM_API_STAGGERING_DELAY,
        "print_summary_report": False,
        "max_new_tokens": 384,
    }

    eval_dataset = load_ecthr_dataset(split=args.eval_split, config=args.ecthr_config)
    label_names = get_label_names(eval_dataset)

    print("\n## 12. ECtHR Batched Evaluation")
    print(
        "This section evaluates whether the trained traversal adapter improves LATTICE retrieval on "
        "`alleged-violation-prediction`. It uses the reusable `EcthrTraversalEvaluator` module and can "
        "optionally compare against the base Qwen model with no adapter."
    )
    print(
        {
            "eval_tree_path": str(eval_tree_path),
            "eval_cases": args.ecthr_eval_n_cases,
            "eval_start": args.ecthr_eval_start,
            "eval_num_iters": args.ecthr_eval_num_iters,
            "eval_top_k_leaves": args.ecthr_eval_top_k_leaves,
            "compare_base_model": args.compare_base_model,
        }
    )

    def make_local_ecthr_evaluator(adapter_path: Path | None) -> EcthrTraversalEvaluator:
        api = LocalModelAPI(
            args.model_id,
            logger=eval_logger,
            timeout=eval_hp.LLM_API_TIMEOUT,
            max_retries=eval_hp.LLM_API_MAX_RETRIES,
            adapter_path=None if adapter_path is None else str(adapter_path),
            use_4bit=args.use_4bit,
            serialize_requests=True,
            log_api_calls=False,
        )
        return EcthrTraversalEvaluator(
            semantic_root_node=semantic_root_node,
            node_registry=node_registry,
            hp=eval_hp,
            logger=eval_logger,
            llm_api=api,
            llm_api_kwargs=eval_llm_api_kwargs,
        )

    def run_one_eval(label: str, adapter_path: Path | None):
        print(f"\n================ Running {label} ================")
        evaluator = make_local_ecthr_evaluator(adapter_path)
        df, results = evaluator.evaluate_ecthr_cases_batched(
            eval_dataset,
            label_names,
            n_cases=args.ecthr_eval_n_cases,
            num_iters=args.ecthr_eval_num_iters,
            top_k=args.ecthr_eval_top_k_leaves,
            start=args.ecthr_eval_start,
            prediction_min_score=args.ecthr_eval_prediction_min_score,
            max_predicted_articles=args.ecthr_eval_max_predicted_articles,
            use_llm_selector=args.ecthr_eval_use_llm_selector,
        )
        summary = summarize_ecthr_cases(df)
        summary.insert(0, "run", label)
        df.to_json(output_dir / f"{label}_ecthr_eval_rows.json", orient="records", indent=2)
        with (output_dir / f"{label}_ecthr_eval_results.json").open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {key: value for key, value in result.items() if key != "sample"}
                    for result in results
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        evaluator.llm_api.runtime.unload()
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary

    summaries = [run_one_eval("trained_tree_traversal_adapter", output_dir)]
    if args.compare_base_model:
        summaries.append(run_one_eval("base_model_no_adapter", None))

    comparison_df = pd.concat(summaries, ignore_index=True)
    comparison_path = output_dir / "ecthr_batched_eval_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n================ ECtHR Evaluation Comparison ================")
    print(comparison_df.to_string(index=False))
    print(f"Saved ECtHR comparison to: {comparison_path}")


def load_model(args: argparse.Namespace, compute_dtype: torch.dtype):
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA/QLoRA multi-GPU training expects CUDA GPUs.")

    rank = local_rank()
    torch.cuda.set_device(rank)

    model_kwargs = {
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # With 4-bit models, each DDP process must load its model on its own GPU.
        model_kwargs["device_map"] = {"": rank}

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(torch.device("cuda", rank))

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    repo_root = repo_root_from_script()
    tree_path = Path(args.tree_path) if args.tree_path else (
        repo_root / "trees" / "EU" / "eu_conventions_notebook" / "eu_conventions_tree-bottom-up-llm.json"
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(
            {
                "model_id": args.model_id,
                "tree_path": str(tree_path),
                "output_dir": str(output_dir),
                "world_size": world_size(),
                "use_4bit": args.use_4bit,
            }
        )

    tree = load_tree(tree_path)
    all_tree_articles = annotate_subtree_articles(tree)
    if is_main_process():
        print(f"Tree article IDs: {len(all_tree_articles)}")

    train_cases = load_dataset(
        args.ecthr_dataset,
        args.ecthr_config,
        split=args.train_split,
        trust_remote_code=True,
    )
    try:
        eval_cases = load_dataset(
            args.ecthr_dataset,
            args.ecthr_config,
            split=args.eval_split,
            trust_remote_code=True,
        )
    except Exception:
        eval_cases = None

    train_rows, train_stats = build_traversal_rows(
        train_cases,
        tree,
        all_tree_articles,
        max_cases=args.max_train_cases,
        max_rows=args.max_train_examples,
        seed=args.seed,
        args=args,
    )
    eval_rows, eval_stats = (
        build_traversal_rows(
            eval_cases,
            tree,
            all_tree_articles,
            max_cases=args.max_eval_cases,
            max_rows=args.max_eval_examples,
            seed=args.seed + 1,
            args=args,
        )
        if eval_cases is not None
        else ([], {})
    )

    if is_main_process():
        print({"train": train_stats, "eval": eval_stats})
        if train_rows:
            print("First training prompt preview:")
            print(train_rows[0]["messages"][1]["content"][:1200])
            print("First training answer:")
            print(train_rows[0]["messages"][2]["content"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = tokenize_dataset(
        Dataset.from_list(train_rows),
        tokenizer,
        args.max_length,
        desc="Tokenizing traversal train examples",
    )
    eval_dataset = (
        tokenize_dataset(
            Dataset.from_list(eval_rows),
            tokenizer,
            args.max_length,
            desc="Tokenizing traversal eval examples",
        )
        if eval_rows
        else None
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compute_dtype = resolve_compute_dtype()
    model = load_model(args, compute_dtype)
    if is_main_process():
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size() > 1 else None,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved LoRA adapter to: {output_dir}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.run_ecthr_batched_eval and is_main_process():
        run_post_training_ecthr_batched_eval(args, output_dir, repo_root, tree_path)


if __name__ == "__main__":
    main()
