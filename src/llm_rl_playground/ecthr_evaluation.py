from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TREE_ARTICLE_RE = re.compile(
    r"(?:Protocol\s+(?P<protocol>\d+)\s+)?Art\.\s*(?P<article>\d+[A-Za-z]?)",
    flags=re.IGNORECASE,
)


ARTICLE_SELECTOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation for selecting or rejecting the candidate articles.",
        },
        "selected_articles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Normalized article IDs copied exactly from the candidate list.",
        },
    },
    "required": ["reasoning", "selected_articles"],
}


def load_ecthr_dataset(split: str = "train", config: str = "alleged-violation-prediction"):
    """Load ECtHR Cases directly from Hugging Face."""
    from datasets import load_dataset

    return load_dataset(
        "AUEB-NLP/ecthr_cases",
        config,
        split=split,
        trust_remote_code=True,
    )


def get_label_names(dataset, label_column: str = "labels") -> list[str] | None:
    """No label-name lookup is needed for `AUEB-NLP/ecthr_cases`."""
    return None


def normalize_article_label(value: Any) -> str | None:
    """Normalize ECtHR dataset labels and article text labels to shared IDs."""
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


def example_gold_articles(
    example: dict,
    label_names: list[str] | None = None,
    label_column: str = "labels",
) -> set[str]:
    """Convert one ECtHR example's labels to normalized article IDs."""
    raw_labels = example.get(label_column, [])
    if raw_labels is None:
        return set()
    if isinstance(raw_labels, (int, str)):
        raw_labels = [raw_labels]

    gold = set()
    for raw_label in raw_labels:
        label_value = label_names[raw_label] if label_names and isinstance(raw_label, int) else raw_label
        normalized = normalize_article_label(label_value)
        if normalized:
            gold.add(normalized)
    return gold


def facts_to_case_prompt(facts: list[str] | str, max_chars: int = 12000) -> str:
    """Join ECtHR fact paragraphs into one retrieval query for LATTICE."""
    if isinstance(facts, str):
        facts = [facts]

    fact_text = "\n".join(f"- {str(fact).strip()}" for fact in facts if str(fact).strip())
    if len(fact_text) > max_chars:
        fact_text = fact_text[:max_chars] + "\n- [facts truncated]"

    return f"""You are an LLM traversing a semantic knowledge tree of European Convention law.

The tree is structured as follows:
- Internal nodes are semantic summaries of legal concepts or clusters of rights.
- Each internal node summarizes the information contained in its child nodes.
- Leaf nodes are specific Convention or Protocol articles and their provisions.

Given the facts of an ECtHR case, traverse the tree from the root toward the most legally relevant leaf articles.

Your task is not to retrieve every article that is semantically related. Your task is to identify the articles most likely to be used in an application before the European Court of Human Rights.

Traversal rules:
1. First identify the central legal injury in the facts.
2. Move only into child nodes whose summary directly matches that injury.
3. Penalize branches that are only factually mentioned, background-related, or speculative.
4. Prefer branches where the facts show an actual state interference, omission, procedural defect, or lack of remedy.
5. Continue traversal until you reach the strongest article leaves.
6. Return only leaf articles whose full path from the root is strongly supported by the facts.

For each selected article leaf:
- Give the article number and title.
- Explain the semantic path that led to it.
- Explain why it is a primary or secondary article.
- Retrieve the relevant provision text.
- Assign a relevance score from 0 to 1.

Do not include weak articles unless they are necessary to explain why they were rejected.

Case facts:
{fact_text}
"""


def case_facts_from_query(query: str) -> str:
    marker = "Case facts:\n"
    return query.split(marker, 1)[-1].strip() if marker in query else query.strip()


def extract_articles_from_tree_text(text: str) -> set[str]:
    """Extract normalized article IDs from retrieved EU conventions tree leaves."""
    predictions = set()
    for match in TREE_ARTICLE_RE.finditer(text or ""):
        article = match.group("article").lower()
        protocol = match.group("protocol")
        if protocol:
            predictions.add(f"protocol_{int(protocol)}_article_{article}")
        else:
            predictions.add(f"article_{article}")
    return predictions


def predicted_articles_from_sample(
    sample: InferSample,
    k: int = 10,
    min_score: float | None = None,
    max_articles: int | None = None,
) -> tuple[set[str], list[dict]]:
    """Take top LATTICE leaf predictions and extract article IDs from their text."""
    predicted = set()
    rows = []

    for rank, (node, score) in enumerate(sample.get_top_predictions(k=k), start=1):
        article_ids = extract_articles_from_tree_text(node.desc)
        passes_filters = bool(article_ids) and (min_score is None or float(score) >= min_score)
        included = False

        if passes_filters:
            for article_id in sorted(article_ids):
                if max_articles is not None and len(predicted) >= max_articles:
                    break
                predicted.add(article_id)
                included = True

        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "node_id": node.id,
                "included": included,
                "articles": sorted(article_ids),
                "text": node.desc[:800],
            }
        )

    return predicted, rows


def aggregate_candidate_articles(top_rows: list[dict], max_evidence_per_article: int = 2) -> list[dict]:
    candidates_by_id = {}
    for row in top_rows:
        for article_id in row.get("articles", []):
            candidate = candidates_by_id.setdefault(
                article_id,
                {
                    "article_id": article_id,
                    "label": article_id_to_display(article_id),
                    "best_rank": row["rank"],
                    "best_score": row["score"],
                    "evidence": [],
                },
            )
            if row["rank"] < candidate["best_rank"]:
                candidate["best_rank"] = row["rank"]
            if row["score"] > candidate["best_score"]:
                candidate["best_score"] = row["score"]
            if len(candidate["evidence"]) < max_evidence_per_article:
                candidate["evidence"].append(row["text"])

    return sorted(candidates_by_id.values(), key=lambda item: (item["best_rank"], -item["best_score"], item["article_id"]))


def build_article_selector_prompt(
    query: str,
    candidate_articles: list[dict],
    max_articles: int | None = None,
) -> str:
    limit_instruction = (
        f"Select at most {max_articles} article IDs. Prefer fewer articles when uncertain."
        if max_articles is not None
        else "Select all and only the candidate article IDs that are strongly supported. Prefer fewer articles when uncertain."
    )
    candidate_text = []
    for candidate in candidate_articles:
        evidence = "\n".join(f"    Evidence {idx + 1}: {text}" for idx, text in enumerate(candidate["evidence"]))
        candidate_text.append(
            f"- {candidate['article_id']} ({candidate['label']}) | "
            f"best_rank={candidate['best_rank']} | best_score={candidate['best_score']:.3f}\n{evidence}"
        )

    return f"""You are doing the final ECtHR alleged-violation article selection after a LATTICE retrieval step.

Choose only articles that are likely to be alleged violations for this case. Do not choose articles that are merely background, procedural context, or weakly related. {limit_instruction}

Return JSON with:
- reasoning: a brief explanation
- selected_articles: normalized IDs copied exactly from the candidate list

Case facts:
{case_facts_from_query(query)}

Candidate articles from LATTICE:
{chr(10).join(candidate_text)}
"""


def parse_selector_response(output: str, allowed_articles: set[str]) -> dict:
    from utils import post_process

    try:
        parsed = post_process(output, return_json=True)
    except Exception as exc:
        parsed = {"reasoning": f"Selector response could not be parsed: {exc}", "selected_articles": []}

    selected = []
    for article in (parsed or {}).get("selected_articles", []):
        normalized = normalize_article_label(article)
        if normalized in allowed_articles and normalized not in selected:
            selected.append(normalized)
    return {"reasoning": (parsed or {}).get("reasoning", ""), "selected_articles": selected}


def score_prediction(gold: set[str], predicted: set[str]) -> dict:
    """Score one multi-label case."""
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "any_gold_found": bool(gold & predicted),
        "all_gold_found": bool(gold) and gold.issubset(predicted),
        "exact_set_match": bool(gold) and gold == predicted,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_ecthr_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row summary DataFrame for ECtHR evaluation results."""
    import pandas as pd

    if df.empty:
        return pd.DataFrame(
            [
                {
                    "cases_evaluated": 0,
                    "any_gold_found": 0.0,
                    "all_gold_found": 0.0,
                    "exact_set_match": 0.0,
                    "mean_gold_removed_by_selector": 0.0,
                    "mean_recall": 0.0,
                    "mean_precision": 0.0,
                    "mean_f1": 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "cases_evaluated": len(df),
                "any_gold_found": int(df["any_gold_found"].sum()) / len(df),
                "all_gold_found": int(df["all_gold_found"].sum()) / len(df),
                "exact_set_match": int(df["exact_set_match"].sum()) / len(df),
                "mean_gold_removed_by_selector": df["gold_removed_by_selector"].mean()
                if "gold_removed_by_selector" in df
                else 0.0,
                "mean_recall": df["recall"].mean(),
                "mean_precision": df["precision"].mean(),
                "mean_f1": df["f1"].mean(),
            }
        ]
    )


@dataclass
class EcthrTraversalEvaluator:
    """Reusable ECtHR evaluator for LATTICE traversal notebooks."""

    semantic_root_node: Any
    node_registry: list[Any]
    hp: Any
    logger: Any
    llm_api: Any
    llm_api_kwargs: dict[str, Any]

    def make_sample(self, query: str) -> InferSample:
        from tree_objects import InferSample

        return InferSample(
            self.semantic_root_node,
            self.node_registry,
            hp=self.hp,
            logger=self.logger,
            query=query,
            gold_paths=[],
            excluded_ids_set=set(),
        )

    async def run_lattice_iterations_for_samples_async(
        self,
        samples: list[InferSample],
        *,
        num_iters: int,
        detailed_logs: bool = False,
    ) -> None:
        """Run traversal iterations for many samples using one shared async LLM batch per step."""
        for step in range(num_iters):
            print(f"\n--- Batched iteration {step + 1}/{num_iters} ({len(samples)} cases) ---")
            inputs_by_sample = [sample.get_step_prompts() for sample in samples]
            counts = [len(inputs) for inputs in inputs_by_sample]
            flat_inputs = [item for inputs in inputs_by_sample for item in inputs]
            if not flat_inputs:
                print("No prompts left to process.")
                break

            flat_prompts = [prompt for prompt, _ in flat_inputs]
            flat_slates = [slate for _, slate in flat_inputs]
            raw_responses = await self.llm_api.run_batch(flat_prompts, **self.llm_api_kwargs)
            from utils import post_process

            flat_response_jsons = [post_process(output, return_json=True) for output in raw_responses]

            offset = 0
            for sample, count in zip(samples, counts):
                sample_slates = flat_slates[offset : offset + count]
                sample_response_jsons = flat_response_jsons[offset : offset + count]
                if count:
                    sample.update(sample_slates, sample_response_jsons)
                    if detailed_logs:
                        self.print_frontier(sample)
                offset += count

    async def select_articles_with_llm_batch_async(
        self,
        queries: list[str],
        top_rows_list: list[list[dict]],
        max_articles: int | None = None,
    ) -> list[dict]:
        candidate_lists = [aggregate_candidate_articles(top_rows) for top_rows in top_rows_list]
        prompts = []
        prompt_indexes = []
        results = []

        for idx, (query, candidate_articles) in enumerate(zip(queries, candidate_lists)):
            allowed_articles = {candidate["article_id"] for candidate in candidate_articles}
            results.append(
                {
                    "reasoning": "No candidate articles were available for selector.",
                    "selected_articles": [],
                    "candidate_articles": sorted(allowed_articles),
                }
            )
            if candidate_articles:
                prompts.append(build_article_selector_prompt(query, candidate_articles, max_articles=max_articles))
                prompt_indexes.append(idx)

        if not prompts:
            return results

        selector_kwargs = {
            **self.llm_api_kwargs,
            "response_schema": ARTICLE_SELECTOR_RESPONSE_SCHEMA,
            "print_summary_report": False,
        }
        raw_outputs = await self.llm_api.run_batch(prompts, **selector_kwargs)
        for result_idx, output in zip(prompt_indexes, raw_outputs):
            allowed_articles = set(results[result_idx]["candidate_articles"])
            parsed = parse_selector_response(output, allowed_articles)
            parsed["candidate_articles"] = results[result_idx]["candidate_articles"]
            results[result_idx] = parsed
        return results

    async def evaluate_ecthr_cases_async(
        self,
        dataset,
        label_names: list[str] | None,
        *,
        n_cases: int = 10,
        num_iters: int = 6,
        top_k: int = 10,
        start: int = 0,
        detailed_logs: bool = False,
        prediction_min_score: float | None = None,
        max_predicted_articles: int | None = None,
        use_llm_selector: bool = False,
        print_cases: bool = True,
        print_summary: bool = True,
        ) -> tuple[pd.DataFrame, list[dict]]:
        """Evaluate many ECtHR cases by batching all LATTICE prompts across cases at each iteration."""
        import pandas as pd

        selected = list(dataset.select(range(start, min(start + n_cases, len(dataset)))))
        samples = []
        queries = []

        for example in selected:
            facts = example.get("text") or example.get("facts") or []
            query = facts_to_case_prompt(facts)
            queries.append(query)
            samples.append(self.make_sample(query))

        await self.run_lattice_iterations_for_samples_async(
            samples,
            num_iters=num_iters,
            detailed_logs=detailed_logs,
        )

        lattice_predictions = []
        top_rows_list = []
        for sample in samples:
            predicted, top_rows = predicted_articles_from_sample(
                sample,
                k=top_k,
                min_score=prediction_min_score,
                max_articles=max_predicted_articles,
            )
            lattice_predictions.append(predicted)
            top_rows_list.append(top_rows)

        selector_results = [None] * len(samples)
        if use_llm_selector:
            selector_results = await self.select_articles_with_llm_batch_async(
                queries,
                top_rows_list,
                max_articles=max_predicted_articles,
            )

        results = []
        for local_idx, (example, query, sample, lattice_predicted, top_rows, selector_result) in enumerate(
            zip(selected, queries, samples, lattice_predictions, top_rows_list, selector_results)
        ):
            case_idx = start + local_idx
            predicted = set(selector_result["selected_articles"]) if selector_result else lattice_predicted
            gold = example_gold_articles(example, label_names)
            gold_removed_by_selector_articles = sorted((gold & lattice_predicted) - predicted) if selector_result else []
            metrics = score_prediction(gold, predicted)
            result = {
                "case_index": case_idx,
                "query": query,
                "gold": sorted(gold),
                "predicted": sorted(predicted),
                "lattice_predicted": sorted(lattice_predicted),
                "gold_removed_by_selector": len(gold_removed_by_selector_articles),
                "gold_removed_by_selector_articles": gold_removed_by_selector_articles,
                "top_rows": top_rows,
                "selector_result": selector_result,
                "sample": sample,
                **metrics,
            }
            results.append(result)

            if print_cases:
                print(f"\n================ ECtHR case {case_idx} ================")
                print("Gold:     ", result["gold"])
                print("Predicted: ", result["predicted"])
                print(
                    f"Correct(all gold found) : {result['all_gold_found']} | "
                    f"Recall: {result['recall']:.2f} | Precision: {result['precision']:.2f} | F1: {result['f1']:.2f}"
                )
                if selector_result:
                    print(
                        "Gold articles removed by selector: "
                        f"{result['gold_removed_by_selector']} {result['gold_removed_by_selector_articles']}"
                    )

        df = pd.DataFrame(
            [
                {
                    "case_index": result["case_index"],
                    "gold": result["gold"],
                    "predicted": result["predicted"],
                    "lattice_predicted": result.get("lattice_predicted", result["predicted"]),
                    "gold_removed_by_selector": result.get("gold_removed_by_selector", 0),
                    "gold_removed_by_selector_articles": result.get("gold_removed_by_selector_articles", []),
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

        if print_summary:
            print("\n================ Summary ================")
            print(summarize_ecthr_cases(df).to_string(index=False))
        return df, results

    def evaluate_ecthr_cases_batched(self, *args, **kwargs) -> tuple[pd.DataFrame, list[dict]]:
        from tree_construction.build_llm_bottom_up_tree import run_coro_sync

        return run_coro_sync(self.evaluate_ecthr_cases_async(*args, **kwargs))

    def run_ecthr_case(
        self,
        example: dict,
        label_names: list[str] | None,
        *,
        num_iters: int = 6,
        top_k: int = 10,
        prediction_min_score: float | None = None,
        max_predicted_articles: int | None = None,
        use_llm_selector: bool = False,
    ) -> dict:
        facts = example.get("text") or example.get("facts") or []
        query = facts_to_case_prompt(facts)
        sample = self.run_query(query, num_iters=num_iters)
        predicted, top_rows = predicted_articles_from_sample(
            sample,
            k=top_k,
            min_score=prediction_min_score,
            max_articles=max_predicted_articles,
        )
        lattice_predicted = set(predicted)
        selector_result = None

        if use_llm_selector:
            from tree_construction.build_llm_bottom_up_tree import run_coro_sync

            selector_result = run_coro_sync(
                self.select_articles_with_llm_batch_async([query], [top_rows], max_articles=max_predicted_articles)
            )[0]
            predicted = set(selector_result["selected_articles"])

        gold = example_gold_articles(example, label_names)
        gold_removed_by_selector_articles = sorted((gold & lattice_predicted) - predicted) if selector_result else []
        metrics = score_prediction(gold, predicted)
        return {
            "query": query,
            "gold": sorted(gold),
            "predicted": sorted(predicted),
            "lattice_predicted": sorted(lattice_predicted),
            "gold_removed_by_selector": len(gold_removed_by_selector_articles),
            "gold_removed_by_selector_articles": gold_removed_by_selector_articles,
            "top_rows": top_rows,
            "selector_result": selector_result,
            "sample": sample,
            **metrics,
        }

    async def run_query_async(self, query: str, num_iters: int = 4, detailed_logs: bool = False) -> InferSample:
        sample = self.make_sample(query)
        if detailed_logs:
            print(f"Running traversal for query: {query}")

        for step in range(num_iters):
            print(f"\n--- Iteration {step + 1} ---")
            inputs = sample.get_step_prompts()
            prompts = [prompt for prompt, _ in inputs]
            slates = [slate for _, slate in inputs]
            raw_responses = await self.llm_api.run_batch(prompts, **self.llm_api_kwargs)
            from utils import post_process

            response_jsons = [post_process(output, return_json=True) for output in raw_responses]
            sample.update(slates, response_jsons)
            if detailed_logs:
                self.print_frontier(sample)
                if response_jsons:
                    print("\nReasoning preview:")
                    print(str(response_jsons[0].get("reasoning", ""))[:800])
        return sample

    def run_query(self, query: str, num_iters: int = 4, detailed_logs: bool = False) -> InferSample:
        from tree_construction.build_llm_bottom_up_tree import run_coro_sync

        return run_coro_sync(self.run_query_async(query, num_iters=num_iters, detailed_logs=detailed_logs))

    @staticmethod
    def print_frontier(sample: InferSample) -> None:
        print("Current beam state paths:")
        for state_path in sample.beam_state_paths:
            node = state_path[-1]
            print(
                f"  path={node.path} | path_relevance={node.path_relevance:.3f} | desc={node.desc[:180]}"
            )

    @staticmethod
    def print_top_predictions(sample: InferSample, k: int = 5) -> None:
        top_preds = sample.get_top_predictions(k=k)
        if not top_preds:
            print("No leaf predictions yet. Increase num_iters.")
            return

        for rank, (node, score) in enumerate(top_preds, start=1):
            print(f"[{rank}] path={node.path} | score={score:.3f} | id={node.id}")
            print(node.desc[:600])
            print("-" * 120)
