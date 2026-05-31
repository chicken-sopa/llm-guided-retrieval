from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    from llm_rl_playground.ecthr_evaluation import (
        TREE_ARTICLE_RE,
        article_id_to_display,
        extract_articles_from_tree_text,
        normalize_article_label,
    )
except ModuleNotFoundError:
    from ecthr_evaluation import (
        TREE_ARTICLE_RE,
        article_id_to_display,
        extract_articles_from_tree_text,
        normalize_article_label,
    )


def clean_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_json_tree(tree_path: str | Path) -> dict:
    with Path(tree_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def example_gold_articles(example: dict, label_column: str = "labels") -> list[str]:
    """Convert one ECtHR example's labels to sorted normalized article IDs."""
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


def facts_to_text(example: dict, max_chars: int = 9000) -> str:
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


def annotate_subtree_articles(node: dict) -> set[str]:
    """Annotate every semantic-tree node with descendant ECtHR article IDs."""
    children = node.get("child") or []
    if not children:
        articles = extract_articles_from_tree_text(node.get("desc", ""))
    else:
        articles = set()
        for child in children:
            articles |= annotate_subtree_articles(child)
    node["_subtree_articles"] = sorted(articles)
    return articles


def format_child_options(children: list[dict], max_child_desc_chars: int = 1100) -> str:
    parts = []
    for idx, child in enumerate(children):
        desc = clean_space(child.get("desc"))[:max_child_desc_chars]
        parts.append(f"[{idx}]. {desc}")
    return "\n\n".join(parts)


def format_current_path(path_labels: list[str], max_path_chars: int = 1800) -> str:
    text = " -> ".join(path_labels)
    if len(text) > max_path_chars:
        text = text[-max_path_chars:]
    return text or "ROOT"


def build_traversal_prompt(
    facts: str,
    path_labels: list[str],
    children: list[dict],
    *,
    max_child_desc_chars: int = 1100,
    max_path_chars: int = 1800,
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
    include_single_child_nodes: bool = False,
    max_child_desc_chars: int = 1100,
    max_path_chars: int = 1800,
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


def make_examples_for_case(
    example: dict,
    tree: dict,
    all_tree_articles: set[str],
    rng: random.Random,
    *,
    max_fact_chars: int = 9000,
    include_single_child_nodes: bool = False,
    max_child_desc_chars: int = 1100,
    max_path_chars: int = 1800,
    max_examples_per_case: int = 8,
) -> list[dict]:
    gold_articles = set(example_gold_articles(example))
    gold_articles &= set(all_tree_articles)
    if not gold_articles:
        return []

    rows = []
    collect_case_traversal_examples(
        tree,
        facts=facts_to_text(example, max_fact_chars),
        gold_articles=gold_articles,
        path_labels=["ROOT"],
        rows=rows,
        include_single_child_nodes=include_single_child_nodes,
        max_child_desc_chars=max_child_desc_chars,
        max_path_chars=max_path_chars,
    )
    rng.shuffle(rows)
    return rows[:max_examples_per_case]


def build_traversal_rows(
    dataset,
    tree: dict,
    all_tree_articles: set[str],
    *,
    max_cases: int,
    max_rows: int,
    seed: int,
    max_fact_chars: int = 9000,
    include_single_child_nodes: bool = False,
    max_child_desc_chars: int = 1100,
    max_path_chars: int = 1800,
    max_examples_per_case: int = 8,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    indexes = list(range(len(dataset)))
    rng.shuffle(indexes)

    rows = []
    cases_used = 0
    skipped_cases = 0
    for idx in indexes[:max_cases]:
        print(f"current new Test case being created {idx}")
        case_rows = make_examples_for_case(
            dataset[idx],
            tree,
            all_tree_articles,
            rng,
            max_fact_chars=max_fact_chars,
            include_single_child_nodes=include_single_child_nodes,
            max_child_desc_chars=max_child_desc_chars,
            max_path_chars=max_path_chars,
            max_examples_per_case=max_examples_per_case,
        )
        if not case_rows:
            skipped_cases += 1
            continue
        rows.extend(case_rows)
        cases_used += 1
        if len(rows) >= max_rows:
            print(f"no more cases added {cases_used + skipped_cases}")
            rows = rows[:max_rows]
            break
    return rows, {"cases_used": cases_used, "skipped_cases": skipped_cases, "rows": len(rows)}


def article_id_from_chunk(chunk: dict) -> str | None:
    text = " ".join(
        clean_space(chunk.get(key))
        for key in ("citation", "article_number", "article_title", "text")
    )
    match = TREE_ARTICLE_RE.search(text)
    if not match:
        return normalize_article_label(chunk.get("citation") or chunk.get("article_number"))
    article = match.group("article").lower()
    protocol = match.group("protocol")
    if protocol:
        return f"protocol_{int(protocol)}_article_{article}"
    return f"article_{article}"


def load_article_chunk_pool(path: str | Path) -> dict[str, list[dict]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", payload) if isinstance(payload, dict) else payload
    pool: dict[str, list[dict]] = {}
    for chunk in chunks:
        text = clean_space(chunk.get("text"))
        if len(text) < 40:
            continue
        article_id = article_id_from_chunk(chunk)
        if not article_id:
            continue
        pool.setdefault(article_id, []).append(
            {
                "article_id": article_id,
                "label": article_id_to_display(article_id),
                "citation": clean_space(chunk.get("citation")),
                "text": text,
            }
        )
    return pool


def make_direct_article_prediction_example(example: dict, *, max_fact_chars: int = 9000) -> dict | None:
    gold = example_gold_articles(example)
    if not gold:
        return None

    user = f"""Given the facts of an ECtHR case, predict the Convention or Protocol articles alleged by the applicant.

Return only JSON with this shape:
{{"reasoning": "brief explanation", "selected_articles": ["article_..."]}}

Case facts:
{facts_to_text(example, max_fact_chars)}"""

    assistant = json.dumps(
        {
            "reasoning": "The selected articles are the alleged Convention or Protocol provisions associated with the case facts.",
            "selected_articles": gold,
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": "You predict alleged ECtHR article IDs from case facts. Return only valid JSON."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def make_selector_article_prediction_example(
    example: dict,
    article_pool: dict[str, list[dict]],
    rng: random.Random,
    *,
    max_fact_chars: int = 9000,
    max_evidence_chars: int = 900,
    num_negative_articles: int = 8,
) -> dict | None:
    gold_all = example_gold_articles(example)
    gold_with_chunks = [article_id for article_id in gold_all if article_id in article_pool]
    if not gold_with_chunks:
        return None

    candidate_ids = list(gold_with_chunks)
    negative_ids = [article_id for article_id in article_pool if article_id not in set(gold_all)]
    rng.shuffle(negative_ids)
    candidate_ids.extend(negative_ids[:num_negative_articles])
    rng.shuffle(candidate_ids)

    candidate_lines = []
    for article_id in candidate_ids:
        chunk = rng.choice(article_pool[article_id])
        candidate_lines.append(
            f"- {article_id} ({chunk['label']})\n"
            f"  Evidence: {chunk['text'][:max_evidence_chars]}"
        )

    user = f"""You are doing the final ECtHR alleged-violation article selection after a LATTICE retrieval step.

Choose only candidate article IDs that are likely to be alleged violations for this case. Do not choose articles that are merely background, procedural context, or weakly related.

Return JSON with:
- reasoning: a brief explanation
- selected_articles: normalized IDs copied exactly from the candidate list

Case facts:
{facts_to_text(example, max_fact_chars)}

Candidate articles from LATTICE:
{chr(10).join(candidate_lines)}"""

    selected = sorted(article_id for article_id in candidate_ids if article_id in set(gold_all))
    assistant = json.dumps(
        {
            "reasoning": "The selected candidate article IDs correspond to the alleged Convention or Protocol provisions in the case labels.",
            "selected_articles": selected,
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": "You are a local legal retrieval selector. Return only valid JSON."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_article_prediction_rows(
    dataset,
    *,
    max_examples: int,
    seed: int,
    mode: str = "selector",
    article_pool: dict[str, list[dict]] | None = None,
    max_fact_chars: int = 9000,
    max_evidence_chars: int = 900,
    num_negative_articles: int = 8,
) -> list[dict]:
    if mode not in {"selector", "direct"}:
        raise ValueError("mode must be 'selector' or 'direct'")

    rng = random.Random(seed)
    indexes = list(range(len(dataset)))
    rng.shuffle(indexes)

    rows = []
    for idx in indexes:
        example = dataset[idx]
        if mode == "selector":
            row = make_selector_article_prediction_example(
                example,
                article_pool or {},
                rng,
                max_fact_chars=max_fact_chars,
                max_evidence_chars=max_evidence_chars,
                num_negative_articles=num_negative_articles,
            )
        else:
            row = make_direct_article_prediction_example(example, max_fact_chars=max_fact_chars)
        if row is not None:
            rows.append(row)
        if len(rows) >= max_examples:
            break
    return rows
