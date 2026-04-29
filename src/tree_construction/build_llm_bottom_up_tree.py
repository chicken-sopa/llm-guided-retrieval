"""Build a bottom-up semantic tree with embedding-based clustering.

This builder follows the same high-level pattern described in the LATTICE paper:

1. Start with leaf passages.
2. Embed the current level of nodes.
3. Cluster nodes into small groups.
4. Ask an LLM to summarize each cluster into a parent node.
5. Repeat until a single root remains.

The output is a plain nested dict that can be loaded by
``tree_objects.SemanticNode.load_dict``. Keeping the on-disk format aligned with
the retrieval code lets us swap in the new tree without changing traversal.

Example usage with live APIs:
    python src/tree_construction/build_llm_bottom_up_tree.py \
      --subset biology \
      --embedding-field embs \
      --llm-api-backend openai \
      --llm gpt-4.1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from datasets import load_dataset
from json_repair import repair_json
from sklearn.cluster import AgglomerativeClustering

# The script lives in src/tree_construction, so we explicitly add src/ to the
# import path before importing sibling modules.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_apis import GenAIAPI, OpenAIResponsesAPI, VllmAPI
from utils import setup_logger

SUMMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "A short retrieval-oriented label for the cluster.",
        },
        "summary": {
            "type": "string",
            "description": (
                "A concise description of what information lives in this subtree, "
                "written to help route retrieval queries."
            ),
        },
        "key_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A short list of specific topics, entities, or tasks.",
            "minItems": 1,
        },
    },
    "required": ["label", "summary", "key_topics"],
}


def clean_text(text: Any) -> str:
    """Collapse whitespace so clustering and prompts see consistent text."""
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slug(value: str, max_len: int = 64) -> str:
    """Convert a label into a filesystem-friendly, stable identifier piece."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "node")[:max_len]


def short_hash(value: str) -> str:
    """Return a short SHA1 fragment so node IDs remain stable but readable."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def mean_embedding(embeddings: Iterable[Sequence[float] | None]) -> list[float] | None:
    """Average a list of same-sized vectors, skipping missing values."""
    vectors = [np.asarray(emb, dtype=float) for emb in embeddings if emb is not None]
    if not vectors:
        return None
    size = vectors[0].shape[0]
    valid_vectors = [vec for vec in vectors if vec.shape[0] == size]
    if not valid_vectors:
        return None
    return np.mean(valid_vectors, axis=0).astype(float).tolist()


def normalize_vector(vector: Sequence[float] | None) -> list[float] | None:
    """L2-normalize embeddings so cosine-based clustering behaves predictably."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0:
        return arr.astype(float).tolist()
    return (arr / norm).astype(float).tolist()


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy-heavy objects into plain JSON-serializable data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def run_coro_sync(coro: Any) -> Any:
    """Run an async coroutine from either scripts or notebook cells.

    Jupyter already owns an event loop, so calling ``asyncio.run`` directly from
    a notebook raises ``RuntimeError``. In that case we run the coroutine in a
    short-lived background thread that gets its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _thread_main() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from common corpus file formats used in this repo."""
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("documents"), list):
            return payload["documents"]
        if isinstance(payload.get("chunks"), list):
            return payload["chunks"]
        return [payload]
    raise ValueError(f"Unsupported input payload type: {type(payload)}")


def infer_first_present(record: dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present field from a record."""
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def infer_record_id(record: dict[str, Any], index: int, id_field: str | None) -> str:
    """Pick a stable document identifier from common corpus field names."""
    if id_field and record.get(id_field) not in (None, ""):
        return str(record[id_field])

    candidate = infer_first_present(
        record,
        ("id", "doc_id", "chunk_id", "source_id", "article_uid", "citation"),
    )
    return str(candidate) if candidate not in (None, "") else f"doc-{index:06d}"


def infer_record_text(record: dict[str, Any], text_field: str | None) -> str:
    """Pick the primary text field used to build leaf descriptions."""
    if text_field and record.get(text_field):
        return clean_text(record[text_field])

    candidate = infer_first_present(record, ("content", "text", "passage", "body", "desc"))
    return clean_text(candidate)


def infer_record_embedding(
    record: dict[str, Any],
    embedding_field: str | None,
) -> list[float] | None:
    """Extract an existing embedding if the input record already provides one."""
    if embedding_field and record.get(embedding_field) is not None:
        embedding = record[embedding_field]
    else:
        embedding = infer_first_present(record, ("embedding", "embs", "vector"))

    if embedding is None:
        return None
    if isinstance(embedding, np.ndarray):
        return embedding.astype(float).tolist()
    if isinstance(embedding, list):
        return [float(x) for x in embedding]
    return None


@dataclass
class WorkingNode:
    """Small in-memory node object used while the tree is being constructed."""

    id: str
    desc: str
    child: list["WorkingNode"] = field(default_factory=list)
    embs: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    node_type: str = "semantic_cluster"
    leaf_count: int = 1
    sort_order: int = 0

    def to_tree_dict(self) -> dict[str, Any]:
        """Serialize the node into the plain dict format consumed by traversal."""
        node = {
            "id": self.id,
            "desc": self.desc,
            "embs": self.embs,
            "child": [child.to_tree_dict() for child in self.child] if self.child else None,
        }
        if self.node_type:
            node["node_type"] = self.node_type
        if self.metadata:
            node["metadata"] = to_jsonable(self.metadata)
        return node


class ClusterSummarizer:
    """Create cluster descriptions for parent nodes with an LLM."""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.summary_cache_path = Path(args.summary_cache) if args.summary_cache else None
        self.summary_cache = self._load_cache()
        self.llm_api, self.llm_api_kwargs = self._make_llm_api()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load cached cluster summaries so repeated builds reuse earlier work."""
        if not self.summary_cache_path or not self.summary_cache_path.exists():
            return {}
        return json.loads(self.summary_cache_path.read_text(encoding="utf-8"))

    def save_cache(self) -> None:
        """Persist cached summaries generated during this run."""
        if not self.summary_cache_path:
            return
        self.summary_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_cache_path.write_text(json.dumps(self.summary_cache), encoding="utf-8")

    def _make_llm_api(self) -> tuple[Any, dict[str, Any]]:
        """Reuse the repo's existing LLM wrappers so builder and traversal match."""
        if self.args.llm_api_backend == "genai":
            api = GenAIAPI(
                self.args.llm,
                logger=self.logger,
                timeout=self.args.llm_api_timeout,
                max_retries=self.args.llm_api_max_retries,
            )
        elif self.args.llm_api_backend == "openai":
            api = OpenAIResponsesAPI(
                self.args.llm,
                logger=self.logger,
                timeout=self.args.llm_api_timeout,
                max_retries=self.args.llm_api_max_retries,
            )
        elif self.args.llm_api_backend == "vllm":
            api = VllmAPI(
                self.args.llm,
                logger=self.logger,
                timeout=self.args.llm_api_timeout,
                max_retries=self.args.llm_api_max_retries,
                base_url=self.args.vllm_base_url,
            )
        else:
            raise ValueError(f"Unknown llm_api_backend: {self.args.llm_api_backend}")

        llm_api_kwargs: dict[str, Any] = {
            "max_concurrent_calls": self.args.llm_max_concurrent_calls,
            "response_schema": SUMMARY_RESPONSE_SCHEMA,
            "response_mime_type": "application/json",
            "staggering_delay": self.args.llm_api_staggering_delay,
            "print_summary_report": False,
            "temperature": 0,
        }
        return api, llm_api_kwargs

    def summarize_clusters(
        self,
        clusters: Sequence[Sequence[WorkingNode]],
        level_index: int,
        *,
        is_root: bool = False,
    ) -> list[dict[str, Any]]:
        """Summarize one construction layer worth of clusters."""
        prompts: list[str] = []
        cache_keys: list[str] = []
        results: list[dict[str, Any] | None] = [None] * len(clusters)

        for idx, cluster in enumerate(clusters):
            cache_key = self._cluster_cache_key(cluster, level_index, is_root=is_root)
            cache_keys.append(cache_key)
            cached = self.summary_cache.get(cache_key)
            if cached is not None:
                results[idx] = cached
                continue
            prompts.append(self._build_summary_prompt(cluster, level_index, is_root=is_root))

        uncached_indices = [idx for idx, value in enumerate(results) if value is None]
        if prompts:
            raw_responses = run_coro_sync(self.llm_api.run_batch(prompts, **self.llm_api_kwargs))
            for result_index, raw_response in zip(uncached_indices, raw_responses):
                cluster = clusters[result_index]
                payload = self._parse_llm_summary(raw_response, cluster, is_root=is_root)
                results[result_index] = payload
                self.summary_cache[cache_keys[result_index]] = payload
            self.save_cache()

        return [payload if payload is not None else {} for payload in results]

    def _cluster_cache_key(
        self,
        cluster: Sequence[WorkingNode],
        level_index: int,
        *,
        is_root: bool,
    ) -> str:
        """Cache cluster summaries by construction level and child IDs."""
        signature = "|".join(node.id for node in cluster)
        mode = f"{self.args.llm_api_backend}:{self.args.llm}"
        return f"{mode}:{level_index}:{int(is_root)}:{short_hash(signature)}"

    def _child_snippet(self, node: WorkingNode, child_index: int) -> str:
        """Format a single child description for the cluster summarization prompt."""
        desc = clean_text(node.desc)
        desc = desc[: self.args.prompt_child_char_limit]
        leaf_note = f"{node.leaf_count} leaf passages"
        return f"[{child_index}] id={node.id} ({leaf_note}) {desc}"

    def _build_summary_prompt(
        self,
        cluster: Sequence[WorkingNode],
        level_index: int,
        *,
        is_root: bool,
    ) -> str:
        """Build a routing-oriented prompt that turns children into one parent summary."""
        child_block = "\n\n".join(
            self._child_snippet(node, child_index)
            for child_index, node in enumerate(cluster)
        )
        role_note = "root node" if is_root else "parent node"
        return f"""You are building a bottom-up semantic retrieval tree.

Your task is to write the text for one {role_note}. The child items below are
either raw passages or summaries of lower-level clusters.

Write for retrieval navigation, not for literary summarization:
- describe what kinds of questions or information this subtree can answer
- mention concrete topics, entities, concepts, or tasks when possible
- avoid vague phrases like "miscellaneous information"
- keep the summary compact but specific

Return JSON with these keys:
- "label": a short label (3-8 words)
- "summary": a concise routing-friendly description
- "key_topics": 3-8 short topic strings

Construction level: {level_index}
Number of children: {len(cluster)}

Child nodes:
{child_block}
"""

    def _parse_llm_summary(
        self,
        raw_response: str,
        cluster: Sequence[WorkingNode],
        *,
        is_root: bool,
    ) -> dict[str, Any]:
        """Parse the JSON summary returned by the LLM."""
        try:
            if isinstance(raw_response, str) and raw_response.startswith("Error:"):
                raise ValueError(raw_response)
            payload = repair_json(raw_response, return_objects=True)
            label = clean_text(payload.get("label", "")) or "Semantic cluster"
            summary = clean_text(payload.get("summary", "")) or label
            key_topics = [
                clean_text(topic)
                for topic in payload.get("key_topics", [])
                if clean_text(topic)
            ]
            if not key_topics:
                raise ValueError("LLM summary omitted key_topics")
            return {
                "label": label,
                "summary": summary,
                "key_topics": key_topics[:8],
                "generator": "llm",
                "is_root": is_root,
            }
        except Exception as exc:
            cluster_ids = ", ".join(node.id for node in cluster[:5])
            raise ValueError(f"Failed to parse summary for cluster [{cluster_ids}]: {exc}") from exc


def load_source_records(args: argparse.Namespace, logger: logging.Logger) -> list[dict[str, Any]]:
    """Load a corpus either from disk or from the BRIGHT dataset."""
    if args.input:
        input_path = Path(args.input)
        logger.info(f"Loading records from {input_path}")
        records = read_json_or_jsonl(input_path)
    else:
        if not args.subset:
            raise ValueError("--subset is required when --input is not provided")

        local_docs_path = SRC_DIR.parent / "data" / args.dataset / args.subset / "documents.jsonl"
        if local_docs_path.exists():
            logger.info(f"Loading local dataset file {local_docs_path}")
            records = read_json_or_jsonl(local_docs_path)
        else:
            logger.info(f"Loading dataset split {args.dataset}/{args.subset} via datasets.load_dataset")
            if args.dataset.upper() != "BRIGHT":
                raise ValueError("Automatic dataset loading is only implemented for BRIGHT.")
            records = list(load_dataset("xlangai/BRIGHT", "documents", split=args.subset))

    if args.max_leaves is not None:
        records = records[: args.max_leaves]
    return records


def build_leaf_nodes(
    records: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> list[WorkingNode]:
    """Convert raw records into the leaf nodes that seed the bottom-up build."""
    leaf_nodes: list[WorkingNode] = []

    for index, record in enumerate(records):
        text = infer_record_text(record, args.text_field)
        if not text:
            continue

        node_id = infer_record_id(record, index, args.id_field)
        existing_embedding = infer_record_embedding(record, args.embedding_field)
        if existing_embedding is None:
            raise ValueError(
                f"Record {index} ({node_id}) is missing an embedding. "
                "This builder assumes embeddings are already present in the input."
            )
        normalized_embedding = normalize_vector(existing_embedding)

        leaf_nodes.append(
            WorkingNode(
                id=str(node_id),
                desc=text,
                child=[],
                embs=normalized_embedding,
                metadata={
                    "source_id": str(node_id),
                    "record_index": index,
                },
                node_type="leaf",
                leaf_count=1,
                sort_order=index,
            )
        )

    logger.info(f"Prepared {len(leaf_nodes)} leaf nodes")
    return leaf_nodes


def cluster_labels_for_matrix(
    matrix: np.ndarray,
    n_clusters: int,
    linkage: str,
) -> np.ndarray:
    """Run one clustering step and return a label per row."""
    # This helper does exactly one non-recursive clustering pass:
    # given N rows, it assigns each row to one of `n_clusters` groups.
    #
    # recursive_cluster_indices relies on this function behaving like a
    # "single split" primitive. It does not need this helper to fully solve the
    # max_children constraint in one shot; it only needs a coarse partition that
    # it can inspect and, if necessary, split again recursively.
    if n_clusters <= 1:
        return np.zeros(matrix.shape[0], dtype=int)
    if n_clusters >= matrix.shape[0]:
        return np.arange(matrix.shape[0], dtype=int)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage=linkage,
    )
    return model.fit_predict(matrix)


def recursive_cluster_indices(
    matrix: np.ndarray,
    max_children: int,
    linkage: str,
    base_indices: list[int] | None = None,
) -> list[list[int]]:
    """Recursively split oversized clusters until every group fits the fanout cap."""
    # `base_indices` keeps track of which rows in the original matrix this local
    # subproblem corresponds to. The recursion repeatedly zooms into one
    # oversized cluster, but the caller ultimately needs indices that still point
    # back to the original list of nodes.
    if base_indices is None:
        base_indices = list(range(matrix.shape[0]))

    # Base case:
    # once the current subproblem already fits under max_children, recursion
    # stops and we return this group as one final sibling cluster.
    if len(base_indices) <= max_children:
        return [sorted(base_indices)]

    # We ask for just enough groups so that, in the ideal balanced case, each
    # group would fit under max_children. Some groups may still come back too
    # large, which is why this function is recursive instead of a single call.
    target_clusters = math.ceil(len(base_indices) / max_children)
    local_labels = cluster_labels_for_matrix(matrix, target_clusters, linkage=linkage)
    grouped_local_indices: dict[int, list[int]] = {}
    for local_index, label in enumerate(local_labels):
        grouped_local_indices.setdefault(int(label), []).append(local_index)

    final_groups: list[list[int]] = []
    for _, local_group in sorted(grouped_local_indices.items(), key=lambda item: min(item[1])):
        # `local_group` contains row numbers relative to the *current* matrix.
        # We remap them back to original node indices before returning anything.
        mapped_group = [base_indices[local_index] for local_index in local_group]
        if len(mapped_group) <= max_children:
            final_groups.append(sorted(mapped_group))
            continue

        # Recursive step:
        # if one of the coarse clusters is still too large, we extract just that
        # submatrix and run the same logic again on the smaller problem.
        #
        # This is the key idea of the function: keep splitting only the parts
        # that violate the fanout cap, and leave already-good groups untouched.
        child_matrix = matrix[np.array(local_group)]
        final_groups.extend(
            recursive_cluster_indices(
                child_matrix,
                max_children=max_children,
                linkage=linkage,
                base_indices=mapped_group,
            )
        )
    return final_groups


def cluster_current_level(
    nodes: Sequence[WorkingNode],
    max_children: int,
    linkage: str,
) -> list[list[WorkingNode]]:
    """Group one tree level into small semantic clusters."""
    if len(nodes) <= max_children:
        return [list(nodes)]

    # recursive_cluster_indices works with matrix row indices, so this wrapper is
    # responsible for translating between "rows in the embedding matrix" and the
    # actual WorkingNode objects that will become tree children.
    matrix = np.asarray([node.embs for node in nodes], dtype=float)
    clusters_as_indices = recursive_cluster_indices(matrix, max_children=max_children, linkage=linkage)

    clusters = []
    for index_group in clusters_as_indices:
        cluster = [nodes[index] for index in sorted(index_group, key=lambda idx: nodes[idx].sort_order)]
        clusters.append(cluster)

    # Stable ordering keeps output paths reproducible across reruns.
    clusters.sort(key=lambda cluster: min(node.sort_order for node in cluster))
    return clusters


def compose_node_description(payload: dict[str, Any], children: Sequence[WorkingNode], *, is_root: bool) -> str:
    """Turn the structured summary payload into the string seen during traversal."""
    label = clean_text(payload.get("label", "Semantic cluster"))
    summary = clean_text(payload.get("summary", label))
    key_topics = [clean_text(topic) for topic in payload.get("key_topics", []) if clean_text(topic)]
    child_count = len(children)
    leaf_count = sum(child.leaf_count for child in children)

    parts = [summary]
    if key_topics:
        parts.append("Key topics: " + "; ".join(key_topics[:8]))
    parts.append(f"{child_count} direct children")
    parts.append(f"{leaf_count} leaf passages")

    body = ". ".join(parts)
    return f"ROOT Node: {body}" if is_root else f"{label}. {body}"


def make_parent_nodes(
    clusters: Sequence[Sequence[WorkingNode]],
    summaries: Sequence[dict[str, Any]],
    level_index: int,
) -> list[WorkingNode]:
    """Convert clustered children plus their summaries into parent nodes."""
    parent_nodes: list[WorkingNode] = []

    for cluster_index, (cluster, payload) in enumerate(zip(clusters, summaries)):
        signature = "|".join(node.id for node in cluster)
        label = clean_text(payload.get("label", "Semantic cluster"))
        desc = compose_node_description(payload, cluster, is_root=False)
        parent_nodes.append(
            WorkingNode(
                id=f"level:{level_index}:cluster:{cluster_index:04d}:{short_hash(signature)}",
                desc=desc,
                child=list(cluster),
                embs=None,
                metadata={
                    "label": label,
                    "level_index": level_index,
                    "summary_payload": payload,
                },
                node_type="semantic_cluster",
                leaf_count=sum(node.leaf_count for node in cluster),
                sort_order=min(node.sort_order for node in cluster),
            )
        )

    for node in parent_nodes:
        # Parent embeddings are always the mean of child embeddings because this
        # builder assumes the input already carries the only embeddings we have.
        node.embs = normalize_vector(mean_embedding(child.embs for child in node.child))

    return parent_nodes


def make_root_node(
    children: Sequence[WorkingNode],
    payload: dict[str, Any],
) -> WorkingNode:
    """Create the final root wrapper so traversal always starts from an internal node."""
    root = WorkingNode(
        id="root",
        desc=compose_node_description(payload, children, is_root=True),
        child=list(children),
        embs=None,
        metadata={
            "label": clean_text(payload.get("label", "Root")),
            "summary_payload": payload,
        },
        node_type="root",
        leaf_count=sum(node.leaf_count for node in children),
        sort_order=min(node.sort_order for node in children) if children else 0,
    )

    root.embs = normalize_vector(mean_embedding(child.embs for child in children))
    return root


def build_bottom_up_tree(
    leaf_nodes: Sequence[WorkingNode],
    args: argparse.Namespace,
    summarizer: ClusterSummarizer,
    logger: logging.Logger,
) -> WorkingNode:
    """Iteratively cluster and summarize until only one root remains."""
    if not leaf_nodes:
        raise ValueError("Cannot build a tree from zero leaf nodes.")

    current_nodes = sorted(leaf_nodes, key=lambda node: node.sort_order)
    level_index = 0

    while len(current_nodes) > args.max_children:
        logger.info(f"Building parent level {level_index} from {len(current_nodes)} nodes")
        clusters = cluster_current_level(
            current_nodes,
            max_children=args.max_children,
            linkage=args.cluster_linkage,
        )
        summaries = summarizer.summarize_clusters(clusters, level_index)
        current_nodes = make_parent_nodes(clusters, summaries, level_index)
        logger.info(
            f"Level {level_index} produced {len(current_nodes)} parent nodes "
            f"(max fanout {max(len(cluster) for cluster in clusters)})"
        )
        level_index += 1

    logger.info(f"Building root from {len(current_nodes)} top-level nodes")
    [root_summary] = summarizer.summarize_clusters([current_nodes], level_index, is_root=True)
    return make_root_node(current_nodes, root_summary)


def count_nodes(node: dict[str, Any]) -> int:
    """Count nodes in the serialized tree for quick sanity checks."""
    return 1 + sum(count_nodes(child) for child in node.get("child") or [])


def count_leaves(node: dict[str, Any]) -> int:
    """Count leaves in the serialized tree for quick sanity checks."""
    children = node.get("child") or []
    if not children:
        return 1
    return sum(count_leaves(child) for child in children)


def max_branching(node: dict[str, Any]) -> int:
    """Measure the widest node so we can verify the fanout cap held."""
    children = node.get("child") or []
    return max([len(children), *(max_branching(child) for child in children)] or [0])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the bottom-up builder."""
    parser = argparse.ArgumentParser(description="Build an LLM-authored bottom-up semantic tree.")
    parser.add_argument("--input", help="Optional path to a local .json or .jsonl corpus file.")
    parser.add_argument("--dataset", default="BRIGHT", help="Dataset name used for defaults.")
    parser.add_argument("--subset", help="Corpus subset name, used for loading and output paths.")
    parser.add_argument("--output", help="Output pickle path. Defaults to trees/<dataset>/<subset>/tree-bottom-up-llm.pkl")
    parser.add_argument("--json-output", help="Optional JSON copy of the produced tree.")
    parser.add_argument("--log-file", help="Optional log file path.")

    parser.add_argument("--id-field", help="Override the input field used as the leaf node ID.")
    parser.add_argument("--text-field", help="Override the input field used as the leaf node text.")
    parser.add_argument("--embedding-field", help="Override the input field used for existing embeddings.")
    parser.add_argument("--max-leaves", type=int, help="Optional cap for smaller runs.")

    parser.add_argument("--max-children", type=int, default=10, help="Maximum direct children for any internal node.")
    parser.add_argument(
        "--cluster-linkage",
        choices=["average", "complete", "single"],
        default="average",
        help="Agglomerative clustering linkage used when grouping siblings.",
    )
    parser.add_argument(
        "--prompt-child-char-limit",
        type=int,
        default=700,
        help="Maximum child description length included in summarization prompts.",
    )

    parser.add_argument("--summary-cache", help="Optional JSON cache path for cluster summaries.")
    parser.add_argument(
        "--llm-api-backend",
        choices=["openai", "genai", "vllm"],
        default="openai",
        help="LLM backend used for cluster summaries.",
    )
    parser.add_argument("--llm", default="gpt-4.1", help="LLM used to summarize clusters.")
    parser.add_argument("--llm-max-concurrent-calls", type=int, default=8)
    parser.add_argument("--llm-api-timeout", type=int, default=120)
    parser.add_argument("--llm-api-max-retries", type=int, default=4)
    parser.add_argument("--llm-api-staggering-delay", type=float, default=0.05)
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")

    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    """Compute output and logging paths from CLI arguments."""
    if args.output:
        output_path = Path(args.output)
    else:
        if not args.subset:
            raise ValueError("--output is required when --subset is not provided")
        output_path = SRC_DIR.parent / "trees" / args.dataset / args.subset / "tree-bottom-up-llm.pkl"

    json_output_path = Path(args.json_output) if args.json_output else output_path.with_suffix(".json")

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_dir = output_path.parent
        log_path = log_dir / f"{output_path.stem}.log"

    return output_path, json_output_path, log_path


def main() -> None:
    """CLI entry point for the LLM-driven bottom-up tree builder."""
    args = parse_args()
    output_path, json_output_path, log_path = resolve_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bottom_up_tree_builder", str(log_path), logging.INFO)

    records = load_source_records(args, logger)
    summarizer = ClusterSummarizer(args, logger)

    leaf_nodes = build_leaf_nodes(records, args, logger)
    root = build_bottom_up_tree(leaf_nodes, args, summarizer, logger)
    tree_dict = root.to_tree_dict()

    output_path.write_bytes(pickle.dumps(tree_dict))
    logger.info(f"Saved pickle tree to {output_path}")

    if json_output_path:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(tree_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved JSON tree to {json_output_path}")

    logger.info(f"Nodes: {count_nodes(tree_dict)}")
    logger.info(f"Leaves: {count_leaves(tree_dict)}")
    logger.info(f"Max branching: {max_branching(tree_dict)}")


if __name__ == "__main__":
    main()
