"""Build a metadata-guided bottom-up tree for Portuguese Civil Code chunks.

The input is the chunk JSON produced by the legal chunker. The output is a
plain nested dict compatible with ``tree_objects.SemanticNode.load_dict``.

Example:
    python src/tree_construction/build_civil_code_tree.py \
      --input src/tree_construction/codigo_civil_chunks_sample.json \
      --output trees/PT/codigo_civil/tree-bottom-up.pkl \
      --json-output trees/PT/codigo_civil/tree-bottom-up.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


STRUCTURE_LEVELS = (
    ("book", "book_number", "book_title"),
    ("title", "title_number", "title_title"),
    ("subtitle", "subtitle_number", "subtitle_title"),
    ("chapter", "chapter_number", "chapter_title"),
    ("section", "section_number", "section_title"),
    ("subsection", "subsection_number", "subsection_title"),
)

CONTENT_CHUNK_TYPES = {"article_body", "article_paragraph", "article_point"}
HEADER_CHUNK_TYPES = {"article_header"}
FOOTER_RE = re.compile(
    r"\s*Vers[aã]o\s+[aà]\s+data\s+de\s+\d{1,2}-\d{1,2}-\d{4}\s+P[aá]g\.\s+\d+\s+de\s+\d+\s*",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = FOOTER_RE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = clean_text(value)
    return value or None


def normalize_for_compare(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[\s.]+", " ", value)
    return value.strip()


def extract_article_title_from_text(text: str, citation: str | None) -> str | None:
    if not citation:
        return None
    text = clean_text(text)
    pattern = rf"{re.escape(citation)}\s*(\([^)]+\))"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def chunk_has_substance(chunk: dict[str, Any]) -> bool:
    if chunk.get("chunk_type") in HEADER_CHUNK_TYPES:
        return False

    text = normalize_for_compare(chunk.get("text", ""))
    citation = normalize_for_compare(str(chunk.get("citation") or ""))
    title = normalize_for_compare(str(chunk.get("article_title") or ""))
    if not text:
        return False

    header_forms = {citation}
    if citation and title:
        header_forms.add(f"{citation} {title}")
    return text not in header_forms


def compact_label(number: Any, title: Any, fallback: str) -> str:
    parts = [str(x).strip() for x in (number, title) if x not in (None, "")]
    return " - ".join(parts) if parts else fallback


def slug(value: str, max_len: int = 60) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "node")[:max_len]


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def parse_article_sort_key(article_number: str | None, fallback: int) -> tuple[int, str, int]:
    if not article_number:
        return (fallback, "", fallback)
    match = re.search(r"\d+", article_number)
    number = int(match.group(0)) if match else fallback
    return (number, article_number, fallback)


def mean_embedding(embeddings: Iterable[list[float] | None]) -> list[float] | None:
    vectors = [emb for emb in embeddings if emb]
    if not vectors:
        return None
    size = len(vectors[0])
    sums = [0.0] * size
    count = 0
    for vector in vectors:
        if len(vector) != size:
            continue
        count += 1
        for idx, value in enumerate(vector):
            sums[idx] += float(value)
    if count == 0:
        return None
    return [value / count for value in sums]


def make_node(
    node_id: str,
    desc: str,
    children: list[dict[str, Any]] | None = None,
    embs: list[float] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    node = {
        "id": node_id,
        "desc": desc,
        "embs": embs,
        "child": children or None,
    }
    if metadata:
        node.update(metadata)
    return node


def node_leaf_count(node: dict[str, Any]) -> int:
    children = node.get("child") or []
    if not children:
        return 1
    return sum(node_leaf_count(child) for child in children)


def child_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    meta = node.get("metadata") or {}
    order = meta.get("order")
    if order is None:
        order = meta.get("article_order")
    return (order if order is not None else 10**12, str(node.get("id", "")))


def article_range_label(children: list[dict[str, Any]]) -> str:
    first_meta = children[0].get("metadata") or {}
    last_meta = children[-1].get("metadata") or {}
    first_order = first_meta.get("article_order", first_meta.get("order", 0))
    last_order = last_meta.get("article_order", last_meta.get("order", first_order))
    first_citation = first_meta.get("citation") or first_meta.get("article_number")
    last_citation = last_meta.get("citation") or last_meta.get("article_number")
    if first_citation and last_citation:
        return f"Article group {first_order + 1}-{last_order + 1}: {first_citation} through {last_citation}"
    return f"{len(children)} child nodes"


def describe_internal(label: str, children: list[dict[str, Any]]) -> str:
    leaf_count = sum(node_leaf_count(child) for child in children)
    citations = []
    titles = []
    for child in children[:12]:
        meta = child.get("metadata") or {}
        if meta.get("citation"):
            citations.append(meta["citation"])
        title = meta.get("article_title") or meta.get("label")
        if title:
            titles.append(str(title).strip())
    parts = [label, f"{len(children)} direct children", f"{leaf_count} leaf passages"]
    if citations:
        parts.append("Citations: " + "; ".join(citations[:8]))
    if titles:
        parts.append("Topics: " + "; ".join(dict.fromkeys(titles[:8])))
    return ". ".join(parts)


def split_large_children(
    node: dict[str, Any],
    max_children: int,
    range_size: int | None = None,
) -> dict[str, Any]:
    children = node.get("child") or []
    if not children:
        return node

    node["child"] = [split_large_children(child, max_children, range_size) for child in children]
    children = sorted(node["child"], key=child_sort_key)
    if len(children) <= max_children:
        node["child"] = children
        node["embs"] = node.get("embs") or mean_embedding(child.get("embs") for child in children)
        return node

    group_size = range_size or max_children
    grouped_children = []
    for group_idx, start in enumerate(range(0, len(children), group_size), start=1):
        group = children[start : start + group_size]
        label = article_range_label(group)
        parent_id = f"{node['id']}:range:{group_idx:04d}:{short_hash(label)}"
        grouped_children.append(
            make_node(
                parent_id,
                describe_internal(label, group),
                group,
                mean_embedding(child.get("embs") for child in group),
                node_type="range",
                metadata={
                    "label": label,
                    "order": child_sort_key(group[0])[0],
                    "source_parent_id": node["id"],
                },
            )
        )

    node["child"] = grouped_children
    node["embs"] = mean_embedding(child.get("embs") for child in grouped_children)
    return node


def article_key_for_header(chunk: dict[str, Any], article_order: int) -> str:
    raw = "|".join(
        str(chunk.get(field) or "")
        for field in (
            "source_id",
            "page_start",
            "article_number",
            "article_title",
            "chapter_title",
            "section_title",
            "subsection_title",
        )
    )
    return f"article:{article_order:06d}:{slug(chunk.get('article_number') or 'article')}:{short_hash(raw)}"


def group_chunks_by_article(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    articles = []
    current = None

    for order, chunk in enumerate(chunks):
        chunk = dict(chunk)
        chunk["text"] = clean_text(chunk.get("text", ""))
        chunk["_order"] = order

        starts_article = chunk.get("chunk_type") in HEADER_CHUNK_TYPES or current is None
        if starts_article:
            current = {
                "article_uid": article_key_for_header(chunk, len(articles) + 1),
                "article_order": len(articles),
                "header": chunk if chunk.get("chunk_type") in HEADER_CHUNK_TYPES else None,
                "chunks": [],
            }
            articles.append(current)

        current["chunks"].append(chunk)

    return articles


def choose_article_metadata(article: dict[str, Any]) -> dict[str, Any]:
    chunks = article["chunks"]
    header = article.get("header") or chunks[0]
    metadata = {key: clean_metadata_value(value) for key, value in header.items()}
    metadata.pop("embedding", None)
    metadata.pop("text", None)
    metadata.pop("_order", None)

    for field in [number_field for _, number_field, _ in STRUCTURE_LEVELS] + [
        title_field for _, _, title_field in STRUCTURE_LEVELS
    ]:
        if metadata.get(field):
            continue
        for chunk in chunks:
            if chunk.get(field):
                metadata[field] = clean_metadata_value(chunk.get(field))
                break

    if not metadata.get("article_title"):
        for chunk in chunks:
            derived_title = extract_article_title_from_text(
                chunk.get("text", ""),
                metadata.get("citation"),
            )
            if derived_title:
                metadata["article_title"] = derived_title
                break

    metadata["article_uid"] = article["article_uid"]
    metadata["article_order"] = article["article_order"]
    metadata["article_sort_key"] = parse_article_sort_key(
        metadata.get("article_number"),
        article["article_order"],
    )
    return metadata


def make_leaf_node(article: dict[str, Any], chunk: dict[str, Any], leaf_idx: int) -> dict[str, Any]:
    citation = chunk.get("citation") or chunk.get("article_number") or article["article_uid"]
    label = citation
    text = clean_text(chunk.get("text", ""))
    desc = text if text.startswith(str(citation)) else f"{citation}. {text}"
    leaf_id = f"leaf:{article['article_uid']}:{leaf_idx:03d}:{chunk.get('chunk_id', short_hash(desc))}"
    metadata = {
        key: clean_metadata_value(value)
        for key, value in chunk.items()
        if key not in {"embedding", "text"}
    }
    if not metadata.get("article_title"):
        metadata["article_title"] = extract_article_title_from_text(
            chunk.get("text", ""),
            chunk.get("citation"),
        )
    metadata.update(
        {
            "label": label,
            "article_uid": article["article_uid"],
            "source_chunk_id": chunk.get("chunk_id"),
            "order": chunk.get("_order", leaf_idx),
        }
    )
    return make_node(
        leaf_id,
        desc,
        children=None,
        embs=chunk.get("embedding"),
        node_type="legal_chunk",
        metadata=metadata,
    )


def make_article_node(article: dict[str, Any]) -> dict[str, Any] | None:
    metadata = choose_article_metadata(article)
    content_chunks = [
        chunk
        for chunk in article["chunks"]
        if chunk.get("chunk_type") in CONTENT_CHUNK_TYPES and chunk_has_substance(chunk)
    ]
    if not content_chunks:
        content_chunks = article["chunks"]

    leaves = [
        make_leaf_node(article, chunk, leaf_idx)
        for leaf_idx, chunk in enumerate(content_chunks, start=1)
        if clean_text(chunk.get("text", ""))
    ]
    if not leaves:
        return None

    citation = metadata.get("citation") or f"CC Art. {metadata.get('article_number')}"
    title = metadata.get("article_title") or ""
    label = f"{citation} {title}".strip()
    desc = describe_internal(label, leaves)
    return make_node(
        article["article_uid"],
        desc,
        leaves,
        mean_embedding(leaf.get("embs") for leaf in leaves),
        node_type="article",
        metadata={
            **metadata,
            "label": label,
            "citation": citation,
            "order": metadata["article_order"],
        },
    )


def structure_path_for_article(article_node: dict[str, Any]) -> list[tuple[str, str, str]]:
    meta = article_node.get("metadata") or {}
    path = []
    for level_name, number_field, title_field in STRUCTURE_LEVELS:
        number = meta.get(number_field)
        title = meta.get(title_field)
        if title or number:
            label = compact_label(number, title, level_name.title())
            path.append((level_name, slug(label), label))
    if not path:
        path.append(("uncaptured_structure", "uncaptured_structure", "Uncaptured or front-matter structure"))
    return path


def insert_article(root_bucket: OrderedDict, article_node: dict[str, Any]) -> None:
    bucket = root_bucket
    current_entry = None
    for level_name, key, label in structure_path_for_article(article_node):
        compound_key = f"{level_name}:{key}"
        if compound_key not in bucket:
            bucket[compound_key] = {
                "_label": label,
                "_level_name": level_name,
                "_order": child_sort_key(article_node)[0],
                "_children": OrderedDict(),
                "_articles": [],
            }
        entry = bucket[compound_key]
        entry["_order"] = min(entry["_order"], child_sort_key(article_node)[0])
        bucket = entry["_children"]
        current_entry = entry
    if current_entry is not None:
        current_entry["_articles"].append(article_node)


def bucket_to_nodes(bucket: OrderedDict, parent_id: str) -> list[dict[str, Any]]:
    nodes = []
    for key, entry in bucket.items():
        children = bucket_to_nodes(entry["_children"], f"{parent_id}:{key}")
        articles = sorted(entry["_articles"], key=child_sort_key)
        all_children = children + articles
        node_id = f"{parent_id}:{key}"
        nodes.append(
            make_node(
                node_id,
                describe_internal(entry["_label"], all_children),
                all_children,
                mean_embedding(child.get("embs") for child in all_children),
                node_type=entry["_level_name"],
                metadata={
                    "label": entry["_label"],
                    "level": entry["_level_name"],
                    "order": entry["_order"],
                },
            )
        )
    return nodes


def build_tree(data: dict[str, Any], max_children: int) -> dict[str, Any]:
    document = data.get("document", {})
    chunks = data.get("chunks", [])
    articles = group_chunks_by_article(chunks)
    article_nodes = [node for article in articles if (node := make_article_node(article))]

    root_bucket = OrderedDict()
    for article_node in article_nodes:
        insert_article(root_bucket, article_node)

    root_children = bucket_to_nodes(root_bucket, "root")
    title = document.get("document_title") or document.get("source_id") or "Portuguese Civil Code"
    abbreviation = document.get("document_abbreviation") or "CC"
    root = make_node(
        f"doc:{slug(document.get('source_id') or title)}",
        describe_internal(f"{abbreviation} - {title}", root_children),
        root_children,
        mean_embedding(child.get("embs") for child in root_children),
        node_type="document",
        metadata={key: value for key, value in document.items() if key != "source_path"},
    )
    return split_large_children(root, max_children)


def count_nodes(node: dict[str, Any]) -> int:
    return 1 + sum(count_nodes(child) for child in node.get("child") or [])


def count_leaves(node: dict[str, Any]) -> int:
    children = node.get("child") or []
    if not children:
        return 1
    return sum(count_leaves(child) for child in children)


def max_branching(node: dict[str, Any]) -> int:
    children = node.get("child") or []
    return max([len(children), *(max_branching(child) for child in children)] or [0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to chunk JSON.")
    parser.add_argument("--output", required=True, help="Path to output pickle tree dict.")
    parser.add_argument("--json-output", help="Optional path to output JSON tree dict.")
    parser.add_argument("--max-children", type=int, default=16)
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    tree = build_tree(data, max_children=args.max_children)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(tree, f)

    if args.json_output:
        json_output_path = Path(args.json_output)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved tree to {output_path}")
    if args.json_output:
        print(f"Saved JSON tree to {args.json_output}")
    print(f"Nodes: {count_nodes(tree)}")
    print(f"Leaves: {count_leaves(tree)}")
    print(f"Max branching: {max_branching(tree)}")


if __name__ == "__main__":
    main()
