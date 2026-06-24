"""Parse the ECHR Convention PDF into one chunk per article.

Source: https://www.echr.coe.int/Documents/Convention_ENG.pdf

Input/output both live in parsing/echr/:
    parsing/echr/Convention_ENG.pdf          (source — download here)
    parsing/echr/convention_articles.jsonl   (parsed chunks — written here)

Output: convention_articles.jsonl, one record per article:
    {"article_id": "article_6", "label": "Article 6", "part": "convention",
     "title": "Right to a fair trial", "content": "..."}

The `article_id` is normalized with the SAME function the ECtHR eval uses
(ecthr_evaluation.normalize_article_label), so predicted articles line up with
the dataset's gold labels.

NOTE: PDF text extraction is heuristic. After running, validate the article
count/ids against src/tree_construction/EU_conventions_example/Convention_ENG_chunks.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


# "ARTICLE 6" optionally followed by an inline "– Right to a fair trial"
ARTICLE_RE = re.compile(r"^\s*ARTICLE\s+(\d+[A-Za-z]?)\s*(?:[–\-—]\s*(.*))?$", re.IGNORECASE)
# "Protocol No. 1" / "PROTOCOL No. 4"
PROTOCOL_RE = re.compile(r"^\s*PROTOCOL\s+No\.?\s*(\d+)\b", re.IGNORECASE)


def _extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse(pdf_path: Path) -> list[dict]:
    _add_src_to_path()
    from llm_rl_playground.ecthr_evaluation import article_id_to_display, normalize_article_label

    lines = [ln.rstrip() for ln in _extract_text(pdf_path).splitlines()]

    records: list[dict] = []
    current: dict | None = None
    protocol: int | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        body = "\n".join(current["_buf"]).strip()
        proto = current["protocol"]
        label_str = (
            f"Protocol {proto} Article {current['number']}" if proto else f"Article {current['number']}"
        )
        norm = normalize_article_label(label_str)
        if norm:
            title = current["title"].strip()
            content = (f"{title}\n{body}" if title else body).strip()
            records.append(
                {
                    "article_id": norm,
                    "label": article_id_to_display(norm),
                    "part": f"protocol_{proto}" if proto else "convention",
                    "title": title,
                    "content": content,
                }
            )
        current = None

    for line in lines:
        proto_match = PROTOCOL_RE.match(line)
        if proto_match:
            flush()
            protocol = int(proto_match.group(1))
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush()
            current = {
                "number": article_match.group(1),
                "title": (article_match.group(2) or "").strip(),
                "protocol": protocol,
                "_buf": [],
            }
            continue

        if current is not None:
            stripped = line.strip()
            # Title may sit on the line after "ARTICLE N": grab a short first line.
            if not current["title"] and not current["_buf"] and stripped and len(stripped) < 80 and not stripped[0].isdigit():
                current["title"] = stripped
            else:
                current["_buf"].append(line)

    flush()

    # Deduplicate by article_id, keeping the longest content (drops TOC stubs).
    best: dict[str, dict] = {}
    for record in records:
        existing = best.get(record["article_id"])
        if existing is None or len(record["content"]) > len(existing["content"]):
            best[record["article_id"]] = record
    return list(best.values())


def main() -> None:
    here = Path(__file__).resolve()
    echr_dir = here.parent / "echr"
    default_pdf = echr_dir / "Convention_ENG.pdf"
    default_out = echr_dir / "convention_articles.jsonl"

    parser = argparse.ArgumentParser(description="Parse the ECHR Convention PDF into per-article chunks.")
    parser.add_argument("--pdf", default=str(default_pdf))
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(
            f"PDF not found: {pdf_path}\nDownload it into the echr folder:\n"
            f"  curl -L https://www.echr.coe.int/Documents/Convention_ENG.pdf -o {pdf_path}"
        )

    records = parse(pdf_path)
    if not records:
        raise SystemExit("Parsed 0 articles — the PDF layout may differ from expectations; inspect the text.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} articles to {out_path}")
    print("Validate against src/tree_construction/EU_conventions_example/Convention_ENG_chunks.json")


if __name__ == "__main__":
    main()
