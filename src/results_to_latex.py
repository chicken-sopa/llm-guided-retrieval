from __future__ import annotations

"""Turn ECtHR / BRIGHT result files (CSV or JSON) into a LaTeX table for the thesis.

Reads one or more result files produced by the eval and distillation scripts and emits a
``booktabs`` table: one row per run, metrics formatted as percentages, best value per column
optionally bolded, and all text properly LaTeX-escaped (model names are full of ``_`` and ``%``).

Accepted inputs (mixed freely, globs allowed):
  * ``*_ecthr_summary.csv``            -- from test_ecthr_models.py
  * ``*_ecthr_summary.json``           -- same, orient=records
  * ``*_teacher_summary.json``         -- from create_ecthr_distillation_dataset.py (reads .metrics)
  * ``*_ecthr_eval_rows.json/.jsonl``  -- per-case rows; use --per-case to tabulate them raw,
                                          otherwise they are aggregated into a single summary row

Examples
--------
    # One table comparing every run you have
    python src/results_to_latex.py results/ecthr/*_ecthr_summary.csv -o table.tex

    # Pick/reorder columns, add caption + label
    python src/results_to_latex.py "results/**/*_summary.csv" \
        --columns run,cases_evaluated,mean_precision,mean_recall,mean_f1 \
        --caption "ECtHR article retrieval results." --label tab:ecthr -o table.tex

    # Per-case table for a single run (first 20 cases)
    python src/results_to_latex.py out/qwen_ecthr_eval_rows.json --per-case --max-rows 20
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

# Metrics that are 0-1 rates and should render as percentages.
RATE_COLUMNS = {
    "any_gold_found",
    "all_gold_found",
    "exact_set_match",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "precision",
    "recall",
    "f1",
}

# Columns where a larger number is better (used by --bold-best).
HIGHER_IS_BETTER = RATE_COLUMNS | {"true_positive"}

# Integer-valued columns.
INT_COLUMNS = {"cases_evaluated", "case_index", "true_positive", "trace_rows", "training_rows"}

# Pretty LaTeX headers for the columns we know about.
HEADERS = {
    "run": "Run",
    "label": "Run",
    "backend": "Backend",
    "model": "Model",
    "model_id": "Model",
    "adapter_path": "Adapter",
    "elapsed_seconds": "Time (s)",
    "cases_evaluated": "$N$",
    "any_gold_found": "Any gold",
    "all_gold_found": "All gold",
    "exact_set_match": "Exact",
    "mean_gold_removed_by_selector": "Gold removed",
    "mean_precision": "P",
    "mean_recall": "R",
    "mean_f1": "F1",
    "precision": "P",
    "recall": "R",
    "f1": "F1",
    "case_index": "Case",
    "gold": "Gold",
    "predicted": "Predicted",
    "true_positive": "TP",
}

# Sensible default column order for a comparison table.
DEFAULT_SUMMARY_COLUMNS = [
    "run",
    "backend",
    "cases_evaluated",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "all_gold_found",
    "exact_set_match",
]

DEFAULT_PER_CASE_COLUMNS = ["case_index", "gold", "predicted", "precision", "recall", "f1"]


def latex_escape(value: Any) -> str:
    """Escape the characters that would otherwise break or mis-render in LaTeX."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs and validate that every input exists."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
            else:
                print(f"Warning: no file matched {pattern!r}", file=sys.stderr)
                continue
        paths.extend(Path(match) for match in matches)
    if not paths:
        raise SystemExit("No input files found.")
    return paths


def _rows_from_json_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    """Pull tabular rows out of the several JSON shapes these scripts produce."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        # Distillation summary: metrics live under .metrics, identity at the top level.
        metrics = payload.get("metrics")
        if isinstance(metrics, list) and metrics:
            identity = {
                key: payload[key]
                for key in ("label", "run", "backend", "model", "model_id")
                if key in payload
            }
            return [{**identity, **row} for row in metrics if isinstance(row, dict)]
        if isinstance(metrics, dict):
            return [{**payload, **metrics}]
        return [payload]

    print(f"Warning: unrecognized JSON structure in {source}", file=sys.stderr)
    return []


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load one result file (CSV, JSON, or JSONL) into a list of row dicts."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        rows = _rows_from_json_payload(json.loads(path.read_text(encoding="utf-8")), path)
    else:
        raise SystemExit(f"Unsupported file type: {path} (expected .csv, .json, or .jsonl)")

    # Remember where each row came from so a run can be named even if the file has no label.
    for row in rows:
        row.setdefault("_source", path.stem)
    return rows


def infer_run_name(row: dict[str, Any]) -> str:
    """Pick the most descriptive available name for a run."""
    for key in ("run", "label", "model_id", "model"):
        value = row.get(key)
        if value not in (None, "", float("nan")):
            return str(value)
    # Fall back to the filename with the boilerplate suffixes stripped.
    return re.sub(r"_(ecthr_)?(teacher_)?summary$", "", str(row.get("_source", "run")))


def aggregate_per_case_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Collapse per-case eval rows into the same shape as a summary row."""
    n = len(rows)
    if n == 0:
        return {"run": name, "cases_evaluated": 0}

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return sum(values) / len(values) if values else 0.0

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / n

    return {
        "run": name,
        "cases_evaluated": n,
        "any_gold_found": rate("any_gold_found"),
        "all_gold_found": rate("all_gold_found"),
        "exact_set_match": rate("exact_set_match"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "mean_f1": mean("f1"),
    }


def build_table_rows(paths: list[Path], per_case: bool) -> list[dict[str, Any]]:
    """Turn the input files into the final list of table rows."""
    table_rows: list[dict[str, Any]] = []
    for path in paths:
        rows = load_rows(path)
        if not rows:
            continue

        is_per_case = "case_index" in rows[0]
        if is_per_case and not per_case:
            # Aggregate a per-case file down to one comparison row.
            table_rows.append(aggregate_per_case_rows(rows, infer_run_name({"_source": path.stem})))
        else:
            for row in rows:
                row.setdefault("run", infer_run_name(row))
                table_rows.append(row)
    return table_rows


def format_cell(column: str, value: Any, *, precision: int, percent: bool) -> str:
    """Format one cell: percentages for rates, ints for counts, escaped text otherwise."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "--"

    if isinstance(value, bool):
        return r"\checkmark" if value else "--"

    if isinstance(value, (list, tuple)):
        return latex_escape(", ".join(str(item) for item in value)) or "--"

    if isinstance(value, (int, float)):
        if column in INT_COLUMNS:
            return f"{int(value)}"
        if column in RATE_COLUMNS and percent:
            return f"{float(value) * 100:.{precision}f}"
        return f"{float(value):.{precision}f}"

    return latex_escape(value)


def find_best_indexes(rows: list[dict[str, Any]], column: str) -> set[int]:
    """Row indexes holding the best (max) numeric value for a higher-is-better column."""
    if column not in HIGHER_IS_BETTER:
        return set()
    numeric = {
        i: float(row[column])
        for i, row in enumerate(rows)
        if isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool)
    }
    if len(numeric) < 2:
        return set()
    best = max(numeric.values())
    # If every run scored the same there is no winner to highlight.
    if best == min(numeric.values()):
        return set()
    return {i for i, value in numeric.items() if value == best}


def render_latex_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    caption: str | None,
    label: str | None,
    precision: int,
    percent: bool,
    bold_best: bool,
    booktabs: bool,
    standalone_floats: bool,
) -> str:
    """Render the rows as a LaTeX tabular, optionally wrapped in a table float."""
    # Left-align text columns, right-align numeric ones.
    alignment = "".join(
        "r" if any(isinstance(row.get(col), (int, float)) and not isinstance(row.get(col), bool) for row in rows)
        else "l"
        for col in columns
    )

    header_cells = []
    for col in columns:
        head = HEADERS.get(col, col.replace("_", " ").title())
        if percent and col in RATE_COLUMNS:
            head = f"{head} (\\%)"
        header_cells.append(f"\\textbf{{{head}}}" if not booktabs else head)

    best_by_column = {col: find_best_indexes(rows, col) for col in columns} if bold_best else {}

    body_lines = []
    for i, row in enumerate(rows):
        cells = []
        for col in columns:
            cell = format_cell(col, row.get(col), precision=precision, percent=percent)
            if bold_best and i in best_by_column.get(col, set()):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        body_lines.append("    " + " & ".join(cells) + r" \\")

    top, mid, bottom = (r"\toprule", r"\midrule", r"\bottomrule") if booktabs else (r"\hline", r"\hline", r"\hline")

    lines = []
    if standalone_floats:
        lines += [r"\begin{table}[htbp]", r"  \centering"]
    lines += [
        f"  \\begin{{tabular}}{{{alignment}}}",
        f"    {top}",
        "    " + " & ".join(header_cells) + r" \\",
        f"    {mid}",
        *body_lines,
        f"    {bottom}",
        r"  \end{tabular}",
    ]
    if standalone_floats:
        if caption:
            lines.append(f"  \\caption{{{caption}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
        lines.append(r"\end{table}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ECtHR/BRIGHT result CSV or JSON files into a LaTeX (booktabs) table."
    )
    parser.add_argument("inputs", nargs="+", help="Result files or globs (.csv/.json/.jsonl).")
    parser.add_argument("-o", "--output", default=None, help="Write the .tex here (default: stdout).")
    parser.add_argument("--columns", default=None,
                        help="Comma-separated columns to include, in order. Default: a sensible metric set.")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None, help="LaTeX \\label, e.g. tab:ecthr-results.")
    parser.add_argument("--precision", type=int, default=1, help="Decimal places (default: 1).")
    parser.add_argument("--percent", action=argparse.BooleanOptionalAction, default=True,
                        help="Render 0-1 rate metrics as percentages (default: on).")
    parser.add_argument("--bold-best", action=argparse.BooleanOptionalAction, default=True,
                        help="Bold the best value in each higher-is-better column (default: on).")
    parser.add_argument("--booktabs", action=argparse.BooleanOptionalAction, default=True,
                        help="Use \\toprule/\\midrule/\\bottomrule (needs \\usepackage{booktabs}).")
    parser.add_argument("--float", dest="standalone_floats", action=argparse.BooleanOptionalAction, default=True,
                        help="Wrap the tabular in a table float with caption/label (default: on).")
    parser.add_argument("--per-case", action="store_true",
                        help="Tabulate per-case rows individually instead of aggregating them.")
    parser.add_argument("--sort-by", default=None, help="Column to sort rows by (descending for metrics).")
    parser.add_argument("--max-rows", type=int, default=None, help="Keep only the first N rows.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    paths = expand_inputs(args.inputs)
    rows = build_table_rows(paths, args.per_case)
    if not rows:
        raise SystemExit("No usable rows found in the given files.")

    if args.sort_by:
        key = args.sort_by
        reverse = key in HIGHER_IS_BETTER
        rows.sort(
            key=lambda row: (row.get(key) is None, row.get(key) if isinstance(row.get(key), (int, float)) else str(row.get(key))),
            reverse=reverse,
        )

    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    if args.columns:
        columns = [col.strip() for col in args.columns.split(",") if col.strip()]
    else:
        defaults = DEFAULT_PER_CASE_COLUMNS if args.per_case else DEFAULT_SUMMARY_COLUMNS
        # Keep the default columns that actually carry data, then append any extra metric columns.
        columns = [col for col in defaults if any(col in row for row in rows)]
        if not columns:
            columns = [key for key in rows[0] if not key.startswith("_")]

    missing = [col for col in columns if not any(col in row for row in rows)]
    if missing:
        print(f"Warning: columns not present in any row: {', '.join(missing)}", file=sys.stderr)

    table = render_latex_table(
        rows,
        columns,
        caption=args.caption,
        label=args.label,
        precision=args.precision,
        percent=args.percent,
        bold_best=args.bold_best,
        booktabs=args.booktabs,
        standalone_floats=args.standalone_floats,
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
        print(f"Wrote LaTeX table ({len(rows)} rows) to: {output_path}", file=sys.stderr)
        if args.booktabs:
            print("Remember: \\usepackage{booktabs} in your preamble.", file=sys.stderr)
    else:
        print(table)


if __name__ == "__main__":
    main()
