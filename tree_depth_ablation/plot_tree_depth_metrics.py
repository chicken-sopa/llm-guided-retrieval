"""Plot ECtHR eval metrics as a function of tree depth.

INPUT
    The aggregated results file written by run_tree_depth_ablation.py — one row
    per fanout config, carrying at least `depth` and the mean metric columns
    (mean_precision / mean_recall / mean_f1). Both the .json and .csv the
    ablation writes are accepted (format is auto-detected from the extension).

OUTPUT
    One PNG per metric (metric vs depth) plus a combined overlay, written to the
    output folder. Depth is emergent from the fanout cap, so points are ordered
    by depth and annotated with their max_children value when present.

USAGE (standalone)
    python tree_depth_ablation/plot_tree_depth_metrics.py \
        --input tree_depth_ablation/tree_depth_ablation.json \
        --output-dir tree_depth_ablation/plots \
        --metrics precision recall f1

    The ablation calls generate_plots(...) itself once the sweep finishes, so a
    normal run produces the plots without a second command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the GPU box
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


# A requested metric token maps to the first column that exists in the frame,
# so callers can say "precision" and get "mean_precision".
def resolve_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    for candidate in (metric, f"mean_{metric}", f"{metric}_mean"):
        if candidate in df.columns:
            return candidate
    return None


def load_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    if input_path.suffix.lower() == ".json":
        return pd.read_json(input_path)
    return pd.read_csv(input_path)


def prepare_frame(
    df: pd.DataFrame,
    depth_col: str,
    *,
    drop_invalid: bool,
    min_coverage: float | None,
) -> pd.DataFrame:
    if depth_col not in df.columns:
        raise KeyError(
            f"Depth column '{depth_col}' not in results (columns: {list(df.columns)})"
        )

    frame = df.copy()
    # Failed configs have no depth/metrics — never plot them.
    frame = frame[frame[depth_col].notna()]

    if drop_invalid and "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "ok"]
    # coverage ~0 means traversal never reached leaf articles -> not a real point.
    if min_coverage is not None and "coverage" in frame.columns:
        frame = frame[frame["coverage"].fillna(0.0) >= min_coverage]

    return frame.sort_values(depth_col)


def _annotate_children(ax, xs, ys, labels) -> None:
    for x, y, label in zip(xs, ys, labels):
        if label is None:
            continue
        ax.annotate(
            f"mc{label}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="0.35",
        )


def generate_plots(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = ("precision", "recall", "f1"),
    depth_col: str = "depth",
    drop_invalid: bool = False,
    min_coverage: float | None = None,
    combined: bool = True,
    dpi: int = 150,
    title_prefix: str = "",
) -> Path:
    """Write one PNG per metric (and a combined overlay) to output_dir.

    Returns the output directory. Safe to call in a try/except from the ablation:
    it raises only on genuinely unusable input, not on a single missing metric.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(input_path)
    frame = prepare_frame(
        df, depth_col, drop_invalid=drop_invalid, min_coverage=min_coverage
    )
    if frame.empty:
        raise ValueError(
            f"No plottable rows in {input_path} after filtering "
            f"(drop_invalid={drop_invalid}, min_coverage={min_coverage})."
        )

    depths = frame[depth_col].tolist()
    children = (
        frame["max_children"].tolist()
        if "max_children" in frame.columns
        else [None] * len(frame)
    )

    resolved = []  # (metric_name, column) actually present
    for metric in metrics:
        column = resolve_metric_column(frame, metric)
        if column is None:
            print(
                f"[plot] skipping '{metric}': no matching column "
                f"(looked for {metric}/mean_{metric}).",
                file=sys.stderr,
            )
            continue
        resolved.append((metric, column))

    if not resolved:
        raise ValueError(
            f"None of the requested metrics {list(metrics)} were found in "
            f"{input_path} (columns: {list(frame.columns)})."
        )

    prefix = f"{title_prefix} " if title_prefix else ""

    # One figure per metric.
    for metric, column in resolved:
        values = frame[column].tolist()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(depths, values, marker="o", linewidth=2)
        _annotate_children(ax, depths, values, children)
        ax.set_xlabel("Tree depth")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{prefix}{metric.capitalize()} vs tree depth")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        out_path = output_dir / f"{metric}_vs_depth.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    # Combined overlay of all resolved metrics.
    if combined and len(resolved) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for metric, column in resolved:
            ax.plot(depths, frame[column].tolist(), marker="o", linewidth=2, label=metric.capitalize())
        ax.set_xlabel("Tree depth")
        ax.set_ylabel("Score")
        ax.set_title(f"{prefix}Metrics vs tree depth")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / "metrics_vs_depth.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot ECtHR metrics vs tree depth.")
    parser.add_argument(
        "--input", "-i", type=Path,
        default=script_dir / "tree_depth_ablation.json",
        help="Aggregated ablation results file (.json or .csv).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Folder for the PNGs (default: <input parent>/plots).",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["precision", "recall", "f1"],
        help="Metrics to plot (resolved to mean_<metric> if needed).",
    )
    parser.add_argument("--depth-col", default="depth", help="Column used for the x-axis.")
    parser.add_argument(
        "--drop-invalid", action="store_true",
        help="Drop rows whose status != 'ok'.",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=None,
        help="Drop rows whose coverage is below this (e.g. 0.01 removes runs that never reached leaves).",
    )
    parser.add_argument("--no-combined", dest="combined", action="store_false",
                        help="Skip the combined overlay figure.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title-prefix", default="", help="Optional prefix for figure titles.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or (args.input.parent / "plots")
    generate_plots(
        args.input,
        output_dir,
        metrics=args.metrics,
        depth_col=args.depth_col,
        drop_invalid=args.drop_invalid,
        min_coverage=args.min_coverage,
        combined=args.combined,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )


if __name__ == "__main__":
    main()
