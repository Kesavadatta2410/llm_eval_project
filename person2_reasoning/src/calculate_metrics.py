"""
Person 2 – Reasoning & Logic
calculate_metrics.py

Loads model responses from results/, computes reasoning-specific metrics
(accuracy, step correctness, sub-category breakdown, timing), and produces
publication-quality visualisations.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).parent
PERSON_DIR  = SRC_DIR.parent
PROJECT_DIR = PERSON_DIR.parent
RESULTS_DIR = PERSON_DIR / "results"
PLOTS_DIR   = PERSON_DIR / "plots"

sys.path.insert(0, str(PROJECT_DIR))

from evaluation.metrics            import exact_match, fuzzy_match, compute_accuracy, inference_statistics
from evaluation.visualization_utils import bar_chart, heatmap, line_chart, save_figure

MODELS = ["gpt2", "llama3", "flan-t5"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_responses(model_key: str) -> list[dict]:
    path = RESULTS_DIR / f"{model_key}_responses.jsonl"
    if not path.exists():
        print(f"  ! No responses file for {model_key} — skipping.")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_correct(response: str, ground_truth: str, expected_behavior: str) -> bool:
    """Flexible correctness check for reasoning tasks."""
    if not ground_truth:
        # No ground truth: check if model attempted a response
        return bool(response.strip())
    if expected_behavior == "should_reason_correctly":
        return fuzzy_match(response, ground_truth, threshold=0.3)
    return exact_match(response, ground_truth) or fuzzy_match(response, ground_truth, threshold=0.25)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(model_key: str) -> dict:
    records = load_responses(model_key)
    if not records:
        return {}

    correct_total = 0
    times         = []
    sub_stats     = defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in records:
        response  = rec.get("response", "")
        gt        = rec.get("ground_truth", "")
        eb        = rec.get("expected_behavior", "should_reason_correctly")
        sub_cat   = rec.get("sub_category", "unknown")

        ok = is_correct(response, gt, eb)
        if ok:
            correct_total += 1
        sub_stats[sub_cat]["correct"] += int(ok)
        sub_stats[sub_cat]["total"]   += 1

        t = rec.get("inference_time")
        if t is not None:
            times.append(float(t))

    n = len(records)
    sub_accuracy = {
        sub: round(v["correct"] / v["total"], 4) if v["total"] else 0.0
        for sub, v in sub_stats.items()
    }

    return {
        "model":          model_key,
        "total_examples": n,
        "correct":        correct_total,
        "accuracy":       round(correct_total / n, 4) if n else 0.0,
        "sub_accuracy":   sub_accuracy,
        "timing":         inference_statistics(times),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_accuracy_by_subcategory(all_metrics: list[dict]):
    """Bar chart: accuracy per sub-category, grouped by model."""
    # Collect all sub-categories
    all_subs = sorted({
        sub
        for m in all_metrics
        for sub in m.get("sub_accuracy", {}).keys()
    })
    if not all_subs:
        return

    data   = {}
    labels = all_subs
    for m in all_metrics:
        key = m["model"]
        data[key] = [m.get("sub_accuracy", {}).get(sub, 0.0) for sub in all_subs]

    fig = bar_chart(
        data=data,
        labels=labels,
        title="Reasoning Accuracy by Sub-Category",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, PLOTS_DIR / "accuracy_by_subcategory.png")
    print("  ✓ Plot saved: accuracy_by_subcategory.png")


def plot_model_subcategory_heatmap(all_metrics: list[dict]):
    """Heatmap: model × sub-category accuracy matrix."""
    all_subs = sorted({
        sub
        for m in all_metrics
        for sub in m.get("sub_accuracy", {}).keys()
    })
    if not all_subs:
        return

    model_names = [m["model"] for m in all_metrics]
    matrix      = [
        [m.get("sub_accuracy", {}).get(sub, 0.0) for sub in all_subs]
        for m in all_metrics
    ]

    fig = heatmap(
        matrix=matrix,
        xlabels=all_subs,
        ylabels=model_names,
        title="Model × Sub-Category Accuracy Heatmap",
        cmap="Blues",
    )
    save_figure(fig, PLOTS_DIR / "model_subcategory_heatmap.png")
    print("  ✓ Plot saved: model_subcategory_heatmap.png")


def plot_overall_accuracy(all_metrics: list[dict]):
    """Simple bar chart of overall accuracy per model."""
    data   = {m["model"]: [m["accuracy"]] for m in all_metrics}
    labels = ["Overall Accuracy"]

    fig = bar_chart(
        data=data,
        labels=labels,
        title="Reasoning Accuracy by Model",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    save_figure(fig, PLOTS_DIR / "overall_accuracy.png")
    print("  ✓ Plot saved: overall_accuracy.png")


def plot_inference_timing(all_metrics: list[dict]):
    """Line chart: mean inference time per model."""
    model_names = [m["model"]                        for m in all_metrics]
    mean_times  = [m.get("timing", {}).get("mean", 0) for m in all_metrics]

    data = {"mean_time": (model_names, mean_times)}
    fig  = line_chart(
        data=data,
        title="Mean Inference Time per Model (Reasoning)",
        xlabel="Model",
        ylabel="Time (s)",
    )
    save_figure(fig, PLOTS_DIR / "inference_timing.png")
    print("  ✓ Plot saved: inference_timing.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_report(all_metrics: list[dict]):
    report_path = RESULTS_DIR / "metrics_report.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Metrics report saved → {report_path.name}")


def print_summary(all_metrics: list[dict]):
    print("\n" + "="*60)
    print("  Person 2 — Reasoning & Logic  |  Metric Summary")
    print("="*60)
    for m in all_metrics:
        print(
            f"  {m['model']:<12} accuracy={m['accuracy']:.3f} "
            f" n={m['total_examples']}"
            f" timing_mean={m.get('timing',{}).get('mean',0):.3f}s"
        )
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate reasoning metrics for Person 2."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models to calculate metrics for.",
    )
    args = parser.parse_args()

    all_metrics = []
    for model_key in args.models:
        print(f"\nProcessing: {model_key}")
        metrics = compute_metrics(model_key)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        print("  ! No metrics to compute. Run run_evaluation.py first.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Visualisations
    plot_accuracy_by_subcategory(all_metrics)
    plot_model_subcategory_heatmap(all_metrics)
    plot_overall_accuracy(all_metrics)
    plot_inference_timing(all_metrics)

    # Report
    save_report(all_metrics)
    print_summary(all_metrics)

    print("\n✅  Person 2 metric calculation complete.")


if __name__ == "__main__":
    main()
