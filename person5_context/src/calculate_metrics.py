"""
Person 5 – Context Length Evaluation
calculate_metrics.py

Loads model responses from results/, computes context-length-specific metrics
(retrieval accuracy, accuracy by context length, degradation slope), and
produces publication-quality visualisations (line chart, heatmap).
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

from evaluation.metrics              import fuzzy_match, exact_match, inference_statistics
from evaluation.visualization_utils import bar_chart, heatmap, line_chart, save_figure

MODELS = ["gpt2", "llama3", "flan-t5"]

# Standard context-length buckets (in tokens)
CONTEXT_BUCKETS = [256, 512, 1024, 2048, 4096]

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


def bucket_context_length(ctx_len: int | None) -> int:
    """Snap a raw context-length to the nearest standard bucket."""
    if ctx_len is None:
        return 256
    diffs  = [abs(ctx_len - b) for b in CONTEXT_BUCKETS]
    return CONTEXT_BUCKETS[diffs.index(min(diffs))]


def is_correct(response: str, ground_truth: str) -> bool:
    if not ground_truth:
        return bool(response.strip())
    return exact_match(response, ground_truth) or fuzzy_match(response, ground_truth, threshold=0.25)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(model_key: str) -> dict:
    records = load_responses(model_key)
    if not records:
        return {}

    times        = []
    bucket_stats: dict[int, dict] = {b: {"correct": 0, "total": 0} for b in CONTEXT_BUCKETS}
    pos_stats: dict[str, dict]    = defaultdict(lambda: {"correct": 0, "total": 0})

    correct_total = 0

    for rec in records:
        response = rec.get("response", "")
        gt       = rec.get("ground_truth", "")
        ctx_len  = rec.get("context_length")
        pos      = rec.get("needle_position") or "unknown"

        ok     = is_correct(response, gt)
        bucket = bucket_context_length(ctx_len)

        if ok:
            correct_total += 1
        bucket_stats[bucket]["correct"] += int(ok)
        bucket_stats[bucket]["total"]   += 1
        pos_stats[pos]["correct"]       += int(ok)
        pos_stats[pos]["total"]         += 1

        t = rec.get("inference_time")
        if t is not None:
            times.append(float(t))

    n = len(records)

    # Accuracy per bucket
    acc_by_length = {
        b: round(v["correct"] / v["total"], 4) if v["total"] else 0.0
        for b, v in bucket_stats.items()
    }

    # Degradation slope (linear regression over bucket accuracies)
    xs      = np.array([b for b in CONTEXT_BUCKETS if bucket_stats[b]["total"] > 0], dtype=float)
    ys      = np.array([acc_by_length[b] for b in CONTEXT_BUCKETS if bucket_stats[b]["total"] > 0])
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0])
    else:
        slope = 0.0

    # Accuracy by needle position
    acc_by_position = {
        pos: round(v["correct"] / v["total"], 4) if v["total"] else 0.0
        for pos, v in pos_stats.items()
    }

    return {
        "model":            model_key,
        "total_examples":   n,
        "retrieval_accuracy": round(correct_total / n, 4) if n else 0.0,
        "acc_by_length":    acc_by_length,
        "acc_by_position":  acc_by_position,
        "degradation_slope": round(slope, 6),
        "timing":            inference_statistics(times),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_retrieval_vs_context_length(all_metrics: list[dict]):
    """Line chart: retrieval accuracy as a function of context length."""
    data = {}
    for m in all_metrics:
        model     = m["model"]
        acc_map   = m.get("acc_by_length", {})
        data[model] = [acc_map.get(b, 0.0) for b in CONTEXT_BUCKETS]

    x_vals = [str(b) for b in CONTEXT_BUCKETS]

    fig = line_chart(
        data={model: (x_vals, vals) for model, vals in data.items()},
        title="Retrieval Accuracy vs. Context Length",
        xlabel="Context Length (tokens)",
        ylabel="Accuracy",
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, PLOTS_DIR / "retrieval_vs_context_length.png")
    print("  ✓ Plot saved: retrieval_vs_context_length.png")


def plot_model_length_heatmap(all_metrics: list[dict]):
    """Heatmap: model × context-length accuracy matrix."""
    model_names = [m["model"] for m in all_metrics]
    matrix      = [
        [m.get("acc_by_length", {}).get(b, 0.0) for b in CONTEXT_BUCKETS]
        for m in all_metrics
    ]
    xlabels = [str(b) for b in CONTEXT_BUCKETS]

    fig = heatmap(
        matrix=matrix, xlabels=xlabels, ylabels=model_names,
        title="Retrieval Accuracy: Model × Context Length",
        cmap="Blues",
    )
    save_figure(fig, PLOTS_DIR / "model_length_heatmap.png")
    print("  ✓ Plot saved: model_length_heatmap.png")


def plot_position_accuracy(all_metrics: list[dict]):
    """Bar chart: accuracy by needle position, grouped by model."""
    positions = ["beginning", "middle", "end", "unknown"]
    data      = {
        m["model"]: [m.get("acc_by_position", {}).get(pos, 0.0) for pos in positions]
        for m in all_metrics
    }

    fig = bar_chart(
        data=data, labels=positions,
        title="Retrieval Accuracy by Needle Position",
        ylabel="Accuracy", ylim=(0, 1),
    )
    save_figure(fig, PLOTS_DIR / "position_accuracy.png")
    print("  ✓ Plot saved: position_accuracy.png")


def plot_degradation_slope(all_metrics: list[dict]):
    """Bar chart: degradation slope per model (negative = gets worse with length)."""
    data   = {m["model"]: [m["degradation_slope"]] for m in all_metrics}
    labels = ["Degradation Slope"]

    min_v = min(m["degradation_slope"] for m in all_metrics)
    max_v = max(m["degradation_slope"] for m in all_metrics)
    margin = max(abs(min_v), abs(max_v)) * 0.2 or 0.001

    fig = bar_chart(
        data=data, labels=labels,
        title="Context-Length Degradation Slope (per model)",
        ylabel="Slope (acc / token)",
        ylim=(min_v - margin, max_v + margin),
    )
    save_figure(fig, PLOTS_DIR / "degradation_slope.png")
    print("  ✓ Plot saved: degradation_slope.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_report(all_metrics: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Metrics report saved → {report_path.name}")


def print_summary(all_metrics: list[dict]):
    print("\n" + "="*60)
    print("  Person 5 — Context Length  |  Metric Summary")
    print("="*60)
    for m in all_metrics:
        print(
            f"  {m['model']:<12}"
            f" retrieval={m['retrieval_accuracy']:.3f}"
            f" slope={m['degradation_slope']:.6f}"
            f" n={m['total_examples']}"
        )
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate context-length metrics for Person 5."
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
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

    plot_retrieval_vs_context_length(all_metrics)
    plot_model_length_heatmap(all_metrics)
    plot_position_accuracy(all_metrics)
    plot_degradation_slope(all_metrics)

    save_report(all_metrics)
    print_summary(all_metrics)

    print("\n✅  Person 5 metric calculation complete.")


if __name__ == "__main__":
    main()
