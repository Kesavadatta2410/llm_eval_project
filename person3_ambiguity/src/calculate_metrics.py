"""
Person 3 – Ambiguity Handling
calculate_metrics.py

Loads model responses from results/, computes ambiguity-specific metrics
(clarification rate, disambiguation success, ambiguity-type breakdown,
timing), and produces publication-quality visualisations.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).parent
PERSON_DIR  = SRC_DIR.parent
PROJECT_DIR = PERSON_DIR.parent
RESULTS_DIR = PERSON_DIR / "results"
PLOTS_DIR   = PERSON_DIR / "plots"

sys.path.insert(0, str(PROJECT_DIR))

from evaluation.metrics              import fuzzy_match, inference_statistics
from evaluation.visualization_utils import bar_chart, heatmap, line_chart, save_figure

MODELS = ["gpt2", "llama3", "flan-t5"]

# ---------------------------------------------------------------------------
# Clarification / disambiguation keywords
# ---------------------------------------------------------------------------
CLARIFICATION_PHRASES = [
    "could you clarify", "it depends", "which one", "ambiguous",
    "unclear", "i'm not sure", "please specify", "more context",
    "could mean", "either", "i need more information",
]

DISAMBIGUATION_PHRASES = [
    "refers to", "means", "in this context", "specifically",
    "the answer is", "it is", "clearly",
]


def _has_phrase(text: str, phrases: list[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


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


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(model_key: str) -> dict:
    records = load_responses(model_key)
    if not records:
        return {}

    clarified_count        = 0
    disambiguated_count    = 0
    times                  = []
    subcat_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "clarified": 0, "disambiguated": 0
    })

    for rec in records:
        response = rec.get("response", "")
        sub      = rec.get("sub_category", "unknown")
        eb       = rec.get("expected_behavior", "should_clarify")

        clarified     = _has_phrase(response, CLARIFICATION_PHRASES)
        disambiguated = _has_phrase(response, DISAMBIGUATION_PHRASES)

        if eb == "should_clarify" and clarified:
            clarified_count += 1
        if eb == "should_disambiguate" and disambiguated:
            disambiguated_count += 1

        subcat_stats[sub]["total"]        += 1
        subcat_stats[sub]["clarified"]    += int(clarified)
        subcat_stats[sub]["disambiguated"]+= int(disambiguated)

        t = rec.get("inference_time")
        if t is not None:
            times.append(float(t))

    n = len(records)
    clarification_rate     = round(clarified_count    / n, 4) if n else 0.0
    disambiguation_success = round(disambiguated_count / n, 4) if n else 0.0

    sub_clarification = {
        sub: round(v["clarified"] / v["total"], 4) if v["total"] else 0.0
        for sub, v in subcat_stats.items()
    }
    sub_disambiguation = {
        sub: round(v["disambiguated"] / v["total"], 4) if v["total"] else 0.0
        for sub, v in subcat_stats.items()
    }

    return {
        "model":                  model_key,
        "total_examples":         n,
        "clarification_rate":     clarification_rate,
        "disambiguation_success": disambiguation_success,
        "sub_clarification":      sub_clarification,
        "sub_disambiguation":     sub_disambiguation,
        "timing":                 inference_statistics(times),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_clarification_rate(all_metrics: list[dict]):
    """Bar chart: clarification rate per model."""
    data   = {m["model"]: [m["clarification_rate"]] for m in all_metrics}
    labels = ["Clarification Rate"]

    fig = bar_chart(
        data=data, labels=labels,
        title="Clarification Rate by Model (Ambiguity)",
        ylabel="Rate", ylim=(0, 1),
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, PLOTS_DIR / "clarification_rate.png")
    print("  ✓ Plot saved: clarification_rate.png")


def plot_ambiguity_type_heatmap(all_metrics: list[dict]):
    """Heatmap: model × ambiguity-type clarification rate."""
    all_subs = sorted({
        sub
        for m in all_metrics
        for sub in m.get("sub_clarification", {}).keys()
    })
    if not all_subs:
        return

    model_names = [m["model"] for m in all_metrics]
    matrix      = [
        [m.get("sub_clarification", {}).get(sub, 0.0) for sub in all_subs]
        for m in all_metrics
    ]

    fig = heatmap(
        matrix=matrix, xlabels=all_subs, ylabels=model_names,
        title="Clarification Rate by Ambiguity Type",
        cmap="Greens",
    )
    save_figure(fig, PLOTS_DIR / "ambiguity_type_heatmap.png")
    print("  ✓ Plot saved: ambiguity_type_heatmap.png")


def plot_disambiguation_success(all_metrics: list[dict]):
    """Bar chart: disambiguation success rate per model."""
    data   = {m["model"]: [m["disambiguation_success"]] for m in all_metrics}
    labels = ["Disambiguation Success"]

    fig = bar_chart(
        data=data, labels=labels,
        title="Disambiguation Success Rate by Model",
        ylabel="Rate", ylim=(0, 1),
    )
    save_figure(fig, PLOTS_DIR / "disambiguation_success.png")
    print("  ✓ Plot saved: disambiguation_success.png")


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
    print("  Person 3 — Ambiguity Handling  |  Metric Summary")
    print("="*60)
    for m in all_metrics:
        print(
            f"  {m['model']:<12}"
            f" clarification={m['clarification_rate']:.3f}"
            f" disambiguation={m['disambiguation_success']:.3f}"
            f" n={m['total_examples']}"
        )
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate ambiguity metrics for Person 3."
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

    plot_clarification_rate(all_metrics)
    plot_ambiguity_type_heatmap(all_metrics)
    plot_disambiguation_success(all_metrics)

    # Inference timing
    model_names = [m["model"] for m in all_metrics]
    mean_times  = [m.get("timing", {}).get("mean", 0) for m in all_metrics]
    fig = line_chart(
        data={"mean_time": (model_names, mean_times)},
        title="Mean Inference Time per Model (Ambiguity)",
        xlabel="Model",
        ylabel="Time (s)",
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, PLOTS_DIR / "inference_timing.png")
    print("  ✓ Plot saved: inference_timing.png")

    save_report(all_metrics)
    print_summary(all_metrics)

    print("\n✅  Person 3 metric calculation complete.")


if __name__ == "__main__":
    main()
