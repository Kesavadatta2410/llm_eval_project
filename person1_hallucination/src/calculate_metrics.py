"""
calculate_metrics.py — Person 1: Calculate hallucination-specific metrics.

Reads model responses from person1_hallucination/results/,
computes hallucination rate, factuality score, confidence analysis,
and generates plots in person1_hallucination/visualizations/.

Usage:
    python person1_hallucination/src/calculate_metrics.py
"""

import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hallucination_detector import HallucinationDetector
from evaluation.metrics import (
    compute_accuracy,
    fuzzy_match,
    inference_statistics,
)
from evaluation.visualization_utils import bar_chart, heatmap, line_chart, save_figure

# ── Paths ───────────────────────────────────────────────────────────────────
PERSON_DIR  = Path(__file__).resolve().parent.parent
RESULTS_DIR = PERSON_DIR / "results"
VIZ_DIR     = PERSON_DIR / "visualizations"

RESPONSE_FILES = {
    "gpt2":    "gpt2_responses.jsonl",
    "llama3":  "llama_responses.jsonl",
    "flan_t5": "flan_t5_responses.jsonl",
}


# ── Load responses ──────────────────────────────────────────────────────────

def load_responses(model_key: str) -> list[dict]:
    path = RESULTS_DIR / RESPONSE_FILES[model_key]
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ── Metric Calculation ──────────────────────────────────────────────────────

def calculate_model_metrics(model_key: str, records: list[dict]) -> dict:
    """Compute all hallucination metrics for one model."""
    if not records:
        return {}

    detector = HallucinationDetector()

    # Hallucination detection
    hal_results = detector.batch_check(records)
    hallucination_rate = sum(1 for r in hal_results if r["hallucinated"]) / len(hal_results)

    avg_confidence  = sum(r["confidence_score"] for r in hal_results) / len(hal_results)
    avg_uncertainty = sum(r["uncertainty_score"] for r in hal_results) / len(hal_results)

    # Factuality: fuzzy-match accuracy on answerable questions
    answerable = [r for r in records if r.get("expected_behavior") == "should_answer_correctly"]
    if answerable:
        factuality_hits = sum(
            1 for r in answerable
            if fuzzy_match(r["response"], r.get("ground_truth", ""))
        )
        factuality_score = factuality_hits / len(answerable)
    else:
        factuality_score = None

    # Hallucination rate by sub-category
    by_subcategory = {}
    for rec, hal in zip(records, hal_results):
        sub = rec.get("sub_category", "unknown")
        by_subcategory.setdefault(sub, []).append(hal["hallucinated"])

    subcategory_rates = {
        sub: sum(vals) / len(vals)
        for sub, vals in by_subcategory.items()
    }

    # Inference timing
    times = [r.get("inference_time", 0) for r in records]
    timing = inference_statistics(times)

    return {
        "model": model_key,
        "num_examples": len(records),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_confidence_score": round(avg_confidence, 4),
        "avg_uncertainty_score": round(avg_uncertainty, 4),
        "factuality_score": round(factuality_score, 4) if factuality_score is not None else None,
        "subcategory_rates": subcategory_rates,
        "inference_time": timing,
    }


# ── Visualization ───────────────────────────────────────────────────────────

def generate_plots(all_metrics: dict):
    """Generate hallucination-specific visualizations."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    models = list(all_metrics.keys())
    if not models:
        return

    # 1. Hallucination rate bar chart
    fig = bar_chart(
        data={m: [all_metrics[m]["hallucination_rate"]] for m in models},
        labels=["Hallucination Rate"],
        title="Hallucination Rate by Model",
        ylabel="Rate (lower is better)",
    )
    save_figure(fig, VIZ_DIR / "hallucination_rate_bar.png")

    # 2. Factuality comparison
    fact_models = [m for m in models if all_metrics[m].get("factuality_score") is not None]
    if fact_models:
        fig = bar_chart(
            data={m: [all_metrics[m]["factuality_score"]] for m in fact_models},
            labels=["Factuality Score"],
            title="Factuality Score by Model",
            ylabel="Score (higher is better)",
        )
        save_figure(fig, VIZ_DIR / "factuality_comparison.png")

    # 3. Subcategory heatmap
    all_subs = sorted(set(
        sub for m in models for sub in all_metrics[m].get("subcategory_rates", {})
    ))
    if all_subs and len(models) > 0:
        matrix = []
        for m in models:
            row = [all_metrics[m].get("subcategory_rates", {}).get(s, 0) for s in all_subs]
            matrix.append(row)
        fig = heatmap(
            matrix=matrix,
            xlabels=all_subs,
            ylabels=models,
            title="Hallucination Rate by Sub-Category",
            cmap="Reds",
        )
        save_figure(fig, VIZ_DIR / "subcategory_heatmap.png")

    # 4. Inference timing line chart
    timing_data = {
        m: (list(range(1, 2)), [all_metrics[m]["inference_time"].get("mean", 0)])
        for m in models
    }
    model_names = list(models)
    mean_times  = [all_metrics[m]["inference_time"].get("mean", 0) for m in models]
    fig = line_chart(
        data={"mean_time": (model_names, mean_times)},
        title="Mean Inference Time per Model (Hallucination)",
        xlabel="Model",
        ylabel="Time (s)",
    )
    save_figure(fig, VIZ_DIR / "inference_timing.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Person 1: Calculating Hallucination Metrics")
    print("=" * 60)

    all_metrics = {}

    for model_key in RESPONSE_FILES:
        records = load_responses(model_key)
        if not records:
            print(f"  ⚠ No responses for {model_key} — skipping")
            continue
        print(f"\n  ▶ {model_key}: {len(records)} responses")
        metrics = calculate_model_metrics(model_key, records)
        all_metrics[model_key] = metrics

        print(f"    Hallucination Rate : {metrics['hallucination_rate']}")
        print(f"    Factuality Score   : {metrics.get('factuality_score', 'N/A')}")
        print(f"    Avg Confidence     : {metrics['avg_confidence_score']}")
        print(f"    Avg Uncertainty    : {metrics['avg_uncertainty_score']}")

    if all_metrics:
        # Save metrics.json
        out_path = RESULTS_DIR / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Metrics saved → {out_path.name}")

        # Generate plots
        print("\n  Generating visualizations…")
        generate_plots(all_metrics)

    print("\n" + "=" * 60)
    print("  Hallucination metrics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
