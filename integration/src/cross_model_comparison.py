"""
cross_model_comparison.py — Statistical comparison across all models and dimensions.

Reads aggregated_metrics.json and produces:
  1. Radar chart comparing models across all dimensions
  2. Model ranking table
  3. Statistical significance tests (paired t-test, Wilcoxon)

Usage:
    python integration/src/cross_model_comparison.py
"""

import json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.visualization_utils import radar_chart, bar_chart, heatmap, save_figure
from scipy import stats

# ── Paths ───────────────────────────────────────────────────────────────────
INTEGRATION_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR      = INTEGRATION_DIR / "final_report"
VIZ_DIR         = INTEGRATION_DIR / "visualizations"

DIMENSIONS = ["hallucination", "reasoning", "ambiguity", "bias", "context"]
MODELS     = ["gpt2", "llama3", "flan_t5"]


# ── Load Aggregated Data ────────────────────────────────────────────────────

def load_aggregated() -> dict:
    path = REPORT_DIR / "aggregated_metrics.json"
    if not path.exists():
        print("  ✗ aggregated_metrics.json not found. Run aggregate_results.py first.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Statistical Tests ───────────────────────────────────────────────────────

def run_significance_tests(aggregated: dict) -> list[dict]:
    """Paired comparisons between all model pairs using available dimension scores."""
    results = []
    summary = aggregated.get("summary", {})
    model_keys = [m for m in MODELS if m in summary]

    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m1, m2 = model_keys[i], model_keys[j]
            scores1 = list(summary[m1].get("per_dimension_scores", {}).values())
            scores2 = list(summary[m2].get("per_dimension_scores", {}).values())

            min_len = min(len(scores1), len(scores2))
            if min_len < 2:
                continue

            s1 = np.array(scores1[:min_len])
            s2 = np.array(scores2[:min_len])

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(s1, s2)

            # Wilcoxon signed-rank (needs n >= 6 ideally, but we try)
            try:
                w_stat, w_pval = stats.wilcoxon(s1, s2)
            except ValueError:
                w_stat, w_pval = float("nan"), float("nan")

            results.append({
                "comparison": f"{m1} vs {m2}",
                "t_statistic": round(float(t_stat), 4),
                "t_p_value": round(float(t_pval), 4),
                "t_significant": float(t_pval) < 0.05,
                "wilcoxon_statistic": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                "wilcoxon_p_value": round(float(w_pval), 4) if not np.isnan(w_pval) else None,
            })

    return results


# ── Visualizations ──────────────────────────────────────────────────────────

def generate_comparison_plots(aggregated: dict):
    """Generate cross-model comparison visualizations."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    summary = aggregated.get("summary", {})

    available_models = [m for m in MODELS if m in summary]
    if not available_models:
        print("  ⚠ No model data available for plotting")
        return

    # 1. Radar chart
    available_dims = sorted(set(
        dim for m in available_models
        for dim in summary[m].get("per_dimension_scores", {})
    ))
    if available_dims:
        radar_data = {}
        for m in available_models:
            scores = summary[m].get("per_dimension_scores", {})
            radar_data[m] = [scores.get(d, 0) for d in available_dims]

        fig = radar_chart(radar_data, available_dims, title="Model Comparison Across Dimensions")
        save_figure(fig, VIZ_DIR / "master_radar_chart.png")

    # 2. Model ranking bar chart
    fig = bar_chart(
        data={m: [summary[m]["avg_score"]] for m in available_models},
        labels=["Overall Average Score"],
        title="Model Ranking (Overall)",
        ylabel="Average Normalized Score",
    )
    save_figure(fig, VIZ_DIR / "model_ranking.png")

    # 3. Failure mode matrix (heatmap)
    if available_dims:
        matrix = []
        for m in available_models:
            scores = summary[m].get("per_dimension_scores", {})
            # Show failure (1 - score) so higher = worse
            row = [round(1 - scores.get(d, 0), 4) for d in available_dims]
            matrix.append(row)

        fig = heatmap(
            matrix=matrix,
            xlabels=available_dims,
            ylabels=available_models,
            title="Failure Mode Matrix (higher = worse)",
            cmap="Reds",
        )
        save_figure(fig, VIZ_DIR / "failure_mode_matrix.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Integration: Cross-Model Comparison")
    print("=" * 60)

    aggregated = load_aggregated()
    if not aggregated:
        return

    # Statistical tests
    print("\n  Running significance tests…")
    sig_results = run_significance_tests(aggregated)
    for r in sig_results:
        sig_marker = "✓" if r.get("t_significant") else "✗"
        print(f"    {r['comparison']:25s}  t={r['t_statistic']:+.4f}  p={r['t_p_value']:.4f}  {sig_marker}")

    # Save significance results
    sig_path = REPORT_DIR / "significance_tests.json"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(sig_results, f, indent=2)
    print(f"\n  ✓ Significance tests saved → {sig_path.name}")

    # Generate plots
    print("\n  Generating comparison visualizations…")
    generate_comparison_plots(aggregated)

    print("\n" + "=" * 60)
    print("  Cross-model comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
