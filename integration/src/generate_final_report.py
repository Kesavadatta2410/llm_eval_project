"""
generate_final_report.py — Generate the executive summary markdown report.

Reads aggregated_metrics.json and significance_tests.json,
produces integration/final_report/executive_summary.md.

Usage:
    python integration/src/generate_final_report.py
"""

import json
from pathlib import Path
from datetime import datetime

# ── Paths ───────────────────────────────────────────────────────────────────
INTEGRATION_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR      = INTEGRATION_DIR / "final_report"

MODELS     = ["gpt2", "llama3", "flan_t5"]
DIMENSIONS = ["hallucination", "reasoning", "ambiguity", "bias", "context"]


# ── Report Generation ───────────────────────────────────────────────────────

def generate_report():
    # Load data
    agg_path = REPORT_DIR / "aggregated_metrics.json"
    sig_path = REPORT_DIR / "significance_tests.json"

    if not agg_path.exists():
        print("  ✗ aggregated_metrics.json not found. Run aggregate_results.py first.")
        return

    with open(agg_path, "r", encoding="utf-8") as f:
        aggregated = json.load(f)

    sig_results = []
    if sig_path.exists():
        with open(sig_path, "r", encoding="utf-8") as f:
            sig_results = json.load(f)

    summary = aggregated.get("summary", {})
    available_models = [m for m in MODELS if m in summary]

    # ── Build Markdown ──────────────────────────────────────────────────────
    lines = []
    lines.append("# Executive Summary: LLM Stress-Testing Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("This report summarizes the stress-testing evaluation of three LLMs ")
    lines.append("across five responsible AI dimensions: Hallucination, Reasoning, ")
    lines.append("Ambiguity, Bias, and Context Length.")
    lines.append("")

    # Model Ranking
    lines.append("## Model Ranking")
    lines.append("")
    lines.append("| Rank | Model | Average Score | Dimensions Evaluated |")
    lines.append("|------|-------|--------------|---------------------|")

    ranked = sorted(available_models, key=lambda m: summary[m]["avg_score"], reverse=True)
    for rank, model in enumerate(ranked, 1):
        info = summary[model]
        lines.append(f"| {rank} | {model} | {info['avg_score']:.4f} | {info['dimensions_available']} |")
    lines.append("")

    # Per-Dimension Breakdown
    lines.append("## Per-Dimension Scores")
    lines.append("")

    # Collect all available dimensions
    all_dims = sorted(set(
        dim for m in available_models
        for dim in summary[m].get("per_dimension_scores", {})
    ))

    if all_dims:
        header = "| Model | " + " | ".join(d.capitalize() for d in all_dims) + " |"
        sep    = "|-------|" + "|".join(["--------"] * len(all_dims)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in available_models:
            scores = summary[model].get("per_dimension_scores", {})
            row = f"| {model} | "
            row += " | ".join(f"{scores.get(d, 0):.4f}" for d in all_dims)
            row += " |"
            lines.append(row)
        lines.append("")

    # Statistical Significance
    if sig_results:
        lines.append("## Statistical Significance")
        lines.append("")
        lines.append("| Comparison | t-statistic | p-value | Significant (α=0.05) |")
        lines.append("|-----------|------------|---------|---------------------|")

        for r in sig_results:
            sig = "✓ Yes" if r.get("t_significant") else "✗ No"
            lines.append(f"| {r['comparison']} | {r['t_statistic']:+.4f} | {r['t_p_value']:.4f} | {sig} |")
        lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")

    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        lines.append(f"1. **Best overall model:** {best} (avg score: {summary[best]['avg_score']:.4f})")
        lines.append(f"2. **Worst overall model:** {worst} (avg score: {summary[worst]['avg_score']:.4f})")
        lines.append("")

        # Find strongest/weakest dimensions per model
        for model in available_models:
            scores = summary[model].get("per_dimension_scores", {})
            if scores:
                best_dim = max(scores, key=scores.get)
                worst_dim = min(scores, key=scores.get)
                lines.append(f"- **{model}:** Strongest in _{best_dim}_ ({scores[best_dim]:.4f}), "
                             f"weakest in _{worst_dim}_ ({scores[worst_dim]:.4f})")
        lines.append("")

    # Visualizations Reference
    lines.append("## Visualizations")
    lines.append("")
    lines.append("| Chart | Path |")
    lines.append("|-------|------|")
    lines.append("| Radar Comparison | `../visualizations/master_radar_chart.png` |")
    lines.append("| Model Ranking | `../visualizations/model_ranking.png` |")
    lines.append("| Failure Mode Matrix | `../visualizations/failure_mode_matrix.png` |")
    lines.append("")

    lines.append("---")
    lines.append("*Report generated automatically by `generate_final_report.py`*")
    lines.append("")

    # ── Write ───────────────────────────────────────────────────────────────
    report_path = REPORT_DIR / "executive_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Executive summary saved → {report_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Integration: Generating Final Report")
    print("=" * 60)

    generate_report()

    print("\n" + "=" * 60)
    print("  Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
