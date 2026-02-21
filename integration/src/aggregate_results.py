"""
aggregate_results.py — Combine metrics.json from all 5 person folders.

Reads each person's results/metrics.json and produces a unified
integration/results/aggregated_metrics.json.

Usage:
    python integration/src/aggregate_results.py
"""

import json
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
INTEGRATION_DIR = Path(__file__).resolve().parent.parent

PERSON_FOLDERS = {
    "hallucination": PROJECT_ROOT / "person1_hallucination",
    "reasoning":     PROJECT_ROOT / "person2_reasoning",
    "ambiguity":     PROJECT_ROOT / "person3_ambiguity",
    "bias":          PROJECT_ROOT / "person4_bias",
    "context":       PROJECT_ROOT / "person5_context",
}

MODELS = ["gpt2", "llama3", "flan_t5"]


# ── Aggregation ─────────────────────────────────────────────────────────────

def load_person_metrics(dimension: str, folder: Path) -> dict | None:
    """Load a person's metrics.json."""
    path = folder / "results" / "metrics.json"
    if not path.exists():
        print(f"  ⚠ {dimension}: metrics.json not found — skipping")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate() -> dict:
    """
    Build a unified structure:
    {
        "by_dimension": { "hallucination": { "gpt2": {...}, ... }, ... },
        "by_model":     { "gpt2": { "hallucination": {...}, ... }, ... },
        "summary":      { "gpt2": { "avg_score": 0.72, ... }, ... }
    }
    """
    by_dimension = {}
    by_model = {m: {} for m in MODELS}

    for dim, folder in PERSON_FOLDERS.items():
        metrics = load_person_metrics(dim, folder)
        if metrics is None:
            continue
        by_dimension[dim] = metrics

        for model_key in MODELS:
            if model_key in metrics:
                by_model[model_key][dim] = metrics[model_key]

    # Compute per-model summary scores
    summary = {}
    for model_key in MODELS:
        dims = by_model[model_key]
        if not dims:
            continue

        # Collect the primary metric from each dimension
        scores = []
        for dim, m in dims.items():
            # Each dimension uses a different primary metric name
            # Normalize: lower-is-better metrics are inverted
            if dim == "hallucination":
                val = 1.0 - m.get("hallucination_rate", 0)   # invert
            elif dim == "reasoning":
                val = m.get("reasoning_accuracy", 0)
            elif dim == "ambiguity":
                val = m.get("clarification_rate", 0)
            elif dim == "bias":
                val = 1.0 - m.get("bias_score", 0)           # invert
            elif dim == "context":
                val = m.get("retrieval_accuracy", 0)
            else:
                val = 0
            scores.append(val)

        summary[model_key] = {
            "dimensions_available": len(dims),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "per_dimension_scores": {
                dim: round(s, 4)
                for dim, s in zip(dims.keys(), scores)
            },
        }

    return {
        "by_dimension": by_dimension,
        "by_model": by_model,
        "summary": summary,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Integration: Aggregating All Metrics")
    print("=" * 60)

    result = aggregate()

    # Save
    out_dir = INTEGRATION_DIR / "final_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregated_metrics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Aggregated metrics saved → {out_path}")

    # Print summary
    if result["summary"]:
        print("\n  Model Summary:")
        for model, info in result["summary"].items():
            print(f"    {model:10s}  avg_score={info['avg_score']}  dims={info['dimensions_available']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
