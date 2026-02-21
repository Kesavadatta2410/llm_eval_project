"""
run_evaluation.py — Person 1: Run all 3 models on the hallucination dataset.

Loads train.jsonl and test.jsonl, runs GPT-2 / LLaMA-3 / FLAN-T5 inference,
and saves raw responses to person1_hallucination/results/.

Usage:
    python person1_hallucination/src/run_evaluation.py
    python person1_hallucination/src/run_evaluation.py --models gpt2 flan_t5
    python person1_hallucination/src/run_evaluation.py --max-examples 20
"""

import json, argparse, sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_model
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────
PERSON_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR     = PERSON_DIR / "data"
RESULTS_DIR  = PERSON_DIR / "results"

MODEL_KEYS = ["gpt2", "llama3", "flan_t5"]
RESPONSE_FILES = {
    "gpt2":    "gpt2_responses.jsonl",
    "llama3":  "llama_responses.jsonl",
    "flan_t5": "flan_t5_responses.jsonl",
}


# ── Load Dataset ────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    """Load both train and test JSONL into a single list."""
    records = []
    for split in ["train.jsonl", "test.jsonl"]:
        path = DATA_DIR / split
        if not path.exists():
            print(f"  ⚠ {path.name} not found — run extract_data.py first")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


# ── Run Inference ───────────────────────────────────────────────────────────

def run_model(model_key: str, records: list[dict], max_examples: int | None = None):
    """Run a model on the dataset and save responses."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / RESPONSE_FILES[model_key]

    # Resume support: check which IDs already have responses
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line).get("id", ""))

    to_process = [r for r in records if r["id"] not in existing_ids]
    if max_examples:
        to_process = to_process[:max_examples]

    if not to_process:
        print(f"  ✓ {model_key}: all examples already processed ({len(existing_ids)} existing)")
        return

    print(f"\n  Loading {model_key}…")
    model = load_model(model_key)

    print(f"  Running inference on {len(to_process)} examples…")
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in tqdm(to_process, desc=f"  {model_key}"):
            result = model.generate_with_timing(rec["prompt"])
            output_rec = {
                "id": rec["id"],
                "prompt": rec["prompt"],
                "expected_behavior": rec.get("expected_behavior", ""),
                "ground_truth": rec.get("ground_truth", ""),
                "model": model_key,
                "response": result["response"],
                "inference_time": result["inference_time"],
            }
            f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

    total = len(existing_ids) + len(to_process)
    print(f"  ✓ {model_key}: {total} responses saved → {output_path.name}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Person 1: Run hallucination evaluation")
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS,
                        choices=MODEL_KEYS, help="Models to evaluate")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of new examples per model")
    args = parser.parse_args()

    print("=" * 60)
    print("  Person 1: Hallucination Evaluation — Inference")
    print("=" * 60)

    records = load_dataset()
    if not records:
        print("  ✗ No data found. Run extract_data.py first.")
        return

    print(f"  Loaded {len(records)} examples")

    for model_key in args.models:
        run_model(model_key, records, args.max_examples)

    print("\n" + "=" * 60)
    print("  Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
