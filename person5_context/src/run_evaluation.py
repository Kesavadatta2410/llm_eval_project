"""
Person 5 – Context Length Evaluation
run_evaluation.py

Runs GPT-2, LLaMA-3, and FLAN-T5 on the context-length dataset and saves
raw responses + inference times to results/. Passes through context_length
and needle_position fields for downstream metric analysis.
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).parent
PERSON_DIR  = SRC_DIR.parent
PROJECT_DIR = PERSON_DIR.parent
DATA_DIR    = PERSON_DIR / "data"
RESULTS_DIR = PERSON_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))

from models.gpt2_wrapper    import GPT2Wrapper
from models.llama_wrapper   import LLaMAWrapper
from models.flan_t5_wrapper import FlanT5Wrapper

MODEL_MAP = {
    "gpt2":    GPT2Wrapper,
    "llama3":  LLaMAWrapper,
    "flan-t5": FlanT5Wrapper,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(split: str = "test") -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  ✓ Loaded {len(records)} records from {path.name}")
    return records


def already_done(output_path: Path) -> set:
    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (KeyError, json.JSONDecodeError):
                        pass
    return done


def run_model(model_key: str, records: list[dict], max_examples: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_key}_responses.jsonl"

    done_ids   = already_done(output_path)
    to_process = [r for r in records if r["id"] not in done_ids]

    if max_examples is not None:
        to_process = to_process[:max_examples]

    if not to_process:
        print(f"  ✓ {model_key}: nothing new to process.")
        return

    print(f"\n{'='*60}")
    print(f"  Model : {model_key}")
    print(f"  Total : {len(records)} | Done : {len(done_ids)} | New : {len(to_process)}")
    print(f"{'='*60}")

    model_cls = MODEL_MAP[model_key]
    model     = model_cls()

    with open(output_path, "a", encoding="utf-8") as f:
        for rec in tqdm(to_process, desc=f"  {model_key}"):
            result = model.generate_with_timing(rec["prompt"])
            output_rec = {
                "id":                rec["id"],
                "prompt":            rec["prompt"],
                "response":          result["response"],
                "inference_time":    result["inference_time"],
                "model":             model_key,
                "category":          rec.get("category", "context_length"),
                "sub_category":      rec.get("sub_category", ""),
                "ground_truth":      rec.get("ground_truth", ""),
                "expected_behavior": rec.get("expected_behavior", ""),
                "context_length":    rec.get("context_length"),
                "needle_position":   rec.get("needle_position"),
            }
            f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

    print(f"  ✓ Saved responses → {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run context-length evaluation for Person 5."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "llama3", "flan-t5"],
        choices=list(MODEL_MAP.keys()),
    )
    parser.add_argument("--split",        default="test", choices=["train", "test"])
    parser.add_argument("--max_examples", type=int, default=200)
    args = parser.parse_args()

    records = load_dataset(args.split)

    for model_key in args.models:
        run_model(model_key, records, args.max_examples)

    print("\n✅  Person 5 Context Length evaluation complete.")


if __name__ == "__main__":
    main()
