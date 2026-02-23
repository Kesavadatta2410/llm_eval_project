"""
extract_data.py — Person 2: Reasoning & Logic

Downloads BigBench tasks (logical_deduction, tracking_shuffled_objects, word_sorting),
fetches Natural Instructions reasoning tasks, and supplements with handcrafted probes.
Outputs train.jsonl and test.jsonl in person2_reasoning/data/.
"""

import json, hashlib, random
from pathlib import Path
import requests

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent          # person2_reasoning/
DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT.parent / "data" / "bigbench" / ".cache"
NI_CACHE  = ROOT.parent / "data" / "natural_instructions" / ".cache"

BIGBENCH_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/"
    "bigbench/benchmark_tasks/{task_name}/task.json"
)
NI_BASE_URL = (
    "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/{task_name}.json"
)

SEED = 42
random.seed(SEED)

# No max samples cap — extract all available data
NI_PER_TASK = None   # None = all instances


# ── Generic Helpers ─────────────────────────────────────────────────────────

def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(records)} examples → {path.name}")


# ── BigBench Helpers ─────────────────────────────────────────────────────────

def download_bigbench_task(task_name: str) -> dict | None:
    """Download a BigBench task.json with local file caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{task_name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = BIGBENCH_URL.format(task_name=task_name)
    print(f"  ↓ Downloading BigBench:{task_name} …")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return None


def extract_bigbench_examples(task_data: dict) -> list:
    records = []
    for ex in task_data.get("examples", []):
        inp    = ex.get("input", "")
        target = ex.get("target", "")
        if isinstance(target, list):
            target = target[0] if target else ""
        choices = []
        if "target_scores" in ex:
            choices = list(ex["target_scores"].keys())
            target  = max(ex["target_scores"], key=ex["target_scores"].get)
        records.append({"prompt": inp, "ground_truth": target, "choices": choices})
    return records


# ── Natural Instructions Helpers ─────────────────────────────────────────────

def download_ni_task(task_name: str) -> dict | None:
    """Download a Natural-Instructions task JSON with local file caching."""
    NI_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = NI_CACHE / f"{task_name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = NI_BASE_URL.format(task_name=task_name)
    print(f"  ↓ Downloading NI:{task_name} …")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return None


# ── BigBench Tasks ───────────────────────────────────────────────────────────

BB_TASKS = ["logical_deduction", "tracking_shuffled_objects", "word_sorting"]


def build_bigbench_records() -> list:
    all_records = []
    for task_name in BB_TASKS:
        data = download_bigbench_task(task_name)
        if data is None:
            continue
        for ex in extract_bigbench_examples(data):
            all_records.append({
                "id":                make_id(ex["prompt"]),
                "prompt":            ex["prompt"],
                "category":          "reasoning",
                "sub_category":      task_name,
                "expected_behavior": "should_answer_correctly",
                "difficulty":        "medium",
                "ground_truth":      ex["ground_truth"],
                "choices":           ex["choices"],
                "source":            "bigbench",
            })
    return all_records


# ── Natural Instructions Tasks ───────────────────────────────────────────────
# Selected for reasoning relevance:
#   task1290 – Ropes reasoning           (multi-step inference)
#   task1503 – Deductive logic           (formal deduction)
#   task275  – Combination lock          (state-tracking reasoning)
#   task1345 – Counterfactual inference  (if-then reasoning)

NI_TASKS = {
    "task1290_xnli_mlm": (
        "textual_entailment", "should_answer_correctly"),
    "task1503_hatexplain_classification": (
        "text_classification", "should_answer_correctly"),
    "task275_enhanced_wsc_paraphrase_generation": (
        "paraphrase_reasoning", "should_answer_correctly"),
    "task1345_glue_sts-b_similarity_classification": (
        "semantic_similarity", "should_answer_correctly"),
}


def build_ni_records() -> list:
    all_records = []
    for task_name, (sub_cat, expected_behavior) in NI_TASKS.items():
        data = download_ni_task(task_name)
        if data is None:
            continue
        definition = " ".join(data.get("Definition", [""])).strip()
        instances  = data.get("Instances", [])
        random.shuffle(instances)
        for inst in (instances if NI_PER_TASK is None else instances[:NI_PER_TASK]):
            inp    = inst.get("input", "").strip()
            output = inst.get("output", [])
            gt     = output[0] if isinstance(output, list) and output else str(output)
            if not inp:
                continue
            prompt = f"[Task: {definition}]\n\nInput: {inp}"
            all_records.append({
                "id":                make_id(inp),
                "prompt":            prompt,
                "category":          "reasoning",
                "sub_category":      sub_cat,
                "expected_behavior": expected_behavior,
                "difficulty":        "medium",
                "ground_truth":      gt,
                "choices":           [],
                "source":            "natural_instructions",
                "ni_task":           task_name,
            })
    return all_records


# ── Handcrafted Probes ───────────────────────────────────────────────────────

HANDCRAFTED = [
    {"prompt": "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Yes — by transitivity of the syllogism.",
     "sub_category": "syllogistic_reasoning", "difficulty": "easy"},
    {"prompt": "A bat and ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "$0.05",
     "sub_category": "arithmetic_reasoning", "difficulty": "medium"},
    {"prompt": "In what order should you put on socks, shoes, and then pants? Briefly reason it out.",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Pants first, then socks, then shoes.",
     "sub_category": "commonsense_ordering", "difficulty": "easy"},
    {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "5 minutes.",
     "sub_category": "rate_reasoning", "difficulty": "medium"},
    {"prompt": "A farmer has 17 sheep; all but 9 die. How many are left?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "9 sheep.",
     "sub_category": "language_ambiguity_math", "difficulty": "easy"},
    {"prompt": "Sort these numbers in descending order: 3, 1, 4, 1, 5, 9, 2, 6.",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "9, 6, 5, 4, 3, 2, 1, 1",
     "sub_category": "sorting", "difficulty": "easy"},
    {"prompt": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "42 (differences increase by 2 each time).",
     "sub_category": "sequence_completion", "difficulty": "medium"},
]


def build_handcrafted_records() -> list:
    return [{
        "id":                make_id(p["prompt"]),
        "prompt":            p["prompt"],
        "category":          "reasoning",
        "sub_category":      p["sub_category"],
        "expected_behavior": p["expected_behavior"],
        "difficulty":        p["difficulty"],
        "ground_truth":      p["ground_truth"],
        "choices":           [],
        "source":            "handcrafted",
    } for p in HANDCRAFTED]


# ── Build & Save ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Person 2: Extracting Reasoning Dataset")
    print("=" * 60)

    bb_records = build_bigbench_records()
    ni_records = build_ni_records()
    hc_records = build_handcrafted_records()

    print(f"\n  Sources → BigBench: {len(bb_records)}  |  NI: {len(ni_records)}"
          f"  |  Handcrafted: {len(hc_records)}")

    all_records = bb_records + ni_records + hc_records
    random.shuffle(all_records)

    # Deduplication
    seen, deduped = set(), []
    for r in all_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    all_records = deduped  # no cap — use all deduplicated records

    split = int(len(all_records) * 0.8)
    save_jsonl(all_records[:split], DATA_DIR / "train.jsonl")
    save_jsonl(all_records[split:], DATA_DIR / "test.jsonl")

    print(f"\n  Total: {len(all_records)}  |  Train: {split}  |  Test: {len(all_records)-split}")
    print("=" * 60)


if __name__ == "__main__":
    main()
