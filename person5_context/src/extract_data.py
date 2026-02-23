"""
extract_data.py — Person 5: Context & Long-form Understanding

Downloads BigBench tasks (strategyqa, understanding_fables, timedial),
fetches Natural Instructions context-understanding tasks, and supplements with handcrafted probes.
Outputs train.jsonl and test.jsonl in person5_context/data/.
"""

import json, hashlib, random
from pathlib import Path
import requests

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent          # person5_context/
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

BB_TASKS = [
    "strategyqa",
    "understanding_fables",
    "timedial",
]


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
                "category":          "context",
                "sub_category":      task_name,
                "expected_behavior": "should_answer_correctly",
                "difficulty":        "medium",
                "ground_truth":      ex["ground_truth"],
                "choices":           ex["choices"],
                "source":            "bigbench",
            })
    return all_records


# ── Natural Instructions Tasks ───────────────────────────────────────────────
# Selected for context/comprehension relevance:
#   task190  – QuALITY long-document QA
#   task1659 – Sentence ordering (discourse coherence)
#   task613  – Contextual question generation
#   task1343 – RACE reading comprehension

NI_TASKS = {
    "task190_snli_hypothesis_classification": (
        "textual_entailment", "should_answer_correctly"),
    "task1659_title_generation": (
        "title_generation", "should_answer_correctly"),
    "task613_politeness_style_transfer": (
        "style_understanding", "should_answer_correctly"),
    "task1343_food_review_helpfulness_classification": (
        "helpfulness_classification", "should_answer_correctly"),
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
                "category":          "context",
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
    {"prompt": "Read this passage: 'The dog ran after the car until it stopped.' What stopped — the dog or the car?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "Ambiguous — 'it' could refer to either the dog or the car.",
     "sub_category": "pronoun_resolution", "difficulty": "medium"},
    {"prompt": "In a story, the hero wins but feels sad. What might this imply about the story's theme?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Pyrrhic victory, moral cost of winning, or that outward success doesn't guarantee happiness.",
     "sub_category": "thematic_inference", "difficulty": "hard"},
    {"prompt": "A user says: 'This product is just okay.' Is this positive, negative, or neutral feedback?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Neutral to mildly negative — 'just okay' implies under-satisfaction.",
     "sub_category": "sentiment_in_context", "difficulty": "easy"},
    {"prompt": "Context: You are a polite customer service agent. User: 'I WANT A REFUND NOW!' How do you respond?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "De-escalate calmly, apologize for inconvenience, ask for order details, and offer the refund process.",
     "sub_category": "role_context_following", "difficulty": "medium"},
    {"prompt": "A poem about autumn that doesn't mention any season. How would you identify it?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Through imagery: falling leaves, cooling air, harvest themes, shorter days, muted colours.",
     "sub_category": "implied_context", "difficulty": "hard"},
    {"prompt": "Two people are arguing. Person A says 'You always leave the milk out.' Person B says 'I did it once.' Who is likely correct?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "Person B is likely more factually accurate; 'always' is an absolute that is often an exaggeration in arguments.",
     "sub_category": "conversational_context", "difficulty": "medium"},
    {"prompt": "A news article title: 'Scientists Discover Water on Mars — Hope Grows'. What is the implicit message?",
     "expected_behavior": "should_answer_correctly",
     "ground_truth": "The discovery may support the possibility of past or present life on Mars.",
     "sub_category": "pragmatic_inference", "difficulty": "medium"},
]


def build_handcrafted_records() -> list:
    return [{
        "id":                make_id(p["prompt"]),
        "prompt":            p["prompt"],
        "category":          "context",
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
    print("  Person 5: Extracting Context & Long-form Understanding Dataset")
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
