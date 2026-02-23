"""
extract_data.py — Person 3: Ambiguity & Disambiguation

Downloads BigBench tasks (disambiguation_qa, winowhy, contextual_parametric_knowledge_conflicts),
fetches Natural Instructions ambiguity/WSD tasks, and supplements with handcrafted probes.
Outputs train.jsonl and test.jsonl in person3_ambiguity/data/.
"""

import json, hashlib, random
from pathlib import Path
import requests

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent          # person3_ambiguity/
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
    "disambiguation_qa",
    "winowhy",
    "contextual_parametric_knowledge_conflicts",
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
                "category":          "ambiguity",
                "sub_category":      task_name,
                "expected_behavior": "should_disambiguate",
                "difficulty":        "medium",
                "ground_truth":      ex["ground_truth"],
                "choices":           ex["choices"],
                "source":            "bigbench",
            })
    return all_records


# ── Natural Instructions Tasks ───────────────────────────────────────────────
# Selected for ambiguity/WSD relevance:
#   task326  – WSD (word sense disambiguation)
#   task648  – WinoGrande pronoun resolution
#   task569  – Winograd schema resolution
#   task330  – Identify unclear pronoun references

NI_TASKS = {
    "task326_jigsaw_classification_disagree": (
        "disagreement_detection", "should_answer_correctly"),
    "task648_answer_generation": (
        "answer_generation", "should_answer_correctly"),
    "task569_recipes_nlg_ner": (
        "entity_identification", "should_answer_correctly"),
    "task330_gap_classification": (
        "pronoun_disambiguation", "should_disambiguate"),
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
                "category":          "ambiguity",
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
    {"prompt": "The bank was steep. What does 'bank' refer to here?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "River bank (embankment).",
     "sub_category": "lexical_ambiguity", "difficulty": "easy"},
    {"prompt": "Can you book a flight and book a table at the same time?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "Yes — 'book' here means 'reserve' in both cases.",
     "sub_category": "lexical_ambiguity", "difficulty": "easy"},
    {"prompt": "I saw the man with the telescope. Who had the telescope?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "Ambiguous — either the speaker used it to see the man, or the man was carrying it.",
     "sub_category": "syntactic_ambiguity", "difficulty": "medium"},
    {"prompt": "Mary and Sue went to the store. She bought milk. Who bought milk?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "Ambiguous — 'she' could refer to either Mary or Sue.",
     "sub_category": "pronoun_ambiguity", "difficulty": "medium"},
    {"prompt": "Time flies like an arrow; fruit flies like a banana. Explain the ambiguity.",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "'Flies' shifts from verb to noun across clauses; 'like' shifts from adverb to verb.",
     "sub_category": "structural_ambiguity", "difficulty": "hard"},
    {"prompt": "I never said she stole the money. Stress the word 'never' then 'she'. Does the meaning change?",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "Yes — stress shifts imply different implicit claims.",
     "sub_category": "prosodic_ambiguity", "difficulty": "hard"},
    {"prompt": "Visiting relatives can be boring. Give two interpretations.",
     "expected_behavior": "should_disambiguate",
     "ground_truth": "1) Going to visit relatives is boring. 2) Relatives who visit can be boring.",
     "sub_category": "syntactic_ambiguity", "difficulty": "medium"},
]


def build_handcrafted_records() -> list:
    return [{
        "id":                make_id(p["prompt"]),
        "prompt":            p["prompt"],
        "category":          "ambiguity",
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
    print("  Person 3: Extracting Ambiguity Dataset")
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
