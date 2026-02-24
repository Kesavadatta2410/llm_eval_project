"""
Person 3 – Ambiguity Handling
extract_data.py

Pulls data from BigBench (disambiguation_qa, question_ambiguity, winowhy),
Natural Instructions, and handcrafted ambiguous probes.  Outputs train.jsonl
and test.jsonl under person3_ambiguity/data/.
"""

import hashlib
import json
import random
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SRC_DIR    = Path(__file__).parent
PERSON_DIR = SRC_DIR.parent
DATA_DIR   = PERSON_DIR / "data"
CACHE_DIR  = PERSON_DIR / ".cache"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80

BB_TASKS = [
    "disambiguation_qa",
    "question_ambiguity",
    "winowhy",
]

NI_TASKS = [
    "task401_numeric_fused_head_identification",
    "task677_ollie_sentence_answer_generation",
    "task760_msr_sqa_question_answer_generation",
    "task1159_bard_analogical_reasoning_causation",
]

NI_BASE_URL = (
    "https://raw.githubusercontent.com/allenai/natural-instructions"
    "/master/tasks/{task_name}.json"
)

BB_BASE_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main"
    "/bigbench/benchmark_tasks/{task_name}/task.json"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(records)} examples → {path.name}")


def _fetch_url(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception as e:
        print(f"  ! Download failed ({url}): {e}")
        return None


def download_bigbench_task(task_name: str) -> dict | None:
    cache = CACHE_DIR / f"bb_{task_name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = _fetch_url(BB_BASE_URL.format(task_name=task_name))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    return data


def extract_bigbench_examples(data: dict) -> list[dict]:
    examples = []
    for ex in data.get("examples", []):
        if isinstance(ex.get("input"), str) and ex["input"].strip():
            choices    = ex.get("target_scores", {})
            gt_choices = [k for k, v in choices.items() if v == max(choices.values())] if choices else []
            ground_truth = gt_choices[0] if gt_choices else ex.get("target", "")
            examples.append({
                "prompt":       ex["input"].strip(),
                "ground_truth": ground_truth,
                "choices":      list(choices.keys()),
            })
    return examples


def download_ni_task(task_name: str) -> dict | None:
    cache = CACHE_DIR / f"ni_{task_name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = _fetch_url(NI_BASE_URL.format(task_name=task_name))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    return data


# ---------------------------------------------------------------------------
# BigBench records
# ---------------------------------------------------------------------------

def build_bigbench_records() -> list:
    all_records = []
    for task_name in BB_TASKS:
        print(f"  Fetching BigBench task: {task_name}")
        data = download_bigbench_task(task_name)
        if data is None:
            print(f"    ↳ Skipped (could not download)")
            continue
        examples = extract_bigbench_examples(data)
        for ex in examples:
            all_records.append({
                "id":                make_id(ex["prompt"]),
                "prompt":            ex["prompt"],
                "category":          "ambiguity",
                "sub_category":      task_name,
                "expected_behavior": "should_clarify"
                                     if task_name == "question_ambiguity"
                                     else "should_disambiguate",
                "difficulty":        "medium",
                "ground_truth":      ex["ground_truth"],
                "choices":           ex["choices"],
                "source":            "bigbench",
            })
    print(f"  BigBench total: {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Natural Instructions records
# ---------------------------------------------------------------------------

def build_ni_records() -> list:
    all_records = []
    for task_name in NI_TASKS:
        print(f"  Fetching NI task: {task_name}")
        data = download_ni_task(task_name)
        if data is None:
            print(f"    ↳ Skipped (could not download)")
            continue
        definition = data.get("Definition", [""])[0]
        instances  = data.get("Instances", [])
        for inst in instances:
            inp = inst.get("input", "").strip()
            if not inp:
                continue
            outputs = inst.get("output", [])
            gt      = outputs[0] if outputs else ""
            prompt  = f"{definition}\n\nInput: {inp}" if definition else inp
            all_records.append({
                "id":                make_id(prompt),
                "prompt":            prompt,
                "category":          "ambiguity",
                "sub_category":      task_name,
                "expected_behavior": "should_clarify",
                "difficulty":        "medium",
                "ground_truth":      gt,
                "choices":           [],
                "source":            "natural_instructions",
            })
    print(f"  NI total: {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Handcrafted ambiguity probes
# ---------------------------------------------------------------------------

def build_handcrafted_records() -> list:
    probes = [
        # Pronoun ambiguity
        {
            "prompt": (
                "Alice told Bob that she was going to be late. Who was going to be late?"
            ),
            "ground_truth":      "Alice",
            "sub_category":      "pronoun_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": (
                "The trophy didn't fit in the suitcase because it was too big. "
                "What was too big — the trophy or the suitcase?"
            ),
            "ground_truth":      "the trophy",
            "sub_category":      "pronoun_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": (
                "The lawyer asked the witness a question, and she seemed nervous. "
                "Who seemed nervous?"
            ),
            "ground_truth":      "the witness",
            "sub_category":      "pronoun_ambiguity",
            "expected_behavior": "should_clarify",
        },
        # Lexical ambiguity
        {
            "prompt": "I saw the man with the telescope. Who had the telescope?",
            "ground_truth":      "ambiguous",
            "sub_category":      "lexical_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": "Can you give me a hand? What is being requested?",
            "ground_truth":      "help",
            "sub_category":      "lexical_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": (
                "The bank was steep. What kind of bank is being described — "
                "a financial institution or a riverside bank?"
            ),
            "ground_truth":      "a riverside bank",
            "sub_category":      "lexical_ambiguity",
            "expected_behavior": "should_clarify",
        },
        # Underspecified instructions
        {
            "prompt": "Move the box from the table to the left.",
            "ground_truth":      "ambiguous",
            "sub_category":      "underspecified_instruction",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": "Call me tomorrow.",
            "ground_truth":      "ambiguous",
            "sub_category":      "underspecified_instruction",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": "Make it bigger.",
            "ground_truth":      "ambiguous",
            "sub_category":      "underspecified_instruction",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": "Send it to them.",
            "ground_truth":      "ambiguous",
            "sub_category":      "underspecified_instruction",
            "expected_behavior": "should_clarify",
        },
        # Scope ambiguity
        {
            "prompt": (
                "Every child loves some cartoon. Does every child love the same cartoon, "
                "or different cartoons?"
            ),
            "ground_truth":      "different cartoons",
            "sub_category":      "scope_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": (
                "I didn't say she stole the money. "
                "What is the intended meaning of this sentence?"
            ),
            "ground_truth":      "ambiguous — depends on stress",
            "sub_category":      "scope_ambiguity",
            "expected_behavior": "should_clarify",
        },
        # Structural ambiguity
        {
            "prompt": (
                "Flying planes can be dangerous. What is dangerous — "
                "the act of flying planes or planes that are flying?"
            ),
            "ground_truth":      "ambiguous",
            "sub_category":      "structural_ambiguity",
            "expected_behavior": "should_clarify",
        },
        {
            "prompt": "We saw the girl with the binoculars. Who had the binoculars?",
            "ground_truth":      "ambiguous",
            "sub_category":      "structural_ambiguity",
            "expected_behavior": "should_clarify",
        },
        # Pragmatic ambiguity
        {
            "prompt": (
                "Can you pass the salt? Is this a question about the person's ability, "
                "or is it a request?"
            ),
            "ground_truth":      "a request",
            "sub_category":      "pragmatic_ambiguity",
            "expected_behavior": "should_clarify",
        },
    ]

    records = []
    for p in probes:
        sub  = p.get("sub_category", "handcrafted")
        eb   = p.get("expected_behavior", "should_clarify")
        text = p["prompt"]
        records.append({
            "id":                make_id(text),
            "prompt":            text,
            "category":          "ambiguity",
            "sub_category":      sub,
            "expected_behavior": eb,
            "difficulty":        "hard",
            "ground_truth":      p.get("ground_truth", ""),
            "choices":           [],
            "source":            "handcrafted",
        })

    print(f"  Handcrafted probes: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("\n=== Person 3 – Ambiguity Handling: extract_data.py ===\n")

    all_records: list[dict] = []

    print("[1/3] BigBench tasks")
    all_records.extend(build_bigbench_records())

    print("\n[2/3] Natural Instructions tasks")
    all_records.extend(build_ni_records())

    print("\n[3/3] Handcrafted probes")
    all_records.extend(build_handcrafted_records())

    # Deduplicate by id
    seen   = set()
    unique = []
    for rec in all_records:
        if rec["id"] not in seen:
            seen.add(rec["id"])
            unique.append(rec)
    all_records = unique

    print(f"\nTotal unique records: {len(all_records)}")

    # Split
    random.shuffle(all_records)
    n_train = int(len(all_records) * TRAIN_RATIO)
    train   = all_records[:n_train]
    test    = all_records[n_train:]

    print(f"\nSplit → train={len(train)}, test={len(test)}")
    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(test,  DATA_DIR / "test.jsonl")

    print("\n✅  Person 3 data extraction complete.")


if __name__ == "__main__":
    main()
