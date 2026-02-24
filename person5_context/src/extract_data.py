"""
Person 5 – Context Length Evaluation
extract_data.py

Pulls data from BigBench (long_context_integration, question_answer_by_type,
elementary_math_qa), Natural Instructions, and synthetic needle-in-a-haystack
probes across 5 context lengths (256–4096 tokens).

Each record includes extra fields: `context_length` and `needle_position`.
Outputs train.jsonl and test.jsonl under person5_context/data/.
"""

import hashlib
import json
import random
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
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
    "long_context_integration",
    "question_answer_by_type",
    "elementary_math_qa",
]

NI_TASKS = [
    "task1161_copa_cause_effect",
    "task1385_anli_r1_entailment",
    "task1543_movie_qa_question_answering",
    "task1624_disfl_qa_question_answering",
]

NI_BASE_URL = (
    "https://raw.githubusercontent.com/allenai/natural-instructions"
    "/master/tasks/{task_name}.json"
)

BB_BASE_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main"
    "/bigbench/benchmark_tasks/{task_name}/task.json"
)

# Needle-in-a-haystack context lengths (in approximate tokens)
CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096]

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
            choices     = ex.get("target_scores", {})
            gt_choices  = [k for k, v in choices.items() if v == max(choices.values())] if choices else []
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
# Filler text pool for needle-in-a-haystack contexts
# ---------------------------------------------------------------------------
FILLER_SENTENCES = [
    "The sun rises in the east and sets in the west.",
    "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
    "The capital of France is Paris, a city known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into food.",
    "The Amazon River is the largest river by discharge in the world.",
    "Jupiter is the largest planet in the solar system.",
    "DNA stands for deoxyribonucleic acid.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Mount Everest is the highest mountain on Earth.",
    "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
    "The human body has 206 bones in the adult skeleton.",
    "The Great Wall of China stretches thousands of miles.",
    "Elephants are the largest land animals.",
    "The periodic table organises elements by atomic number.",
    "The moon orbits the Earth once every 27.3 days approximately.",
]


def _make_filler(token_target: int) -> str:
    """Generate filler text with approximately `token_target` tokens (≈ 0.75 words/token)."""
    word_target = int(token_target * 0.75)
    words: list[str] = []
    rng = random.Random(12345)
    while len(words) < word_target:
        sentence = rng.choice(FILLER_SENTENCES)
        words.extend(sentence.split())
    return " ".join(words[:word_target])


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
            prompt = ex["prompt"]
            # Estimate context length in tokens (approx: len(prompt.split()) / 0.75)
            est_tokens = int(len(prompt.split()) / 0.75)
            all_records.append({
                "id":                make_id(prompt),
                "prompt":            prompt,
                "category":          "context_length",
                "sub_category":      task_name,
                "expected_behavior": "should_retrieve_correctly",
                "difficulty":        "medium",
                "ground_truth":      ex["ground_truth"],
                "choices":           ex["choices"],
                "source":            "bigbench",
                "context_length":    est_tokens,
                "needle_position":   None,
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
            est_tokens = int(len(prompt.split()) / 0.75)
            all_records.append({
                "id":                make_id(prompt),
                "prompt":            prompt,
                "category":          "context_length",
                "sub_category":      task_name,
                "expected_behavior": "should_retrieve_correctly",
                "difficulty":        "medium",
                "ground_truth":      gt,
                "choices":           [],
                "source":            "natural_instructions",
                "context_length":    est_tokens,
                "needle_position":   None,
            })
    print(f"  NI total: {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Needle-in-a-haystack synthetic probes
# ---------------------------------------------------------------------------

# Core facts to hide in filler text ("needles")
NEEDLES = [
    {"fact": "The secret password is DELTA-7.", "question": "What is the secret password?", "answer": "DELTA-7"},
    {"fact": "Dr. Eleanor Price discovered compound XR-42.", "question": "Who discovered compound XR-42?", "answer": "Dr. Eleanor Price"},
    {"fact": "The meeting is scheduled for Thursday at 14:30.", "question": "When is the meeting scheduled?", "answer": "Thursday at 14:30"},
    {"fact": "Agent code name is NIGHTHAWK.", "question": "What is the agent code name?", "answer": "NIGHTHAWK"},
    {"fact": "The treasure is buried under the oak tree at coordinates 48.8566° N, 2.3522° E.", "question": "Where is the treasure buried?", "answer": "under the oak tree at coordinates 48.8566° N, 2.3522° E"},
    {"fact": "The activation key for the system is ALPHA-ZETA-9.", "question": "What is the activation key?", "answer": "ALPHA-ZETA-9"},
    {"fact": "Project Aurora will launch on the 15th of next month.", "question": "When will Project Aurora launch?", "answer": "the 15th of next month"},
]

NEEDLE_POSITIONS = ["beginning", "middle", "end"]


def build_needle_records() -> list:
    """Synthetic needle-in-a-haystack probes across 5 context lengths."""
    records = []
    rng     = random.Random(RANDOM_SEED)

    for ctx_len in CONTEXT_LENGTHS:
        for needle_info in NEEDLES:
            for position in NEEDLE_POSITIONS:
                filler = _make_filler(ctx_len)
                filler_words = filler.split()
                needle_sentence = needle_info["fact"]

                # Insert needle at the desired position
                n = len(filler_words)
                if position == "beginning":
                    insert_at = 0
                elif position == "middle":
                    insert_at = n // 2
                else:
                    insert_at = n

                context_words = filler_words[:insert_at] + needle_sentence.split() + filler_words[insert_at:]
                context = " ".join(context_words)

                prompt = (
                    f"Read the following passage carefully and answer the question.\n\n"
                    f"Passage:\n{context}\n\n"
                    f"Question: {needle_info['question']}\n"
                    f"Answer:"
                )

                records.append({
                    "id":                make_id(prompt),
                    "prompt":            prompt,
                    "category":          "context_length",
                    "sub_category":      "needle_in_haystack",
                    "expected_behavior": "should_retrieve_correctly",
                    "difficulty":        "hard",
                    "ground_truth":      needle_info["answer"],
                    "choices":           [],
                    "source":            "synthetic",
                    "context_length":    ctx_len,
                    "needle_position":   position,
                })

    print(f"  Needle-in-haystack probes: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("\n=== Person 5 – Context Length Evaluation: extract_data.py ===\n")

    all_records: list[dict] = []

    print("[1/3] BigBench tasks")
    all_records.extend(build_bigbench_records())

    print("\n[2/3] Natural Instructions tasks")
    all_records.extend(build_ni_records())

    print("\n[3/3] Needle-in-a-haystack synthetic probes")
    all_records.extend(build_needle_records())

    # Deduplicate by id
    seen   = set()
    unique = []
    for rec in all_records:
        if rec["id"] not in seen:
            seen.add(rec["id"])
            unique.append(rec)
    all_records = unique

    print(f"\nTotal unique records: {len(all_records)}")

    random.shuffle(all_records)
    n_train = int(len(all_records) * TRAIN_RATIO)
    train   = all_records[:n_train]
    test    = all_records[n_train:]

    print(f"Split → train={len(train)}, test={len(test)}")
    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(test,  DATA_DIR / "test.jsonl")

    print("\n✅  Person 5 data extraction complete.")


if __name__ == "__main__":
    main()
