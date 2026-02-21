"""
metrics.py — Common evaluation metrics shared across all dimensions.

Functions:
  - normalize_text(text)          → cleaned, lowercased text
  - exact_match(pred, gold)       → bool
  - fuzzy_match(pred, gold, thr)  → bool
  - compute_accuracy(preds, golds)→ float
  - compute_f1(preds, golds)      → dict with precision, recall, f1
  - compute_bleu(pred, reference) → float
  - token_count(text)             → int
  - inference_statistics(times)   → dict with mean, median, p95
"""

import re
import numpy as np
from collections import Counter


# ── Text Normalization ──────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text


# ── Matching ────────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive exact match after normalization."""
    return normalize_text(prediction) == normalize_text(ground_truth)


def fuzzy_match(prediction: str, ground_truth: str, threshold: float = 0.6) -> bool:
    """
    Token-overlap fuzzy match.
    Returns True if the F1 overlap between prediction and ground truth
    tokens exceeds the threshold.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not gold_tokens:
        return len(pred_tokens) == 0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return False

    precision = num_common / len(pred_tokens) if pred_tokens else 0
    recall    = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1 >= threshold


# ── Aggregate Metrics ───────────────────────────────────────────────────────

def compute_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Fraction of exact matches."""
    if not predictions:
        return 0.0
    matches = sum(exact_match(p, g) for p, g in zip(predictions, ground_truths))
    return matches / len(predictions)


def compute_f1(predictions: list[str], ground_truths: list[str]) -> dict:
    """Micro-averaged token-level precision, recall, F1."""
    total_common = 0
    total_pred = 0
    total_gold = 0

    for pred, gold in zip(predictions, ground_truths):
        pred_tokens = normalize_text(pred).split()
        gold_tokens = normalize_text(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        total_common += sum(common.values())
        total_pred   += len(pred_tokens)
        total_gold   += len(gold_tokens)

    precision = total_common / total_pred if total_pred else 0
    recall    = total_common / total_gold if total_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_bleu(prediction: str, reference: str) -> float:
    """Simple unigram BLEU score (0–1)."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens  = normalize_text(reference).split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts  = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    clipped = {w: min(c, ref_counts.get(w, 0)) for w, c in pred_counts.items()}
    numerator = sum(clipped.values())

    brevity = min(1.0, len(pred_tokens) / len(ref_tokens))
    return brevity * (numerator / len(pred_tokens))


# ── Cross-Cutting Metrics ──────────────────────────────────────────────────

def token_count(text: str) -> int:
    """Approximate token count (whitespace split)."""
    return len(text.split())


def inference_statistics(times: list[float]) -> dict:
    """Compute mean, median, P95 inference time from a list of durations."""
    if not times:
        return {"mean": 0, "median": 0, "p95": 0}
    arr = np.array(times)
    return {
        "mean":   round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95":    round(float(np.percentile(arr, 95)), 4),
    }
