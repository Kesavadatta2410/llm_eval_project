"""
evaluation – Shared evaluation utilities.

Provides common metrics, hallucination detection, safety checking,
and visualization helpers used by all person modules.
"""

from evaluation.metrics import (
    exact_match,
    fuzzy_match,
    compute_accuracy,
    compute_f1,
    normalize_text,
)
from evaluation.hallucination_detector import HallucinationDetector
from evaluation.safety_checker import SafetyChecker
