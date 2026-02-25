#!/usr/bin/env python3
"""Object-level precision/recall/F1 scoring for expected vs detected labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class F1MetricResult:
    """Full metric breakdown for one expected-vs-detected comparison."""

    expected_objects: tuple[str, ...]
    detected_objects: tuple[str, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize result to JSON-compatible shape."""
        return {
            "expected_objects": list(self.expected_objects),
            "detected_objects": list(self.detected_objects),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _unique_normalized_labels(labels: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = _normalize_label(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def compute_precision(true_positives: int, false_positives: int) -> float:
    """Compute precision with safe handling of zero predicted positives."""
    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def compute_recall(true_positives: int, false_negatives: int) -> float:
    """Compute recall with safe handling of zero expected positives."""
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 1.0
    return true_positives / denominator


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall with division-by-zero protection."""
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / denominator


def score_object_detection(
    *,
    expected_objects: Sequence[str],
    detected_objects: Sequence[str],
) -> F1MetricResult:
    """Compute object-level precision, recall, and F1 from label collections."""
    normalized_expected = _unique_normalized_labels(expected_objects)
    normalized_detected = _unique_normalized_labels(detected_objects)

    expected_set = set(normalized_expected)
    detected_set = set(normalized_detected)

    true_positives = len(expected_set & detected_set)
    false_positives = len(detected_set - expected_set)
    false_negatives = len(expected_set - detected_set)

    precision = compute_precision(true_positives, false_positives)
    recall = compute_recall(true_positives, false_negatives)
    f1 = compute_f1_score(precision, recall)

    if not normalized_expected and not normalized_detected:
        precision = 1.0
        recall = 1.0
        f1 = 1.0

    return F1MetricResult(
        expected_objects=normalized_expected,
        detected_objects=normalized_detected,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )
