from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.f1_metric import (
    compute_f1_score,
    compute_precision,
    compute_recall,
    score_object_detection,
)


def test_score_object_detection_exact_match() -> None:
    result = score_object_detection(
        expected_objects=["cat", "dog"],
        detected_objects=["cat", "dog"],
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_score_object_detection_partial_overlap() -> None:
    result = score_object_detection(
        expected_objects=["cat", "dog"],
        detected_objects=["dog", "chair"],
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5


def test_score_object_detection_handles_empty_sets_safely() -> None:
    both_empty = score_object_detection(expected_objects=[], detected_objects=[])
    assert both_empty.precision == 1.0
    assert both_empty.recall == 1.0
    assert both_empty.f1 == 1.0

    missing_all = score_object_detection(expected_objects=["cat"], detected_objects=[])
    assert missing_all.precision == 0.0
    assert missing_all.recall == 0.0
    assert missing_all.f1 == 0.0

    false_alarm_only = score_object_detection(expected_objects=[], detected_objects=["cat"])
    assert false_alarm_only.precision == 0.0
    assert false_alarm_only.recall == 1.0
    assert false_alarm_only.f1 == 0.0


def test_score_object_detection_normalizes_and_dedupes_labels() -> None:
    result = score_object_detection(
        expected_objects=[" Cat ", "DOG", "dog"],
        detected_objects=["cat", "dog", "cat"],
    )

    assert result.expected_objects == ("cat", "dog")
    assert result.detected_objects == ("cat", "dog")
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_metric_components_handle_zero_denominators() -> None:
    assert compute_precision(true_positives=0, false_positives=0) == 0.0
    assert compute_recall(true_positives=0, false_negatives=0) == 1.0
    assert compute_f1_score(precision=0.0, recall=0.0) == 0.0
