from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.negative_prompt_strategy import (
    NegativePromptStrategyConfig,
    build_structured_negative_prompt,
    evaluate_hallucination_reduction,
)


def test_build_structured_negative_prompt_contains_expected_blocks() -> None:
    config = NegativePromptStrategyConfig(
        clutter_suppression_level="medium",
        custom_terms=["lens flare artifacts"],
    )
    negative_prompt = build_structured_negative_prompt(
        expected_objects=["cat", "chair"],
        base_negative_prompt="overexposed highlights",
        config=config,
    )

    lowered = negative_prompt.lower()
    assert "overexposed highlights" in lowered
    assert "low quality" in lowered
    assert "background clutter" in lowered
    assert "extra objects" in lowered
    assert "abstract art" in lowered
    assert "target object set locked to: cat, chair" in lowered
    assert "lens flare artifacts" in lowered


def test_clutter_level_high_has_more_terms_than_low() -> None:
    low = build_structured_negative_prompt(
        expected_objects=[],
        config=NegativePromptStrategyConfig(clutter_suppression_level="low"),
    )
    high = build_structured_negative_prompt(
        expected_objects=[],
        config=NegativePromptStrategyConfig(clutter_suppression_level="high"),
    )

    assert len(high.split(",")) > len(low.split(","))
    assert "chaotic composition" in high.lower()


def test_hallucination_reduction_report_improves_when_extras_drop() -> None:
    report = evaluate_hallucination_reduction(
        expected_objects=["cat", "chair"],
        before_detected=["cat", "chair", "dog", "person"],
        after_detected=["cat", "chair"],
    )

    assert report.before_rate > report.after_rate
    assert report.absolute_reduction > 0
    assert report.relative_reduction > 0
    assert "dog" in report.before_hallucinations
    assert report.after_hallucinations == ()
