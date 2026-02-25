#!/usr/bin/env python3
"""Structured negative prompt strategy for precision-oriented generation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

try:
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from evaluation.prompt_parser import NlpCallable, extract_expected_objects, get_spacy_nlp
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from evaluation.prompt_parser import NlpCallable, extract_expected_objects, get_spacy_nlp

_QUALITY_SUPPRESSION_TERMS: tuple[str, ...] = (
    "low quality",
    "blur",
    "out of focus",
    "jpeg artifacts",
    "watermark",
    "text overlay",
    "logo",
    "distorted anatomy",
)

_HALLUCINATION_SUPPRESSION_TERMS: tuple[str, ...] = (
    "extra objects",
    "duplicate objects",
    "repeated objects",
    "unrequested items",
    "object fragments",
    "floating objects",
)

_STYLE_SUPPRESSION_TERMS: tuple[str, ...] = (
    "cartoon style",
    "anime style",
    "abstract art",
    "illustration style",
    "3d render",
    "cgi look",
)

_CLUTTER_SUPPRESSION_TERMS: dict[str, tuple[str, ...]] = {
    "low": (
        "busy background",
        "background clutter",
        "unnecessary props",
    ),
    "medium": (
        "busy background",
        "background clutter",
        "crowded scene",
        "visual clutter",
        "too many small objects",
    ),
    "high": (
        "busy background",
        "background clutter",
        "crowded scene",
        "visual clutter",
        "too many small objects",
        "chaotic composition",
        "object overlap",
        "occluded objects",
        "messy scene",
    ),
}


class NegativePromptStrategyConfig(BaseModel):
    """Config for structured negative-prompt generation."""

    model_config = ConfigDict(extra="ignore")

    clutter_suppression_level: Literal["low", "medium", "high"] = "medium"
    include_quality_terms: bool = True
    include_hallucination_terms: bool = True
    include_style_terms: bool = True
    enforce_expected_object_lock: bool = True

    extract_expected_objects: bool = True
    spacy_model: str = "en_core_web_sm"

    custom_terms: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HallucinationReductionReport:
    """Comparison report for hallucination rate before and after a strategy."""

    expected_objects: tuple[str, ...]
    before_detected: tuple[str, ...]
    after_detected: tuple[str, ...]
    before_hallucinations: tuple[str, ...]
    after_hallucinations: tuple[str, ...]
    before_rate: float
    after_rate: float
    absolute_reduction: float
    relative_reduction: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize report to JSON-friendly dictionary."""
        return {
            "expected_objects": list(self.expected_objects),
            "before_detected": list(self.before_detected),
            "after_detected": list(self.after_detected),
            "before_hallucinations": list(self.before_hallucinations),
            "after_hallucinations": list(self.after_hallucinations),
            "before_rate": self.before_rate,
            "after_rate": self.after_rate,
            "absolute_reduction": self.absolute_reduction,
            "relative_reduction": self.relative_reduction,
        }


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _split_negative_prompt_terms(text: str) -> list[str]:
    if not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _dedupe_terms(terms: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_label(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(term.strip())
    return result


def build_object_lock_terms(expected_objects: Sequence[str]) -> list[str]:
    """Build constraints to discourage objects outside expected set."""
    normalized_expected = [_normalize_label(item) for item in expected_objects if item.strip()]
    if not normalized_expected:
        return []

    object_list = ", ".join(normalized_expected)
    return [
        "objects outside target set",
        "extra instances beyond requested count",
        f"target object set locked to: {object_list}",
    ]


def build_structured_negative_prompt(
    *,
    expected_objects: Sequence[str],
    base_negative_prompt: str = "",
    config: NegativePromptStrategyConfig | None = None,
) -> str:
    """Create a structured negative prompt with clutter/hallucination suppression."""
    runtime = config or NegativePromptStrategyConfig()
    terms: list[str] = []

    terms.extend(_split_negative_prompt_terms(base_negative_prompt))

    if runtime.include_quality_terms:
        terms.extend(_QUALITY_SUPPRESSION_TERMS)
    terms.extend(_CLUTTER_SUPPRESSION_TERMS[runtime.clutter_suppression_level])
    if runtime.include_hallucination_terms:
        terms.extend(_HALLUCINATION_SUPPRESSION_TERMS)
    if runtime.include_style_terms:
        terms.extend(_STYLE_SUPPRESSION_TERMS)
    if runtime.enforce_expected_object_lock:
        terms.extend(build_object_lock_terms(expected_objects))

    terms.extend(runtime.custom_terms)
    return ", ".join(_dedupe_terms(terms))


def maybe_load_nlp_for_negative_prompt_strategy(
    config: NegativePromptStrategyConfig,
) -> NlpCallable | None:
    """Best-effort spaCy loader for expected-object extraction."""
    if not config.extract_expected_objects:
        return None
    try:
        return get_spacy_nlp(config.spacy_model)
    except Exception:
        return None


def resolve_expected_objects_for_prompt(
    prompt: PromptRecord,
    *,
    nlp: NlpCallable | None,
    config: NegativePromptStrategyConfig,
) -> list[str]:
    """Resolve expected objects from record metadata or prompt text."""
    if prompt.expected_objects:
        return list(prompt.expected_objects)
    if nlp is None:
        return []
    return extract_expected_objects(prompt.text, nlp, keep_unmapped=False)


def build_negative_prompt_for_prompt(
    prompt: PromptRecord,
    *,
    base_negative_prompt: str,
    config: NegativePromptStrategyConfig,
    nlp: NlpCallable | None = None,
) -> str:
    """Build prompt-specific structured negative prompt string."""
    expected_objects = resolve_expected_objects_for_prompt(prompt, nlp=nlp, config=config)
    return build_structured_negative_prompt(
        expected_objects=expected_objects,
        base_negative_prompt=base_negative_prompt,
        config=config,
    )


def hallucinated_objects(
    expected_objects: Sequence[str], detected_objects: Sequence[str]
) -> list[str]:
    """Return detected objects not present in expected object set."""
    expected = {_normalize_label(item) for item in expected_objects if item.strip()}
    extras = {_normalize_label(item) for item in detected_objects if item.strip()} - expected
    return sorted(extras)


def hallucination_rate(expected_objects: Sequence[str], detected_objects: Sequence[str]) -> float:
    """Compute hallucination rate = extra_object_count / detected_object_count."""
    detected = {_normalize_label(item) for item in detected_objects if item.strip()}
    if not detected:
        return 0.0
    extras = detected - {_normalize_label(item) for item in expected_objects if item.strip()}
    return len(extras) / len(detected)


def evaluate_hallucination_reduction(
    *,
    expected_objects: Sequence[str],
    before_detected: Sequence[str],
    after_detected: Sequence[str],
) -> HallucinationReductionReport:
    """Compare hallucination rates before vs after applying negative prompt strategy."""
    before_rate = hallucination_rate(expected_objects, before_detected)
    after_rate = hallucination_rate(expected_objects, after_detected)
    absolute_reduction = before_rate - after_rate
    relative_reduction = (absolute_reduction / before_rate) if before_rate > 0 else 0.0

    return HallucinationReductionReport(
        expected_objects=tuple(_normalize_label(item) for item in expected_objects if item.strip()),
        before_detected=tuple(_normalize_label(item) for item in before_detected if item.strip()),
        after_detected=tuple(_normalize_label(item) for item in after_detected if item.strip()),
        before_hallucinations=tuple(hallucinated_objects(expected_objects, before_detected)),
        after_hallucinations=tuple(hallucinated_objects(expected_objects, after_detected)),
        before_rate=before_rate,
        after_rate=after_rate,
        absolute_reduction=absolute_reduction,
        relative_reduction=relative_reduction,
    )


def preview_negative_prompts(
    prompt_file: str | Path,
    *,
    base_negative_prompt: str = "",
    config: NegativePromptStrategyConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate structured negative prompts for all prompts in a file."""
    runtime = config or NegativePromptStrategyConfig()
    nlp = maybe_load_nlp_for_negative_prompt_strategy(runtime)
    prompts = load_prompt_file(prompt_file)

    output: list[dict[str, Any]] = []
    for prompt in prompts:
        structured_negative_prompt = build_negative_prompt_for_prompt(
            prompt,
            base_negative_prompt=base_negative_prompt,
            config=runtime,
            nlp=nlp,
        )
        output.append(
            {
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "negative_prompt": structured_negative_prompt,
            }
        )
    return output


def dump_preview_json(
    prompt_file: str | Path,
    output_path: str | Path,
    *,
    base_negative_prompt: str = "",
    config: NegativePromptStrategyConfig | None = None,
) -> None:
    """Write structured negative prompt preview JSON to disk."""
    payload = preview_negative_prompts(
        prompt_file,
        base_negative_prompt=base_negative_prompt,
        config=config,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

