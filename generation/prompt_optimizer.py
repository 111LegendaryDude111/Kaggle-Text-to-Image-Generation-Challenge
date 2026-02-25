#!/usr/bin/env python3
"""Rewrite prompts to improve object grounding for detection-oriented generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

try:
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from evaluation.prompt_parser import (
        NlpCallable,
        extract_expected_objects,
        get_spacy_nlp,
        normalize_to_yolo_label,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from evaluation.prompt_parser import (
        NlpCallable,
        extract_expected_objects,
        get_spacy_nlp,
        normalize_to_yolo_label,
    )

_DIGIT_RE = re.compile(r"^\d+$")
_PUNCT_EDGE_RE = re.compile(r"(^[^a-z0-9]+|[^a-z0-9]+$)")

_NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "both": 2,
    "pair": 2,
    "couple": 2,
    "three": 3,
    "several": 3,
    "few": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_IRREGULAR_PLURALS: dict[str, str] = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "mouse": "mice",
    "tooth": "teeth",
    "foot": "feet",
}


class PromptOptimizerConfig(BaseModel):
    """Configuration for deterministic prompt rewriting."""

    model_config = ConfigDict(extra="ignore")

    prompt_file: str = "prompts/DreamLayer-Prompt-Kaggle.txt"
    output_path: str = "generation/optimized_prompts.json"
    spacy_model: str = "en_core_web_sm"
    keep_unmapped_nouns: bool = False

    expand_noun_phrases: bool = True
    reinforce_object_count: bool = True
    add_realism_bias: bool = True
    reduce_ambiguity: bool = True

    realism_prefix: str = (
        "A photorealistic, high-resolution photograph with natural lighting and sharp details."
    )
    ambiguity_directive: str = (
        "Keep object boundaries clear, avoid occlusion, and use a simple uncluttered composition."
    )
    anti_style_directive: str = (
        "Avoid abstract art, heavy stylization, strong motion blur, and ambiguous silhouettes."
    )


@dataclass(frozen=True, slots=True)
class OptimizedPromptRecord:
    """Prompt record after deterministic optimization."""

    prompt_id: str
    original_text: str
    optimized_text: str
    expected_objects: tuple[str, ...]
    object_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        """Render optimized record as JSON-compatible dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "text": self.original_text,
            "optimized_text": self.optimized_text,
            "expected_objects": list(self.expected_objects),
            "object_counts": dict(self.object_counts),
        }


def load_prompt_optimizer_config(config_path: str | Path) -> PromptOptimizerConfig:
    """Load optimizer config from JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Optimizer config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optimizer config JSON must be an object")

    if isinstance(payload.get("prompt_optimization"), dict):
        payload = payload["prompt_optimization"]

    return PromptOptimizerConfig.model_validate(payload)


def _normalize_count_token(raw_token: str) -> str:
    lowered = raw_token.strip().lower()
    return _PUNCT_EDGE_RE.sub("", lowered)


def parse_count_token(raw_token: str) -> int | None:
    """Parse a single token into a concrete count if possible."""
    normalized = _normalize_count_token(raw_token)
    if not normalized:
        return None
    if _DIGIT_RE.match(normalized):
        return max(1, int(normalized))
    return _NUMBER_WORDS.get(normalized)


def infer_object_counts(
    prompt_text: str,
    nlp: NlpCallable,
    expected_objects: Sequence[str],
) -> dict[str, int]:
    """Infer object counts from local token context around detected nouns."""
    counts: dict[str, int] = {}
    tokens = list(nlp(prompt_text))

    for index, token in enumerate(tokens):
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        token_lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
        normalized_label = normalize_to_yolo_label(token_lemma)
        if normalized_label is None:
            continue

        inferred_count = 1
        if index > 0:
            immediate = parse_count_token(tokens[index - 1].text)
            if immediate is not None:
                inferred_count = immediate

        if index > 1 and _normalize_count_token(tokens[index - 1].text) == "of":
            indirect = parse_count_token(tokens[index - 2].text)
            if indirect is not None:
                inferred_count = indirect

        current = counts.get(normalized_label, 0)
        counts[normalized_label] = max(current, inferred_count)

    for expected in expected_objects:
        counts.setdefault(expected, 1)

    return counts


def _pluralize_word(word: str) -> str:
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    return f"{word}s"


def pluralize_label(label: str, count: int) -> str:
    """Pluralize a YOLO label for readable rewritten text."""
    if count == 1:
        return label
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if " " not in label:
        return _pluralize_word(label)
    prefix, suffix = label.rsplit(" ", 1)
    return f"{prefix} {_pluralize_word(suffix)}"


def _article_for(word: str) -> str:
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def build_expanded_object_phrases(
    expected_objects: Sequence[str],
    object_counts: dict[str, int],
) -> list[str]:
    """Expand object mentions into explicit, detection-friendly noun phrases."""
    phrases: list[str] = []
    for label in expected_objects:
        count = object_counts.get(label, 1)
        if count == 1:
            article = _article_for(label)
            phrases.append(f"{article} clearly visible {label}")
            continue
        plural = pluralize_label(label, count)
        phrases.append(f"{count} clearly visible {plural}")
    return phrases


def build_count_reinforcement_clause(
    expected_objects: Sequence[str],
    object_counts: dict[str, int],
) -> str:
    """Build explicit count constraints to reinforce object grounding."""
    clauses = []
    for label in expected_objects:
        count = object_counts.get(label, 1)
        unit = "instance" if count == 1 else "instances"
        clauses.append(f"exactly {count} {unit} of {label}")
    return "Object count constraint: " + "; ".join(clauses) + "."


def compose_optimized_prompt(
    original_text: str,
    expected_objects: Sequence[str],
    object_counts: dict[str, int],
    config: PromptOptimizerConfig,
) -> str:
    """Compose final optimized prompt text from deterministic rewrite components."""
    segments: list[str] = []

    if config.add_realism_bias and config.realism_prefix.strip():
        segments.append(config.realism_prefix.strip())

    segments.append(f"Scene description: {original_text.strip()}")

    if config.expand_noun_phrases and expected_objects:
        object_phrases = build_expanded_object_phrases(expected_objects, object_counts)
        segments.append("Main objects: " + ", ".join(object_phrases) + ".")

    if config.reinforce_object_count and expected_objects:
        segments.append(build_count_reinforcement_clause(expected_objects, object_counts))

    if config.reduce_ambiguity and config.ambiguity_directive.strip():
        segments.append(config.ambiguity_directive.strip())

    if config.add_realism_bias and config.anti_style_directive.strip():
        segments.append(config.anti_style_directive.strip())

    return " ".join(part for part in segments if part).strip()


def optimize_prompt_record(
    prompt: PromptRecord,
    *,
    nlp: NlpCallable,
    config: PromptOptimizerConfig,
) -> OptimizedPromptRecord:
    """Optimize a single prompt record."""
    expected_objects = extract_expected_objects(
        prompt.text,
        nlp,
        keep_unmapped=config.keep_unmapped_nouns,
    )
    object_counts = infer_object_counts(prompt.text, nlp, expected_objects)
    optimized_text = compose_optimized_prompt(
        prompt.text,
        expected_objects,
        object_counts,
        config,
    )

    return OptimizedPromptRecord(
        prompt_id=prompt.prompt_id,
        original_text=prompt.text,
        optimized_text=optimized_text,
        expected_objects=tuple(expected_objects),
        object_counts=object_counts,
    )


def optimize_prompts(
    prompts: Sequence[PromptRecord],
    *,
    nlp: NlpCallable | None = None,
    config: PromptOptimizerConfig | None = None,
) -> list[OptimizedPromptRecord]:
    """Optimize all prompts with deterministic rewrite rules."""
    runtime_config = config or PromptOptimizerConfig()
    runtime_nlp = nlp if nlp is not None else get_spacy_nlp(runtime_config.spacy_model)

    return [
        optimize_prompt_record(prompt, nlp=runtime_nlp, config=runtime_config)
        for prompt in prompts
    ]


def load_and_optimize_prompts(
    prompt_file: str | Path,
    *,
    nlp: NlpCallable | None = None,
    config: PromptOptimizerConfig | None = None,
) -> list[OptimizedPromptRecord]:
    """Load prompt file and return optimized prompts."""
    prompts = load_prompt_file(prompt_file)
    return optimize_prompts(prompts, nlp=nlp, config=config)


def optimized_records_to_json(
    records: Sequence[OptimizedPromptRecord],
) -> list[dict[str, Any]]:
    """Convert optimized records to JSON-serializable dictionaries."""
    return [record.to_json_dict() for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON config. Uses `prompt_optimization` section when present.",
    )
    parser.add_argument("--input", help="Optional prompt file override.")
    parser.add_argument("--output", help="Optional output JSON path override.")
    parser.add_argument("--spacy-model", help="Optional spaCy model override.")
    parser.add_argument(
        "--keep-unmapped-nouns",
        action="store_true",
        help="Keep nouns that do not map to YOLO labels.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: PromptOptimizerConfig,
    args: argparse.Namespace,
) -> PromptOptimizerConfig:
    """Apply explicit CLI overrides on top of config values."""
    updates: dict[str, Any] = {}
    if args.input is not None:
        updates["prompt_file"] = args.input
    if args.output is not None:
        updates["output_path"] = args.output
    if args.spacy_model is not None:
        updates["spacy_model"] = args.spacy_model
    if args.keep_unmapped_nouns:
        updates["keep_unmapped_nouns"] = True

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return PromptOptimizerConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        config = load_prompt_optimizer_config(args.config)
        config = apply_cli_overrides(config, args)
        records = load_and_optimize_prompts(config.prompt_file, config=config)
        payload = optimized_records_to_json(records)

        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "optimized_count": len(records),
                    "output_path": str(output_path.resolve()),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
