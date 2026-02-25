#!/usr/bin/env python3
"""Extract expected objects from prompts using POS tagging."""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

try:
    from evaluation.prompt_loader import (
        PromptRecord,
        load_prompt_file,
        prompts_to_json_records,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import (
        PromptRecord,
        load_prompt_file,
        prompts_to_json_records,
    )

_YOLO_COCO_LABELS: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
_YOLO_LABEL_SET = frozenset(_YOLO_COCO_LABELS)
_YOLO_COMPACT_LOOKUP = {label.replace(" ", ""): label for label in _YOLO_COCO_LABELS}

_ALIAS_TO_YOLO_LABEL: dict[str, str] = {
    "people": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "kid": "person",
    "biker": "person",
    "skater": "person",
    "surfer": "person",
    "cyclist": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "plane": "airplane",
    "aircraft": "airplane",
    "automobile": "car",
    "vehicle": "car",
    "lorry": "truck",
    "ship": "boat",
    "vessel": "boat",
    "trafficlight": "traffic light",
    "traffic lamp": "traffic light",
    "hydrant": "fire hydrant",
    "parkingmeter": "parking meter",
    "sofa": "couch",
    "plant": "potted plant",
    "table": "dining table",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "phone": "cell phone",
    "fridge": "refrigerator",
    "doughnut": "donut",
    "purse": "handbag",
    "racquet": "tennis racket",
    "racket": "tennis racket",
    "hairdryer": "hair drier",
    "dryer": "hair drier",
}

_WHITESPACE_RE = re.compile(r"[\s/_-]+")
_DROP_CHARS_RE = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE_RE = re.compile(r"\s+")


class TokenLike(Protocol):
    """Minimal token contract needed from spaCy outputs."""

    text: str
    lemma_: str
    pos_: str


NlpCallable = Callable[[str], Sequence[TokenLike]]


def _normalize_text(value: str) -> str:
    normalized = _WHITESPACE_RE.sub(" ", value.strip().lower())
    normalized = _DROP_CHARS_RE.sub("", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _lookup_label(noun: str) -> str | None:
    if noun in _ALIAS_TO_YOLO_LABEL:
        return _ALIAS_TO_YOLO_LABEL[noun]
    if noun in _YOLO_LABEL_SET:
        return noun
    compact = noun.replace(" ", "")
    if compact in _YOLO_COMPACT_LOOKUP:
        return _YOLO_COMPACT_LOOKUP[compact]
    return None


def normalize_to_yolo_label(noun: str) -> str | None:
    """Map a noun to a YOLO-compatible COCO label."""
    normalized_noun = _normalize_text(noun)
    if not normalized_noun:
        return None

    mapped = _lookup_label(normalized_noun)
    if mapped is not None:
        return mapped

    if normalized_noun.endswith("ies") and len(normalized_noun) > 3:
        singular = normalized_noun[:-3] + "y"
        mapped = _lookup_label(singular)
        if mapped is not None:
            return mapped

    if normalized_noun.endswith("s") and not normalized_noun.endswith("ss"):
        singular = normalized_noun[:-1]
        mapped = _lookup_label(singular)
        if mapped is not None:
            return mapped

    return None


def extract_nouns(prompt_text: str, nlp: NlpCallable) -> list[str]:
    """Extract lemmatized nouns from prompt text via POS tagging."""
    nouns: list[str] = []
    for token in nlp(prompt_text):
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        lemma_or_text = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
        normalized = _normalize_text(lemma_or_text)
        if normalized:
            nouns.append(normalized)
    return _dedupe_preserve_order(nouns)


def normalize_nouns_to_yolo_labels(
    nouns: Sequence[str], *, keep_unmapped: bool = False
) -> list[str]:
    """Normalize nouns to YOLO labels and optionally keep unknown nouns."""
    labels: list[str] = []
    for noun in nouns:
        yolo_label = normalize_to_yolo_label(noun)
        if yolo_label is not None:
            labels.append(yolo_label)
            continue
        if keep_unmapped:
            labels.append(noun)
    return _dedupe_preserve_order(labels)


def extract_expected_objects(
    prompt_text: str, nlp: NlpCallable, *, keep_unmapped: bool = False
) -> list[str]:
    """Extract and normalize expected objects for a single prompt."""
    nouns = extract_nouns(prompt_text, nlp)
    return normalize_nouns_to_yolo_labels(nouns, keep_unmapped=keep_unmapped)


@lru_cache(maxsize=2)
def get_spacy_nlp(model_name: str = "en_core_web_sm") -> NlpCallable:
    """Load and cache spaCy NLP model for POS tagging."""
    try:
        import spacy
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("spaCy is not installed in the environment.") from exc

    try:
        return cast(NlpCallable, spacy.load(model_name))
    except OSError as exc:  # pragma: no cover - model availability guard
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: .venv/bin/python -m spacy download {model_name}"
        ) from exc


def enrich_prompts_with_expected_objects(
    prompts: Sequence[PromptRecord],
    *,
    nlp: NlpCallable | None = None,
    spacy_model: str = "en_core_web_sm",
    keep_unmapped: bool = False,
) -> list[PromptRecord]:
    """Populate `expected_objects` for every prompt record."""
    nlp_runtime = nlp if nlp is not None else get_spacy_nlp(spacy_model)

    enriched: list[PromptRecord] = []
    for prompt in prompts:
        expected_objects = extract_expected_objects(
            prompt.text,
            nlp_runtime,
            keep_unmapped=keep_unmapped,
        )
        enriched.append(
            PromptRecord(
                prompt_id=prompt.prompt_id,
                text=prompt.text,
                expected_objects=tuple(expected_objects),
            )
        )
    return enriched


def load_prompts_with_expected_objects(
    input_path: str | Path,
    *,
    nlp: NlpCallable | None = None,
    spacy_model: str = "en_core_web_sm",
    keep_unmapped: bool = False,
) -> list[PromptRecord]:
    """Load prompt file and populate expected objects for each record."""
    prompts = load_prompt_file(input_path)
    return enrich_prompts_with_expected_objects(
        prompts,
        nlp=nlp,
        spacy_model=spacy_model,
        keep_unmapped=keep_unmapped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="prompts/DreamLayer-Prompt-Kaggle.txt",
        help="Path to DreamLayer prompt file.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model used for POS tagging.",
    )
    parser.add_argument(
        "--keep-unmapped",
        action="store_true",
        help="Keep nouns that cannot be normalized to YOLO labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prompts = load_prompts_with_expected_objects(
            args.input,
            spacy_model=args.spacy_model,
            keep_unmapped=args.keep_unmapped,
        )
        payload = prompts_to_json_records(prompts)
        rendered = json.dumps(payload, indent=2)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
