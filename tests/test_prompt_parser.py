from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_loader import PromptRecord
from evaluation.prompt_parser import (
    enrich_prompts_with_expected_objects,
    extract_expected_objects,
    extract_nouns,
    load_prompts_with_expected_objects,
    normalize_to_yolo_label,
)


class DummyToken:
    def __init__(self, text: str, pos: str, lemma: str | None = None) -> None:
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma if lemma is not None else text


class DummyNlp:
    def __init__(self, mapping: dict[str, list[DummyToken]]) -> None:
        self._mapping = mapping

    def __call__(self, text: str) -> list[DummyToken]:
        return self._mapping[text]


def test_extract_nouns_uses_pos_and_lemmatization() -> None:
    nlp = DummyNlp(
        {
            "A woman and two dogs near a bench.": [
                DummyToken("A", "DET"),
                DummyToken("woman", "NOUN", "woman"),
                DummyToken("and", "CCONJ"),
                DummyToken("two", "NUM"),
                DummyToken("dogs", "NOUN", "dog"),
                DummyToken("near", "ADP"),
                DummyToken("a", "DET"),
                DummyToken("bench", "NOUN", "bench"),
            ]
        }
    )

    nouns = extract_nouns("A woman and two dogs near a bench.", nlp)

    assert nouns == ["woman", "dog", "bench"]


def test_normalize_to_yolo_label_handles_aliases_and_variants() -> None:
    assert normalize_to_yolo_label("woman") == "person"
    assert normalize_to_yolo_label("cellphone") == "cell phone"
    assert normalize_to_yolo_label("traffic-light") == "traffic light"
    assert normalize_to_yolo_label("bikes") == "bicycle"
    assert normalize_to_yolo_label("dragon") is None


def test_extract_expected_objects_filters_unmapped_by_default() -> None:
    nlp = DummyNlp(
        {
            "A dragon and a dog.": [
                DummyToken("A", "DET"),
                DummyToken("dragon", "NOUN", "dragon"),
                DummyToken("and", "CCONJ"),
                DummyToken("dog", "NOUN", "dog"),
            ]
        }
    )

    objects = extract_expected_objects("A dragon and a dog.", nlp)

    assert objects == ["dog"]


def test_enrich_prompts_populates_expected_objects() -> None:
    prompts = [
        PromptRecord(prompt_id="0001", text="A woman with a phone on a bench."),
        PromptRecord(prompt_id="0002", text="A pizza on the table."),
    ]
    nlp = DummyNlp(
        {
            "A woman with a phone on a bench.": [
                DummyToken("A", "DET"),
                DummyToken("woman", "NOUN", "woman"),
                DummyToken("phone", "NOUN", "phone"),
                DummyToken("bench", "NOUN", "bench"),
            ],
            "A pizza on the table.": [
                DummyToken("A", "DET"),
                DummyToken("pizza", "NOUN", "pizza"),
                DummyToken("table", "NOUN", "table"),
            ],
        }
    )

    enriched = enrich_prompts_with_expected_objects(prompts, nlp=nlp)

    assert enriched[0].expected_objects == ("person", "cell phone", "bench")
    assert enriched[1].expected_objects == ("pizza", "dining table")


def test_load_prompts_with_expected_objects_from_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "DreamLayer-Prompt-Kaggle.txt"
    prompt_file.write_text("A dog and a cat.\n", encoding="utf-8")

    nlp = DummyNlp(
        {
            "A dog and a cat.": [
                DummyToken("A", "DET"),
                DummyToken("dog", "NOUN", "dog"),
                DummyToken("cat", "NOUN", "cat"),
            ]
        }
    )

    prompts = load_prompts_with_expected_objects(prompt_file, nlp=nlp)

    assert prompts[0].prompt_id == "0001"
    assert prompts[0].expected_objects == ("dog", "cat")
