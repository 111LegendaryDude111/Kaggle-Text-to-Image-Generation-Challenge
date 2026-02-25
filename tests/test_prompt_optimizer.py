from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_loader import PromptRecord
from generation.prompt_optimizer import (
    PromptOptimizerConfig,
    infer_object_counts,
    load_and_optimize_prompts,
    optimize_prompt_record,
    parse_count_token,
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


def test_parse_count_token_supports_digits_words_and_articles() -> None:
    assert parse_count_token("2") == 2
    assert parse_count_token("three") == 3
    assert parse_count_token("an") == 1
    assert parse_count_token("pair") == 2
    assert parse_count_token("unknown") is None


def test_infer_object_counts_uses_local_numeric_context() -> None:
    text = "Two dogs and a cat near a bench."
    nlp = DummyNlp(
        {
            text: [
                DummyToken("Two", "NUM"),
                DummyToken("dogs", "NOUN", "dog"),
                DummyToken("and", "CCONJ"),
                DummyToken("a", "DET"),
                DummyToken("cat", "NOUN", "cat"),
                DummyToken("near", "ADP"),
                DummyToken("a", "DET"),
                DummyToken("bench", "NOUN", "bench"),
            ]
        }
    )

    counts = infer_object_counts(text, nlp, expected_objects=["dog", "cat", "bench"])

    assert counts == {"dog": 2, "cat": 1, "bench": 1}


def test_optimize_prompt_record_applies_rewrite_components() -> None:
    text = "A woman with two dogs near a bench."
    prompt = PromptRecord(prompt_id="0001", text=text)
    nlp = DummyNlp(
        {
            text: [
                DummyToken("A", "DET"),
                DummyToken("woman", "NOUN", "woman"),
                DummyToken("with", "ADP"),
                DummyToken("two", "NUM"),
                DummyToken("dogs", "NOUN", "dog"),
                DummyToken("near", "ADP"),
                DummyToken("a", "DET"),
                DummyToken("bench", "NOUN", "bench"),
            ]
        }
    )
    config = PromptOptimizerConfig()

    optimized = optimize_prompt_record(prompt, nlp=nlp, config=config)

    assert optimized.expected_objects == ("person", "dog", "bench")
    assert optimized.object_counts["dog"] == 2
    assert "photorealistic" in optimized.optimized_text.lower()
    assert "main objects:" in optimized.optimized_text.lower()
    assert "object count constraint:" in optimized.optimized_text.lower()
    assert "avoid occlusion" in optimized.optimized_text.lower()


def test_load_and_optimize_prompts_from_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "DreamLayer-Prompt-Kaggle.txt"
    prompt_file.write_text("A cat on a chair.\n", encoding="utf-8")

    nlp = DummyNlp(
        {
            "A cat on a chair.": [
                DummyToken("A", "DET"),
                DummyToken("cat", "NOUN", "cat"),
                DummyToken("on", "ADP"),
                DummyToken("a", "DET"),
                DummyToken("chair", "NOUN", "chair"),
            ]
        }
    )

    records = load_and_optimize_prompts(prompt_file, nlp=nlp, config=PromptOptimizerConfig())

    assert len(records) == 1
    assert records[0].prompt_id == "0001"
    assert records[0].expected_objects == ("cat", "chair")
    assert "clearly visible cat" in records[0].optimized_text.lower()
