from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_loader import (
    PromptFormatError,
    load_prompt_file_as_json_records,
    parse_prompt_file_content,
)


def test_parse_implicit_prompt_lines_assigns_sequential_ids() -> None:
    content = """
    # Prompt file comment
    A cat sitting on a chair.

    A lamp beside a wooden table.
    """

    prompts = parse_prompt_file_content(content)

    assert [prompt.prompt_id for prompt in prompts] == ["0001", "0002"]
    assert [prompt.text for prompt in prompts] == [
        "A cat sitting on a chair.",
        "A lamp beside a wooden table.",
    ]


def test_parse_explicit_prompt_ids_supports_pipe_and_tab() -> None:
    content = "1|A red car parked near trees.\n0002\tTwo dogs on the grass."

    prompts = parse_prompt_file_content(content)

    assert [prompt.prompt_id for prompt in prompts] == ["0001", "0002"]
    assert [prompt.text for prompt in prompts] == [
        "A red car parked near trees.",
        "Two dogs on the grass.",
    ]


def test_mixed_explicit_and_implicit_formats_raise_validation_error() -> None:
    content = "0001|A train at the station.\nA person waiting nearby."

    with pytest.raises(PromptFormatError):
        parse_prompt_file_content(content)


def test_duplicate_prompt_id_raises_validation_error() -> None:
    content = "0001|A single prompt.\n1|Another prompt with duplicate id."

    with pytest.raises(PromptFormatError):
        parse_prompt_file_content(content)


def test_empty_or_comment_only_file_raises_validation_error() -> None:
    content = "# header\n\n# no prompts"

    with pytest.raises(PromptFormatError):
        parse_prompt_file_content(content)


def test_json_output_matches_prompt_contract(tmp_path) -> None:
    prompt_file = tmp_path / "DreamLayer-Prompt-Kaggle.txt"
    prompt_file.write_text("A zebra in a field.\n", encoding="utf-8")

    payload = load_prompt_file_as_json_records(prompt_file)

    assert payload == [
        {
            "prompt_id": "0001",
            "text": "A zebra in a field.",
            "expected_objects": [],
        }
    ]
    json.dumps(payload)
