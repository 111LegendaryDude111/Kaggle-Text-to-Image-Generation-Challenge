#!/usr/bin/env python3
"""Load and validate challenge prompt files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_COMMENT_PREFIX = "#"
_EXPLICIT_PROMPT_RE = re.compile(r"^(?P<prompt_id>\d+)\s*(?:\||\t)\s*(?P<text>.*)$")


class PromptFormatError(ValueError):
    """Raised when a prompt file violates the expected format contract."""


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """Internal prompt representation used by downstream pipeline components."""

    prompt_id: str
    text: str
    expected_objects: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """Render the record in the JSON-compatible challenge contract."""
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "expected_objects": list(self.expected_objects),
        }


def _normalize_prompt_id(raw_prompt_id: str, line_number: int) -> str:
    try:
        numeric_prompt_id = int(raw_prompt_id)
    except ValueError as exc:
        raise PromptFormatError(
            f"Line {line_number}: prompt_id '{raw_prompt_id}' is not numeric."
        ) from exc

    if numeric_prompt_id <= 0:
        raise PromptFormatError(
            f"Line {line_number}: prompt_id must be positive; got {numeric_prompt_id}."
        )

    return f"{numeric_prompt_id:04d}"


def parse_prompt_file_content(file_content: str) -> list[PromptRecord]:
    """Parse and validate prompt file content."""
    records: list[PromptRecord] = []
    seen_prompt_ids: set[str] = set()
    parsed_mode: str | None = None
    implicit_prompt_index = 1

    for line_number, raw_line in enumerate(file_content.splitlines(), start=1):
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith(_COMMENT_PREFIX):
            continue

        explicit_match = _EXPLICIT_PROMPT_RE.match(stripped_line)
        if explicit_match:
            current_mode = "explicit"
            prompt_id = _normalize_prompt_id(explicit_match.group("prompt_id"), line_number)
            prompt_text = explicit_match.group("text").strip()
            if not prompt_text:
                raise PromptFormatError(
                    f"Line {line_number}: missing prompt text for prompt_id '{prompt_id}'."
                )
        else:
            current_mode = "implicit"
            prompt_id = f"{implicit_prompt_index:04d}"
            prompt_text = stripped_line
            implicit_prompt_index += 1

        if parsed_mode is None:
            parsed_mode = current_mode
        elif parsed_mode != current_mode:
            raise PromptFormatError(
                "Prompt file mixes explicit IDs with implicit prompt lines; "
                f"line {line_number} broke consistency."
            )

        if prompt_id in seen_prompt_ids:
            raise PromptFormatError(
                f"Line {line_number}: duplicate prompt_id '{prompt_id}' detected."
            )

        seen_prompt_ids.add(prompt_id)
        records.append(PromptRecord(prompt_id=prompt_id, text=prompt_text))

    if not records:
        raise PromptFormatError(
            "Prompt file is empty or only contains comments/blank lines."
        )

    return records


def load_prompt_file(input_path: str | Path) -> list[PromptRecord]:
    """Load a prompt file from disk and parse it into prompt records."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    raw_content = path.read_text(encoding="utf-8")
    return parse_prompt_file_content(raw_content)


def prompts_to_json_records(prompts: Sequence[PromptRecord]) -> list[dict[str, Any]]:
    """Convert parsed prompt records into JSON-compatible dictionaries."""
    return [prompt.to_json_dict() for prompt in prompts]


def load_prompt_file_as_json_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Load, validate, and return prompt records in JSON-compatible structure."""
    prompts = load_prompt_file(input_path)
    return prompts_to_json_records(prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="prompts/DreamLayer-Prompt-Kaggle.txt",
        help="Path to DreamLayer prompt file.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file. If omitted, JSON is printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = load_prompt_file_as_json_records(args.input)
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
