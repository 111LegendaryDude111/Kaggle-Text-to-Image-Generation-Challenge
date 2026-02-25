from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.generate_baseline import BaselineGenerationConfig
from generation.generate_report_bundle import (
    ReportBundleGenerationConfig,
    load_report_bundle_generation_config,
    run_report_bundle_generation,
    validate_report_bundle_from_config,
)


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=color)
    image.save(path)


def test_load_report_bundle_generation_config_reads_nested_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "export": {
                    "report_bundle_generation": {
                        "output_dir": "dreamlayer_export/custom_bundle",
                        "source_image_dir": "dreamlayer_export/custom_images",
                        "copy_images_to_output_dir": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_report_bundle_generation_config(config_path)

    assert config.output_dir == "dreamlayer_export/custom_bundle"
    assert config.source_image_dir == "dreamlayer_export/custom_images"
    assert config.copy_images_to_output_dir is False


def test_run_report_bundle_generation_creates_results_and_config(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "1|A cat on a chair.\n2|A dog near a lamp.\n",
        encoding="utf-8",
    )

    source_image_dir = tmp_path / "final_images"
    _write_png(source_image_dir / "0001.png", color=(255, 255, 255))
    _write_png(source_image_dir / "0002.png", color=(0, 0, 0))

    output_dir = tmp_path / "bundle"
    results_csv_path = output_dir / "results.csv"
    config_dreamlayer_path = output_dir / "config-dreamlayer.json"

    generation_config = BaselineGenerationConfig(
        model_name="unit-test-model",
        prompt_file=str(prompt_file),
        output_dir=str(tmp_path / "unused_output"),
        seed=42,
        guidance_scale=7.5,
        num_inference_steps=30,
        sampler="default",
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        use_structured_negative_prompt=False,
    )
    bundle_config = ReportBundleGenerationConfig(
        output_dir=str(output_dir),
        source_image_dir=str(source_image_dir),
        results_csv_path=str(results_csv_path),
        config_dreamlayer_path=str(config_dreamlayer_path),
        prompt_file=str(prompt_file),
        copy_images_to_output_dir=True,
        enforce_exact_one_image_per_prompt=True,
        validate_integrity=True,
        confirm_no_manual_edits=True,
    )

    report = run_report_bundle_generation(
        bundle_config,
        generation_config=generation_config,
    )

    assert report.prompt_count == 2
    assert report.image_count == 2
    assert report.integrity_valid is True
    assert report.no_manual_edits_confirmed is True

    assert (output_dir / "0001.png").exists()
    assert (output_dir / "0002.png").exists()
    assert results_csv_path.exists()
    assert config_dreamlayer_path.exists()

    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["run_id"] == "0001"
    assert rows[0]["prompt"] == "A cat on a chair."
    assert rows[0]["filenames"] == "0001.png"
    assert rows[1]["run_id"] == "0002"
    assert rows[1]["filenames"] == "0002.png"

    config_payload = json.loads(config_dreamlayer_path.read_text(encoding="utf-8"))
    assert config_payload["results_contract"]["row_count"] == 2
    assert config_payload["integrity"]["results_csv"]["path"] == "results.csv"


def test_validate_report_bundle_from_config_detects_manual_csv_edit(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("1|A single cat.\n", encoding="utf-8")

    source_image_dir = tmp_path / "final_images"
    _write_png(source_image_dir / "0001.png", color=(128, 128, 128))

    output_dir = tmp_path / "bundle"
    results_csv_path = output_dir / "results.csv"
    config_dreamlayer_path = output_dir / "config-dreamlayer.json"

    generation_config = BaselineGenerationConfig(
        model_name="unit-test-model",
        prompt_file=str(prompt_file),
        output_dir=str(tmp_path / "unused_output"),
        seed=42,
        guidance_scale=7.5,
        num_inference_steps=30,
        sampler="default",
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        use_structured_negative_prompt=False,
    )
    bundle_config = ReportBundleGenerationConfig(
        output_dir=str(output_dir),
        source_image_dir=str(source_image_dir),
        results_csv_path=str(results_csv_path),
        config_dreamlayer_path=str(config_dreamlayer_path),
        prompt_file=str(prompt_file),
        copy_images_to_output_dir=True,
    )

    run_report_bundle_generation(
        bundle_config,
        generation_config=generation_config,
    )

    tampered = results_csv_path.read_text(encoding="utf-8").replace(
        "A single cat.",
        "manually edited prompt",
    )
    results_csv_path.write_text(tampered, encoding="utf-8")

    with pytest.raises(RuntimeError):
        validate_report_bundle_from_config(config_dreamlayer_path)
