from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_loader import PromptRecord
from generation.generate_multisample import (
    MultiSampleGenerationConfig,
    compute_guidance_schedule,
    load_multisample_config,
    run_multisample_generation,
)


class FakePipelineOutput:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "FakePipeline":
        return self

    def __call__(self, **kwargs: Any) -> FakePipelineOutput:
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(255, 255, 255))
        return FakePipelineOutput(image)


def test_compute_guidance_schedule_is_symmetric() -> None:
    schedule = compute_guidance_schedule(
        base_guidance_scale=7.5,
        guidance_scale_step=0.5,
        num_variants=4,
    )
    assert schedule == [6.75, 7.25, 7.75, 8.25]


def test_run_multisample_generates_variants_and_selects_best(tmp_path: Path) -> None:
    evaluator_calls: list[dict[str, Any]] = []

    def evaluator(**kwargs: Any) -> dict[str, Any]:
        evaluator_calls.append(kwargs)
        return {"score": float(kwargs["variant_index"]), "source": "unit_test"}

    config = MultiSampleGenerationConfig(
        model_name="unit-test-model",
        prompt_file="unused.txt",
        output_dir=str(tmp_path / "unused"),
        final_output_dir=str(tmp_path / "final"),
        temp_output_dir=str(tmp_path / "temp"),
        metadata_path=str(tmp_path / "metadata.json"),
        num_variants=4,
        seed=123,
        seed_strategy="incremental",
        guidance_scale=7.5,
        guidance_scale_step=0.5,
        num_inference_steps=20,
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        keep_temp_images=True,
        use_structured_negative_prompt=False,
    )
    prompts = [PromptRecord(prompt_id="0001", text="A cat on a chair.")]
    fake_pipeline = FakePipeline()

    report = run_multisample_generation(
        config,
        prompts=prompts,
        pipeline=fake_pipeline,
        evaluator=evaluator,
    )

    assert len(report.variants) == 4
    assert len(evaluator_calls) == 4
    assert report.selections[0].best_variant_index == 3
    assert Path(report.selections[0].final_image_path).exists()

    for variant_index in range(1, 5):
        expected_temp = tmp_path / "temp" / "0001" / f"0001_v{variant_index:02d}.png"
        assert expected_temp.exists()

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_count"] == 1
    assert metadata["num_variants"] == 4
    assert len(metadata["variants"]) == 4


def test_run_multisample_can_cleanup_temp_images(tmp_path: Path) -> None:
    config = MultiSampleGenerationConfig(
        model_name="unit-test-model",
        prompt_file="unused.txt",
        output_dir=str(tmp_path / "unused"),
        final_output_dir=str(tmp_path / "final"),
        temp_output_dir=str(tmp_path / "temp"),
        metadata_path=str(tmp_path / "metadata.json"),
        num_variants=4,
        seed=1,
        guidance_scale=7.5,
        num_inference_steps=20,
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        keep_temp_images=False,
        use_structured_negative_prompt=False,
    )
    prompts = [PromptRecord(prompt_id="0001", text="A dog near a bench.")]
    fake_pipeline = FakePipeline()

    run_multisample_generation(config, prompts=prompts, pipeline=fake_pipeline, evaluator=lambda **_: 0.0)

    assert (tmp_path / "final" / "0001.png").exists()
    assert not (tmp_path / "temp" / "0001").exists()


def test_load_multisample_config_reads_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "multisample": {
                    "model_name": "mock-model",
                    "num_variants": 4,
                    "width": 64,
                    "height": 64,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_multisample_config(config_path)

    assert config.model_name == "mock-model"
    assert config.num_variants == 4
    assert config.width == 64
    assert config.height == 64
