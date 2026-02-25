from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_loader import PromptRecord
from generation.generate_baseline import (
    BaselineGenerationConfig,
    compute_prompt_seed,
    load_generation_config,
    normalize_sampler_name,
    run_baseline_generation,
)
from generation.negative_prompt_strategy import NegativePromptStrategyConfig


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
        image = Image.new(
            "RGB", (kwargs["width"], kwargs["height"]), color=(255, 255, 255)
        )
        return FakePipelineOutput(image)


def test_compute_prompt_seed_respects_strategy() -> None:
    assert compute_prompt_seed(42, 0, "fixed") == 42
    assert compute_prompt_seed(42, 3, "fixed") == 42
    assert compute_prompt_seed(42, 0, "incremental") == 42
    assert compute_prompt_seed(42, 3, "incremental") == 45


def test_run_baseline_generation_saves_prompt_id_png_and_passes_cfg(
    tmp_path: Path,
) -> None:
    config = BaselineGenerationConfig(
        model_name="unit-test-model",
        output_dir=str(tmp_path),
        seed=100,
        seed_strategy="incremental",
        guidance_scale=6.25,
        num_inference_steps=20,
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
    )
    prompts = [
        PromptRecord(prompt_id="0001", text="A dog near a bench."),
        PromptRecord(prompt_id="0002", text="A cat on a chair."),
    ]
    fake_pipeline = FakePipeline()

    records = run_baseline_generation(config, prompts=prompts, pipeline=fake_pipeline)

    assert [Path(record.output_path).name for record in records] == [
        "0001.png",
        "0002.png",
    ]
    assert (tmp_path / "0001.png").exists()
    assert (tmp_path / "0002.png").exists()

    assert fake_pipeline.calls[0]["guidance_scale"] == 6.25
    assert fake_pipeline.calls[0]["num_inference_steps"] == 20
    assert fake_pipeline.calls[0]["num_images_per_prompt"] == 1
    assert fake_pipeline.calls[0]["generator"].initial_seed() == 100
    assert fake_pipeline.calls[1]["generator"].initial_seed() == 101


def test_load_generation_config_reads_baseline_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "model_name": "mock-model",
                    "width": 64,
                    "height": 64,
                    "seed": 7,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_generation_config(config_path)

    assert config.model_name == "mock-model"
    assert config.width == 64
    assert config.height == 64
    assert config.seed == 7


def test_sampler_alias_normalization() -> None:
    assert normalize_sampler_name("Euler-A") == "euler_a"
    config = BaselineGenerationConfig(sampler="dpm_solver_multistep")
    assert config.sampler == "dpmpp_2m"


def test_resolution_validation_requires_multiple_of_8() -> None:
    try:
        BaselineGenerationConfig(width=66)
    except ValidationError:
        return
    raise AssertionError(
        "Expected resolution validation error for non-multiple-of-8 width"
    )


def test_run_baseline_generation_uses_structured_negative_prompt(
    tmp_path: Path,
) -> None:
    config = BaselineGenerationConfig(
        model_name="unit-test-model",
        output_dir=str(tmp_path),
        seed=1,
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        negative_prompt="overexposed highlights",
        use_structured_negative_prompt=True,
        negative_prompt_strategy=NegativePromptStrategyConfig(
            clutter_suppression_level="low",
            extract_expected_objects=False,
        ),
    )
    prompts = [
        PromptRecord(
            prompt_id="0001",
            text="A cat on a chair.",
            expected_objects=("cat", "chair"),
        )
    ]
    fake_pipeline = FakePipeline()

    run_baseline_generation(config, prompts=prompts, pipeline=fake_pipeline)

    negative_prompt = str(fake_pipeline.calls[0]["negative_prompt"]).lower()
    assert "overexposed highlights" in negative_prompt
    assert "background clutter" in negative_prompt
    assert "extra objects" in negative_prompt
    assert "target object set locked to: cat, chair" in negative_prompt
