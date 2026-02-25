from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.object_count_reinforcement import ObjectCountReinforcementConfig
from evaluation.prompt_loader import PromptRecord
from evaluation.realism_bias_tuning import RealismBiasTuningConfig
from generation.generate_baseline import BaselineGenerationConfig
from generation.generate_final import (
    FinalImageGenerationConfig,
    load_final_generation_config,
    resolve_frozen_final_config,
    run_final_image_generation,
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
        image = Image.new(
            "RGB",
            (kwargs["width"], kwargs["height"]),
            color=(255, 255, 255),
        )
        return FakePipelineOutput(image)


def test_load_final_generation_config_reads_export_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "export": {
                    "final_image_generation": {
                        "output_dir": "dreamlayer_export/custom_final",
                        "frozen_config_path": "dreamlayer_export/custom_frozen.json",
                        "use_best_hyperparameters": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_final_generation_config(config_path)

    assert config.output_dir == "dreamlayer_export/custom_final"
    assert config.frozen_config_path == "dreamlayer_export/custom_frozen.json"
    assert config.use_best_hyperparameters is False


def test_resolve_frozen_final_config_uses_best_reports(tmp_path: Path) -> None:
    hyper_report = tmp_path / "sweep_report.json"
    hyper_report.write_text(
        json.dumps(
            {
                "best_config": {
                    "guidance_scale": 8.2,
                    "num_inference_steps": 42,
                    "sampler": "euler_a",
                    "seed": 99,
                }
            }
        ),
        encoding="utf-8",
    )
    realism_report = tmp_path / "realism_report.json"
    realism_report.write_text(
        json.dumps({"best_variant": "photorealistic"}),
        encoding="utf-8",
    )
    count_report = tmp_path / "count_report.json"
    count_report.write_text(
        json.dumps({"best_variant": "count_reinforced"}),
        encoding="utf-8",
    )

    base = BaselineGenerationConfig(
        model_name="unit-test-model",
        seed=1,
        guidance_scale=7.5,
        num_inference_steps=30,
        sampler="default",
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        use_structured_negative_prompt=False,
    )
    final_config = FinalImageGenerationConfig(
        hyperparameter_report_path=str(hyper_report),
        realism_report_path=str(realism_report),
        object_count_report_path=str(count_report),
        require_reports=True,
    )

    runtime, frozen = resolve_frozen_final_config(
        base_generation_config=base,
        final_config=final_config,
    )

    assert runtime.guidance_scale == 8.2
    assert runtime.num_inference_steps == 42
    assert runtime.sampler == "euler_a"
    assert runtime.seed == 99
    assert frozen.realism_variant == "photorealistic"
    assert frozen.object_count_variant == "count_reinforced"


def test_run_final_image_generation_freezes_and_saves_prompt_id_png(
    tmp_path: Path,
) -> None:
    hyper_report = tmp_path / "sweep_report.json"
    hyper_report.write_text(
        json.dumps(
            {
                "best_config": {
                    "guidance_scale": 6.8,
                    "num_inference_steps": 25,
                    "sampler": "default",
                    "seed": 77,
                }
            }
        ),
        encoding="utf-8",
    )
    realism_report = tmp_path / "realism_report.json"
    realism_report.write_text(
        json.dumps({"best_variant": "photorealistic"}),
        encoding="utf-8",
    )
    count_report = tmp_path / "count_report.json"
    count_report.write_text(
        json.dumps({"best_variant": "count_reinforced"}),
        encoding="utf-8",
    )

    prompts = [
        PromptRecord(
            prompt_id="0001",
            text="Two dogs near a chair.",
            expected_objects=("dog", "chair"),
        ),
        PromptRecord(prompt_id="0002", text="A single cat.", expected_objects=("cat",)),
    ]
    generation_config = BaselineGenerationConfig(
        model_name="unit-test-model",
        output_dir=str(tmp_path / "unused"),
        seed=1,
        guidance_scale=7.5,
        num_inference_steps=30,
        sampler="default",
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        use_structured_negative_prompt=False,
    )
    final_config = FinalImageGenerationConfig(
        output_dir=str(tmp_path / "final_images"),
        frozen_config_path=str(tmp_path / "frozen.json"),
        report_path=str(tmp_path / "final_report.json"),
        hyperparameter_report_path=str(hyper_report),
        realism_report_path=str(realism_report),
        object_count_report_path=str(count_report),
        require_reports=True,
    )
    realism_tuning_config = RealismBiasTuningConfig()
    count_tuning_config = ObjectCountReinforcementConfig(
        use_spacy_count_inference=False,
    )
    fake_pipeline = FakePipeline()

    report = run_final_image_generation(
        final_config,
        generation_config=generation_config,
        prompts=prompts,
        pipeline=fake_pipeline,
        realism_tuning_config=realism_tuning_config,
        count_tuning_config=count_tuning_config,
    )

    assert report.prompt_count == 2
    assert report.generated_count == 2
    assert (tmp_path / "final_images" / "0001.png").exists()
    assert (tmp_path / "final_images" / "0002.png").exists()
    assert Path(report.frozen_config_path).exists()
    assert Path(report.report_path).exists()
    assert fake_pipeline.calls[0]["generator"].initial_seed() == 77
    assert fake_pipeline.calls[1]["generator"].initial_seed() == 78
    lowered_prompt = str(fake_pipeline.calls[0]["prompt"]).lower()
    assert "object count constraint:" in lowered_prompt
    assert "exactly 1 instance of dog" in lowered_prompt
    assert "photorealistic" in str(fake_pipeline.calls[0]["prompt"]).lower()

    frozen_payload = json.loads(
        Path(report.frozen_config_path).read_text(encoding="utf-8")
    )
    assert frozen_payload["generation_config"]["guidance_scale"] == 6.8
    assert frozen_payload["realism_variant"] == "photorealistic"
    assert frozen_payload["object_count_variant"] == "count_reinforced"
