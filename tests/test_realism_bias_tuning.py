from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.offline_runner import OfflineValidationConfig
from evaluation.prompt_loader import PromptRecord
from evaluation.realism_bias_tuning import (
    RealismBiasTuningConfig,
    build_variant_prompt_text,
    load_realism_bias_tuning_config,
    run_realism_bias_tuning,
)
from evaluation.yolo_detector import YoloDetectionConfig
from generation.generate_baseline import BaselineGenerationConfig


@dataclass(frozen=True, slots=True)
class FakeOfflineReport:
    prompt_count: int
    average_precision: float
    average_recall: float
    average_f1: float
    generation_runtime_sec: float
    evaluation_runtime_sec: float
    total_runtime_sec: float
    report_path: str
    per_prompt_csv_path: str


def test_load_realism_bias_tuning_config_reads_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "optimization": {
                    "realism_bias_tuning": {
                        "output_dir": "experiments/custom_realism",
                        "styles": ["baseline", "realistic", "simple-background"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_realism_bias_tuning_config(config_path)

    assert config.output_dir == "experiments/custom_realism"
    assert config.variants == [
        "baseline",
        "photorealistic",
        "photorealistic_simple_background",
    ]


def test_build_variant_prompt_text_applies_expected_profile_terms() -> None:
    config = RealismBiasTuningConfig()
    baseline = build_variant_prompt_text(
        "A cat on a chair.",
        variant="baseline",
        config=config,
    )
    realistic = build_variant_prompt_text(
        "A cat on a chair.",
        variant="photorealistic",
        config=config,
    )
    simplified = build_variant_prompt_text(
        "A cat on a chair.",
        variant="photorealistic_simple_background",
        config=config,
    )
    stylized = build_variant_prompt_text(
        "A cat on a chair.",
        variant="stylized",
        config=config,
    )

    assert baseline == "A cat on a chair."
    assert "photorealistic" in realistic.lower()
    assert "simple uncluttered background" in simplified.lower()
    assert "stylized artistic illustration" in stylized.lower()


def test_run_realism_bias_tuning_compares_stylized_vs_realistic(tmp_path: Path) -> None:
    pipeline_factory_calls: list[tuple[str, str]] = []

    def fake_pipeline_factory(config: BaselineGenerationConfig, device: str) -> object:
        pipeline_factory_calls.append((config.sampler, device))
        return object()

    def fake_evaluation_runner(
        validation_config: OfflineValidationConfig,
        *,
        generation_config: BaselineGenerationConfig,
        yolo_config: YoloDetectionConfig,
        prompts: list[PromptRecord] | None = None,
        generation_pipeline: Any = None,
        yolo_model: Any = None,
    ) -> FakeOfflineReport:
        assert prompts is not None
        assert generation_pipeline is not None
        assert yolo_model is not None

        text = prompts[0].text.lower()
        if "stylized artistic illustration" in text:
            score = 0.20
        elif "simple uncluttered background" in text:
            score = 0.61
        elif "photorealistic" in text:
            score = 0.54
        else:
            score = 0.40

        return FakeOfflineReport(
            prompt_count=len(prompts),
            average_precision=score,
            average_recall=score,
            average_f1=score,
            generation_runtime_sec=0.1,
            evaluation_runtime_sec=0.2,
            total_runtime_sec=0.3,
            report_path=str(Path(validation_config.report_path).resolve()),
            per_prompt_csv_path=str(
                Path(validation_config.per_prompt_csv_path).resolve()
            ),
        )

    tuning_config = RealismBiasTuningConfig(
        output_dir=str(tmp_path / "realism"),
        variants=[
            "baseline",
            "photorealistic",
            "photorealistic_simple_background",
            "stylized",
        ],
    )
    generation_config = BaselineGenerationConfig(
        model_name="unit-test-model",
        output_dir=str(tmp_path / "unused_output"),
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
    validation_config = OfflineValidationConfig()
    yolo_config = YoloDetectionConfig(confidence_threshold=0.25)
    prompts = [PromptRecord(prompt_id="0001", text="A dog.", expected_objects=("dog",))]

    report = run_realism_bias_tuning(
        tuning_config,
        generation_config=generation_config,
        validation_config=validation_config,
        yolo_config=yolo_config,
        prompts=prompts,
        yolo_model=object(),
        pipeline_factory=fake_pipeline_factory,
        evaluation_runner=fake_evaluation_runner,
    )

    assert report.total_variants == 4
    assert report.best_variant == "photorealistic_simple_background"
    assert (
        report.comparisons.best_realistic_variant == "photorealistic_simple_background"
    )
    assert round(float(report.comparisons.photorealistic_vs_baseline_delta_f1), 2) == 0.14
    assert (
        round(float(report.comparisons.simple_background_vs_photorealistic_delta_f1), 2)
        == 0.07
    )
    assert round(float(report.comparisons.stylized_vs_best_realistic_delta_f1), 2) == -0.41
    assert pipeline_factory_calls == [("default", "cpu")]
    assert Path(report.summary_json_path).exists()
    assert Path(report.summary_csv_path).exists()

    loaded_json = json.loads(Path(report.summary_json_path).read_text(encoding="utf-8"))
    assert loaded_json["best_variant"] == "photorealistic_simple_background"

    with Path(report.summary_csv_path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
