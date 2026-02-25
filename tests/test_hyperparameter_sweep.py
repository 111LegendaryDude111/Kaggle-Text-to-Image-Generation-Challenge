from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.hyperparameter_sweep import (
    HyperparameterSweepConfig,
    build_sweep_grid,
    load_hyperparameter_sweep_config,
    run_hyperparameter_sweep,
)
from evaluation.offline_runner import OfflineValidationConfig
from evaluation.prompt_loader import PromptRecord
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


def test_load_hyperparameter_sweep_config_reads_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "optimization": {
                    "hyperparameter_sweep": {
                        "output_dir": "experiments/custom_sweep",
                        "guidance_scale": [6.0, 7.0],
                        "steps": [20],
                        "sampler": "euler-a",
                        "seed": [10, 11],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_hyperparameter_sweep_config(config_path)

    assert config.output_dir == "experiments/custom_sweep"
    assert config.guidance_scales == [6.0, 7.0]
    assert config.num_inference_steps == [20]
    assert config.samplers == ["euler_a"]
    assert config.seeds == [10, 11]


def test_build_sweep_grid_is_deterministic() -> None:
    config = HyperparameterSweepConfig(
        guidance_scales=[6.0, 7.0],
        num_inference_steps=[20],
        samplers=["default", "euler_a"],
        seeds=[1, 2],
    )

    grid = build_sweep_grid(config)

    assert len(grid) == 8
    assert (
        grid[0].run_index,
        grid[0].guidance_scale,
        grid[0].sampler,
        grid[0].seed,
    ) == (
        1,
        6.0,
        "default",
        1,
    )
    assert (
        grid[-1].run_index,
        grid[-1].guidance_scale,
        grid[-1].sampler,
        grid[-1].seed,
    ) == (
        8,
        7.0,
        "euler_a",
        2,
    )


def test_run_hyperparameter_sweep_logs_runs_and_selects_best(tmp_path: Path) -> None:
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

        sampler_bonus = 0.2 if generation_config.sampler == "euler_a" else 0.0
        score = (
            (generation_config.guidance_scale * 0.01)
            + (generation_config.num_inference_steps * 0.001)
            + sampler_bonus
            + (generation_config.seed * 0.0001)
        )
        return FakeOfflineReport(
            prompt_count=len(prompts),
            average_precision=score / 2.0,
            average_recall=score / 2.0,
            average_f1=score,
            generation_runtime_sec=0.1,
            evaluation_runtime_sec=0.2,
            total_runtime_sec=0.3,
            report_path=str(Path(validation_config.report_path).resolve()),
            per_prompt_csv_path=str(
                Path(validation_config.per_prompt_csv_path).resolve()
            ),
        )

    sweep_config = HyperparameterSweepConfig(
        output_dir=str(tmp_path / "sweep"),
        guidance_scales=[6.0, 7.0],
        num_inference_steps=[20],
        samplers=["default", "euler_a"],
        seeds=[1, 2],
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

    report = run_hyperparameter_sweep(
        sweep_config,
        generation_config=generation_config,
        validation_config=validation_config,
        yolo_config=yolo_config,
        prompts=prompts,
        yolo_model=object(),
        pipeline_factory=fake_pipeline_factory,
        evaluation_runner=fake_evaluation_runner,
    )

    assert report.total_runs == 8
    assert report.best_config == {
        "guidance_scale": 7.0,
        "num_inference_steps": 20,
        "sampler": "euler_a",
        "seed": 2,
    }
    assert pipeline_factory_calls == [("default", "cpu"), ("euler_a", "cpu")]
    assert Path(report.summary_json_path).exists()
    assert Path(report.summary_csv_path).exists()

    loaded_json = json.loads(Path(report.summary_json_path).read_text(encoding="utf-8"))
    assert loaded_json["total_runs"] == 8
    assert loaded_json["best_config"]["sampler"] == "euler_a"

    with Path(report.summary_csv_path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert rows[0]["run_index"] == "1"
    assert rows[-1]["run_index"] == "8"
