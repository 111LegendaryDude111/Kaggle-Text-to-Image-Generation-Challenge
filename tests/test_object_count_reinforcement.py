from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.object_count_reinforcement import (
    ObjectCountReinforcementConfig,
    build_count_reinforced_prompt_text,
    load_object_count_reinforcement_config,
    run_object_count_reinforcement,
)
from evaluation.offline_runner import OfflineValidationConfig
from evaluation.prompt_loader import PromptRecord
from evaluation.yolo_detector import YoloDetectionConfig
from generation.generate_baseline import BaselineGenerationConfig


@dataclass(frozen=True, slots=True)
class FakePromptResult:
    prompt_id: str
    false_positives: int
    recall: float
    f1: float


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
    results: tuple[FakePromptResult, ...]


def test_load_object_count_reinforcement_config_reads_nested_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "optimization": {
                    "object_count_reinforcement": {
                        "output_dir": "experiments/custom_count",
                        "profiles": ["baseline", "count"],
                        "max_false_positive_examples": 5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_object_count_reinforcement_config(config_path)

    assert config.output_dir == "experiments/custom_count"
    assert config.variants == ["baseline", "count_reinforced"]
    assert config.max_false_positive_examples == 5


def test_build_count_reinforced_prompt_text_injects_exactly_n_patterns() -> None:
    prompt = PromptRecord(
        prompt_id="0001",
        text="Two dogs near a chair.",
        expected_objects=("dog", "chair"),
    )
    config = ObjectCountReinforcementConfig()

    reinforced = build_count_reinforced_prompt_text(
        prompt,
        object_counts={"dog": 2, "chair": 1},
        config=config,
    )

    lowered = reinforced.lower()
    assert "exactly 2 instances of dog" in lowered
    assert "exactly 1 instance of chair" in lowered
    assert "ensure every required object is present" in lowered


def test_run_object_count_reinforcement_analyzes_false_positive_drift(
    tmp_path: Path,
) -> None:
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

        is_reinforced = "object count constraint:" in prompts[0].text.lower()
        if is_reinforced:
            return FakeOfflineReport(
                prompt_count=2,
                average_precision=0.55,
                average_recall=0.70,
                average_f1=0.60,
                generation_runtime_sec=0.1,
                evaluation_runtime_sec=0.2,
                total_runtime_sec=0.3,
                report_path=str(Path(validation_config.report_path).resolve()),
                per_prompt_csv_path=str(
                    Path(validation_config.per_prompt_csv_path).resolve()
                ),
                results=(
                    FakePromptResult(
                        prompt_id="0001", false_positives=2, recall=0.8, f1=0.7
                    ),
                    FakePromptResult(
                        prompt_id="0002", false_positives=1, recall=0.6, f1=0.5
                    ),
                ),
            )
        return FakeOfflineReport(
            prompt_count=2,
            average_precision=0.50,
            average_recall=0.50,
            average_f1=0.45,
            generation_runtime_sec=0.1,
            evaluation_runtime_sec=0.2,
            total_runtime_sec=0.3,
            report_path=str(Path(validation_config.report_path).resolve()),
            per_prompt_csv_path=str(
                Path(validation_config.per_prompt_csv_path).resolve()
            ),
            results=(
                FakePromptResult(
                    prompt_id="0001", false_positives=0, recall=0.4, f1=0.35
                ),
                FakePromptResult(
                    prompt_id="0002", false_positives=1, recall=0.6, f1=0.55
                ),
            ),
        )

    count_config = ObjectCountReinforcementConfig(
        output_dir=str(tmp_path / "count"),
        variants=["baseline", "count_reinforced"],
        use_spacy_count_inference=False,
        max_false_positive_examples=5,
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
    prompts = [
        PromptRecord(
            prompt_id="0001",
            text="Two dogs near a chair.",
            expected_objects=("dog", "chair"),
        ),
        PromptRecord(prompt_id="0002", text="A single cat.", expected_objects=("cat",)),
    ]

    report = run_object_count_reinforcement(
        count_config,
        generation_config=generation_config,
        validation_config=validation_config,
        yolo_config=yolo_config,
        prompts=prompts,
        yolo_model=object(),
        pipeline_factory=fake_pipeline_factory,
        evaluation_runner=fake_evaluation_runner,
    )

    assert report.total_variants == 2
    assert report.best_variant == "count_reinforced"
    assert round(float(report.comparisons.recall_delta), 2) == 0.20
    assert round(float(report.comparisons.f1_delta), 2) == 0.15
    assert report.comparisons.false_positive_delta == 2
    assert (
        round(float(report.comparisons.average_false_positive_delta_per_prompt), 2)
        == 1.0
    )
    assert len(report.comparisons.prompts_with_increased_false_positives) == 1
    assert (
        report.comparisons.prompts_with_increased_false_positives[0].prompt_id == "0001"
    )
    assert pipeline_factory_calls == [("default", "cpu")]
    assert Path(report.summary_json_path).exists()
    assert Path(report.summary_csv_path).exists()

    loaded_json = json.loads(Path(report.summary_json_path).read_text(encoding="utf-8"))
    assert loaded_json["best_variant"] == "count_reinforced"

    with Path(report.summary_csv_path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
