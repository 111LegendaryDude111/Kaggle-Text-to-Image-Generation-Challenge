"""Offline leaderboard simulation: prompt -> image -> YOLO -> object-level F1."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

try:
    from evaluation.f1_metric import score_object_detection
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import load_prompts_with_expected_objects
    from evaluation.yolo_detector import (
        YoloDetectionConfig,
        YoloDetectionResult,
        YoloModelLike,
        build_yolo_model,
        detect_objects_in_image,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        GeneratedImageRecord,
        TextToImagePipeline,
        run_baseline_generation,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.f1_metric import score_object_detection
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import load_prompts_with_expected_objects
    from evaluation.yolo_detector import (
        YoloDetectionConfig,
        YoloDetectionResult,
        YoloModelLike,
        build_yolo_model,
        detect_objects_in_image,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        GeneratedImageRecord,
        TextToImagePipeline,
        run_baseline_generation,
    )


class OfflineValidationConfig(BaseModel):
    """Configuration for offline validation flow and reporting."""
    model_config = ConfigDict(extra="ignore")
    prompt_file: str | None = None
    spacy_model: str = "en_core_web_sm"
    keep_unmapped_expected_objects: bool = False
    generated_output_dir: str | None = None
    report_path: str = "experiments/offline_validation/report.json"
    per_prompt_csv_path: str = "experiments/offline_validation/per_prompt_metrics.csv"


@dataclass(frozen=True, slots=True)
class OfflinePromptResult:
    """Per-prompt offline validation record."""

    prompt_id: str
    prompt_text: str
    image_path: str
    seed: int
    expected_objects: tuple[str, ...]
    detected_objects: tuple[str, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    num_raw_detections: int
    num_kept_detections: int
    evaluation_runtime_sec: float
    def to_json_dict(self) -> dict[str, Any]:
        """Serialize per-prompt result to JSON-compatible shape."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "image_path": self.image_path,
            "seed": self.seed,
            "expected_objects": list(self.expected_objects),
            "detected_objects": list(self.detected_objects),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "num_raw_detections": self.num_raw_detections,
            "num_kept_detections": self.num_kept_detections,
            "evaluation_runtime_sec": self.evaluation_runtime_sec,
        }


@dataclass(frozen=True, slots=True)
class OfflineValidationReport:
    """Aggregate offline validation report for one run."""

    prompt_count: int
    average_precision: float
    average_recall: float
    average_f1: float
    generation_runtime_sec: float
    evaluation_runtime_sec: float
    total_runtime_sec: float
    report_path: str
    per_prompt_csv_path: str
    results: tuple[OfflinePromptResult, ...]
    def to_json_dict(self) -> dict[str, Any]:
        """Serialize report to JSON-compatible shape."""
        return {
            "prompt_count": self.prompt_count,
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
            "average_f1": self.average_f1,
            "generation_runtime_sec": self.generation_runtime_sec,
            "evaluation_runtime_sec": self.evaluation_runtime_sec,
            "total_runtime_sec": self.total_runtime_sec,
            "report_path": self.report_path,
            "per_prompt_csv_path": self.per_prompt_csv_path,
            "results": [item.to_json_dict() for item in self.results],
        }


def load_offline_validation_config(config_path: str | Path) -> OfflineValidationConfig:
    """Load offline runner config section from JSON or return defaults."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("evaluation"), dict):
        payload = payload["evaluation"]
    if isinstance(payload.get("offline_runner"), dict):
        payload = payload["offline_runner"]
    else:
        payload = {}

    return OfflineValidationConfig.model_validate(payload)


def _resolve_runtime_generation_config(
    generation_config: BaselineGenerationConfig,
    validation_config: OfflineValidationConfig,
) -> BaselineGenerationConfig:
    updates: dict[str, Any] = {}
    if validation_config.generated_output_dir is not None:
        updates["output_dir"] = validation_config.generated_output_dir
    if not updates:
        return generation_config
    return generation_config.model_copy(update=updates)


def _load_runtime_prompts(
    *,
    validation_config: OfflineValidationConfig,
    generation_config: BaselineGenerationConfig,
) -> list[PromptRecord]:
    prompt_file = validation_config.prompt_file or generation_config.prompt_file
    return load_prompts_with_expected_objects(
        prompt_file,
        spacy_model=validation_config.spacy_model,
        keep_unmapped=validation_config.keep_unmapped_expected_objects,
    )


def _build_prompt_lookup(prompts: Sequence[PromptRecord]) -> dict[str, PromptRecord]:
    lookup = {prompt.prompt_id: prompt for prompt in prompts}
    if len(lookup) != len(prompts):
        raise ValueError("Prompt IDs must be unique for offline validation.")
    return lookup


def _score_prompt(
    *,
    prompt: PromptRecord,
    generated: GeneratedImageRecord,
    yolo_result: YoloDetectionResult,
    evaluation_runtime_sec: float,
) -> OfflinePromptResult:
    metric = score_object_detection(
        expected_objects=prompt.expected_objects,
        detected_objects=yolo_result.detected_labels,
    )
    return OfflinePromptResult(
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.text,
        image_path=generated.output_path,
        seed=generated.seed,
        expected_objects=metric.expected_objects,
        detected_objects=metric.detected_objects,
        true_positives=metric.true_positives,
        false_positives=metric.false_positives,
        false_negatives=metric.false_negatives,
        precision=metric.precision,
        recall=metric.recall,
        f1=metric.f1,
        num_raw_detections=yolo_result.num_raw_detections,
        num_kept_detections=len(yolo_result.detections),
        evaluation_runtime_sec=evaluation_runtime_sec,
    )


def write_offline_validation_report(
    report: OfflineValidationReport,
    *,
    report_path: str | Path,
    per_prompt_csv_path: str | Path,
) -> None:
    """Write report JSON and per-prompt CSV outputs."""
    json_destination = Path(report_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")

    csv_destination = Path(per_prompt_csv_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "seed",
                "image_path",
                "expected_objects",
                "detected_objects",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "num_raw_detections",
                "num_kept_detections",
                "evaluation_runtime_sec",
            ],
        )
        writer.writeheader()
        for item in report.results:
            writer.writerow(
                {
                    "prompt_id": item.prompt_id,
                    "seed": item.seed,
                    "image_path": item.image_path,
                    "expected_objects": ",".join(item.expected_objects),
                    "detected_objects": ",".join(item.detected_objects),
                    "true_positives": item.true_positives,
                    "false_positives": item.false_positives,
                    "false_negatives": item.false_negatives,
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                    "num_raw_detections": item.num_raw_detections,
                    "num_kept_detections": item.num_kept_detections,
                    "evaluation_runtime_sec": item.evaluation_runtime_sec,
                }
            )


def run_offline_validation(
    validation_config: OfflineValidationConfig,
    *,
    generation_config: BaselineGenerationConfig,
    yolo_config: YoloDetectionConfig,
    prompts: Sequence[PromptRecord] | None = None,
    generation_pipeline: TextToImagePipeline | None = None,
    yolo_model: YoloModelLike | None = None,
) -> OfflineValidationReport:
    """Run prompt->generation->detection->F1 pipeline and persist metrics."""
    total_start = time.perf_counter()
    runtime_generation_config = _resolve_runtime_generation_config(
        generation_config=generation_config,
        validation_config=validation_config,
    )

    prompt_records = list(prompts) if prompts is not None else _load_runtime_prompts(
        validation_config=validation_config,
        generation_config=runtime_generation_config,
    )
    if not prompt_records:
        raise ValueError("No prompts available for offline validation.")
    prompt_lookup = _build_prompt_lookup(prompt_records)

    generation_start = time.perf_counter()
    generated_records = run_baseline_generation(
        runtime_generation_config,
        prompts=prompt_records,
        pipeline=generation_pipeline,
    )
    generation_runtime_sec = time.perf_counter() - generation_start

    detector = yolo_model if yolo_model is not None else build_yolo_model(yolo_config)

    evaluation_start = time.perf_counter()
    scored_results: list[OfflinePromptResult] = []
    for generated in generated_records:
        prompt = prompt_lookup.get(generated.prompt_id)
        if prompt is None:
            raise KeyError(f"Generated prompt_id '{generated.prompt_id}' not found in prompt set.")

        prompt_eval_start = time.perf_counter()
        detection = detect_objects_in_image(generated.output_path, model=detector, config=yolo_config)
        prompt_eval_runtime = time.perf_counter() - prompt_eval_start
        scored_results.append(
            _score_prompt(
                prompt=prompt,
                generated=generated,
                yolo_result=detection,
                evaluation_runtime_sec=prompt_eval_runtime,
            )
        )

    evaluation_runtime_sec = time.perf_counter() - evaluation_start

    prompt_count = len(scored_results)
    average_precision = sum(item.precision for item in scored_results) / prompt_count
    average_recall = sum(item.recall for item in scored_results) / prompt_count
    average_f1 = sum(item.f1 for item in scored_results) / prompt_count
    total_runtime_sec = time.perf_counter() - total_start

    report = OfflineValidationReport(
        prompt_count=prompt_count,
        average_precision=average_precision,
        average_recall=average_recall,
        average_f1=average_f1,
        generation_runtime_sec=generation_runtime_sec,
        evaluation_runtime_sec=evaluation_runtime_sec,
        total_runtime_sec=total_runtime_sec,
        report_path=str(Path(validation_config.report_path).resolve()),
        per_prompt_csv_path=str(Path(validation_config.per_prompt_csv_path).resolve()),
        results=tuple(scored_results),
    )
    write_offline_validation_report(
        report,
        report_path=validation_config.report_path,
        per_prompt_csv_path=validation_config.per_prompt_csv_path,
    )
    return report
