#!/usr/bin/env python3
"""Run deterministic hyperparameter grid search for object-level F1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

try:
    from evaluation.offline_runner import (
        OfflineValidationConfig,
        run_offline_validation,
        load_offline_validation_config,
    )
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import load_prompts_with_expected_objects
    from evaluation.yolo_detector import (
        YoloDetectionConfig,
        YoloModelLike,
        build_yolo_model,
        load_yolo_detection_config,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        TextToImagePipeline,
        build_pipeline,
        load_generation_config,
        normalize_sampler_name,
        resolve_device,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.offline_runner import (
        OfflineValidationConfig,
        run_offline_validation,
        load_offline_validation_config,
    )
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import load_prompts_with_expected_objects
    from evaluation.yolo_detector import (
        YoloDetectionConfig,
        YoloModelLike,
        build_yolo_model,
        load_yolo_detection_config,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        TextToImagePipeline,
        build_pipeline,
        load_generation_config,
        normalize_sampler_name,
        resolve_device,
    )


def _dedupe_preserve_order(values: Sequence[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _parse_csv_arg(raw: str, *, cast: type[float] | type[int] | type[str]) -> list[Any]:
    values: list[Any] = []
    for item in raw.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        values.append(cast(normalized))
    if not values:
        raise ValueError("CSV override produced an empty list.")
    return values


class OfflineValidationReportLike(Protocol):
    """Shape required from offline validation report objects."""

    prompt_count: int
    average_precision: float
    average_recall: float
    average_f1: float
    generation_runtime_sec: float
    evaluation_runtime_sec: float
    total_runtime_sec: float
    report_path: str
    per_prompt_csv_path: str


class PipelineFactory(Protocol):
    """Factory contract for creating text-to-image pipelines."""

    def __call__(
        self, config: BaselineGenerationConfig, device: str
    ) -> TextToImagePipeline:
        """Create a pipeline for one sampler group."""


class OfflineValidationRunner(Protocol):
    """Callable contract for running one offline validation pass."""

    def __call__(
        self,
        validation_config: OfflineValidationConfig,
        *,
        generation_config: BaselineGenerationConfig,
        yolo_config: YoloDetectionConfig,
        prompts: Sequence[PromptRecord] | None = None,
        generation_pipeline: TextToImagePipeline | None = None,
        yolo_model: YoloModelLike | None = None,
    ) -> OfflineValidationReportLike:
        """Run offline validation for one hyperparameter set."""


class HyperparameterSweepConfig(BaseModel):
    """Search-space and output settings for hyperparameter sweep."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "experiments/hyperparameter_sweep"
    guidance_scales: list[float] = Field(
        default_factory=lambda: [7.5],
        validation_alias=AliasChoices("guidance_scales", "guidance_scale"),
    )
    num_inference_steps: list[int] = Field(
        default_factory=lambda: [30],
        validation_alias=AliasChoices("num_inference_steps", "steps"),
    )
    samplers: list[str] = Field(
        default_factory=lambda: ["default"],
        validation_alias=AliasChoices("samplers", "sampler"),
    )
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        validation_alias=AliasChoices("seeds", "seed"),
    )

    @field_validator(
        "guidance_scales",
        "num_inference_steps",
        "samplers",
        "seeds",
        mode="before",
    )
    @classmethod
    def _coerce_to_list(cls, value: Any) -> list[Any]:
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @field_validator("guidance_scales")
    @classmethod
    def _validate_guidance_scales(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]
        if any(item < 0.0 for item in normalized):
            raise ValueError("guidance_scales must contain non-negative values")
        deduped = _dedupe_preserve_order(normalized)
        if not deduped:
            raise ValueError("guidance_scales cannot be empty")
        return deduped

    @field_validator("num_inference_steps")
    @classmethod
    def _validate_steps(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]
        if any(item < 1 for item in normalized):
            raise ValueError("num_inference_steps must be >= 1")
        deduped = _dedupe_preserve_order(normalized)
        if not deduped:
            raise ValueError("num_inference_steps cannot be empty")
        return deduped

    @field_validator("samplers")
    @classmethod
    def _validate_samplers(cls, value: list[str]) -> list[str]:
        normalized = [normalize_sampler_name(str(item)) for item in value]
        deduped = _dedupe_preserve_order(normalized)
        if not deduped:
            raise ValueError("samplers cannot be empty")
        return deduped

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]
        if any(item < 0 for item in normalized):
            raise ValueError("seeds must be >= 0")
        deduped = _dedupe_preserve_order(normalized)
        if not deduped:
            raise ValueError("seeds cannot be empty")
        return deduped


@dataclass(frozen=True, slots=True)
class HyperparameterCombination:
    """One grid-search point."""

    run_index: int
    guidance_scale: float
    num_inference_steps: int
    sampler: str
    seed: int


@dataclass(frozen=True, slots=True)
class HyperparameterSweepRunRecord:
    """Run-level metrics for one tested hyperparameter combination."""

    run_index: int
    guidance_scale: float
    num_inference_steps: int
    sampler: str
    seed: int
    prompt_count: int
    average_precision: float
    average_recall: float
    average_f1: float
    generation_runtime_sec: float
    evaluation_runtime_sec: float
    total_runtime_sec: float
    generated_output_dir: str
    report_path: str
    per_prompt_csv_path: str

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize sweep run record to JSON-compatible shape."""
        return {
            "run_index": self.run_index,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "sampler": self.sampler,
            "seed": self.seed,
            "prompt_count": self.prompt_count,
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
            "average_f1": self.average_f1,
            "generation_runtime_sec": self.generation_runtime_sec,
            "evaluation_runtime_sec": self.evaluation_runtime_sec,
            "total_runtime_sec": self.total_runtime_sec,
            "generated_output_dir": self.generated_output_dir,
            "report_path": self.report_path,
            "per_prompt_csv_path": self.per_prompt_csv_path,
        }


@dataclass(frozen=True, slots=True)
class HyperparameterSweepReport:
    """Aggregate hyperparameter sweep output."""

    output_dir: str
    summary_json_path: str
    summary_csv_path: str
    total_runs: int
    best_run_index: int
    best_average_f1: float
    best_config: dict[str, Any]
    runs: tuple[HyperparameterSweepRunRecord, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize sweep report to JSON-compatible shape."""
        return {
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_csv_path": self.summary_csv_path,
            "total_runs": self.total_runs,
            "best_run_index": self.best_run_index,
            "best_average_f1": self.best_average_f1,
            "best_config": self.best_config,
            "runs": [item.to_json_dict() for item in self.runs],
        }


def load_hyperparameter_sweep_config(
    config_path: str | Path,
) -> HyperparameterSweepConfig:
    """Load sweep config section from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("optimization"), dict):
        payload = payload["optimization"]
    if isinstance(payload.get("hyperparameter_sweep"), dict):
        payload = payload["hyperparameter_sweep"]

    return HyperparameterSweepConfig.model_validate(payload)


def build_sweep_grid(
    config: HyperparameterSweepConfig,
) -> list[HyperparameterCombination]:
    """Build deterministic cartesian-product sweep grid."""
    combinations: list[HyperparameterCombination] = []
    run_index = 1
    for guidance_scale in config.guidance_scales:
        for num_inference_steps in config.num_inference_steps:
            for sampler in config.samplers:
                for seed in config.seeds:
                    combinations.append(
                        HyperparameterCombination(
                            run_index=run_index,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            sampler=sampler,
                            seed=seed,
                        )
                    )
                    run_index += 1
    return combinations


def _resolve_prompt_records(
    *,
    validation_config: OfflineValidationConfig,
    generation_config: BaselineGenerationConfig,
    prompts: Sequence[PromptRecord] | None = None,
) -> list[PromptRecord]:
    if prompts is not None:
        return list(prompts)

    prompt_file = validation_config.prompt_file or generation_config.prompt_file
    return load_prompts_with_expected_objects(
        prompt_file,
        spacy_model=validation_config.spacy_model,
        keep_unmapped=validation_config.keep_unmapped_expected_objects,
    )


def write_hyperparameter_sweep_report(report: HyperparameterSweepReport) -> None:
    """Write sweep summary report JSON + CSV."""
    summary_json_path = Path(report.summary_json_path)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    summary_csv_path = Path(report.summary_csv_path)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_index",
                "guidance_scale",
                "num_inference_steps",
                "sampler",
                "seed",
                "prompt_count",
                "average_precision",
                "average_recall",
                "average_f1",
                "generation_runtime_sec",
                "evaluation_runtime_sec",
                "total_runtime_sec",
                "generated_output_dir",
                "report_path",
                "per_prompt_csv_path",
            ],
        )
        writer.writeheader()
        for run in report.runs:
            writer.writerow(run.to_json_dict())


def run_hyperparameter_sweep(
    sweep_config: HyperparameterSweepConfig,
    *,
    generation_config: BaselineGenerationConfig,
    validation_config: OfflineValidationConfig,
    yolo_config: YoloDetectionConfig,
    prompts: Sequence[PromptRecord] | None = None,
    yolo_model: YoloModelLike | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
    evaluation_runner: OfflineValidationRunner = run_offline_validation,
) -> HyperparameterSweepReport:
    """Run hyperparameter sweep and return aggregate report."""
    grid = build_sweep_grid(sweep_config)
    if not grid:
        raise ValueError("Hyperparameter grid is empty.")

    output_dir = Path(sweep_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = _resolve_prompt_records(
        validation_config=validation_config,
        generation_config=generation_config,
        prompts=prompts,
    )
    if not prompt_records:
        raise ValueError("No prompts available for hyperparameter sweep.")

    active_yolo_model = (
        yolo_model if yolo_model is not None else build_yolo_model(yolo_config)
    )

    grouped_by_sampler: dict[str, list[HyperparameterCombination]] = {}
    for combination in grid:
        grouped_by_sampler.setdefault(combination.sampler, []).append(combination)

    run_records: list[HyperparameterSweepRunRecord] = []
    for sampler, combinations in grouped_by_sampler.items():
        sampler_generation_config = generation_config.model_copy(
            update={"sampler": sampler}
        )
        device = resolve_device(sampler_generation_config.device)
        pipeline = pipeline_factory(sampler_generation_config, device)

        for combination in combinations:
            run_dir = output_dir / f"run_{combination.run_index:04d}"
            generated_output_dir = run_dir / "generated"
            report_path = run_dir / "report.json"
            per_prompt_csv_path = run_dir / "per_prompt_metrics.csv"

            runtime_generation_config = sampler_generation_config.model_copy(
                update={
                    "output_dir": str(generated_output_dir),
                    "guidance_scale": combination.guidance_scale,
                    "num_inference_steps": combination.num_inference_steps,
                    "seed": combination.seed,
                }
            )
            runtime_validation_config = validation_config.model_copy(
                update={
                    "generated_output_dir": str(generated_output_dir),
                    "report_path": str(report_path),
                    "per_prompt_csv_path": str(per_prompt_csv_path),
                }
            )

            validation_report = evaluation_runner(
                runtime_validation_config,
                generation_config=runtime_generation_config,
                yolo_config=yolo_config,
                prompts=prompt_records,
                generation_pipeline=pipeline,
                yolo_model=active_yolo_model,
            )
            run_records.append(
                HyperparameterSweepRunRecord(
                    run_index=combination.run_index,
                    guidance_scale=combination.guidance_scale,
                    num_inference_steps=combination.num_inference_steps,
                    sampler=combination.sampler,
                    seed=combination.seed,
                    prompt_count=int(validation_report.prompt_count),
                    average_precision=float(validation_report.average_precision),
                    average_recall=float(validation_report.average_recall),
                    average_f1=float(validation_report.average_f1),
                    generation_runtime_sec=float(
                        validation_report.generation_runtime_sec
                    ),
                    evaluation_runtime_sec=float(
                        validation_report.evaluation_runtime_sec
                    ),
                    total_runtime_sec=float(validation_report.total_runtime_sec),
                    generated_output_dir=str(generated_output_dir.resolve()),
                    report_path=str(Path(validation_report.report_path).resolve()),
                    per_prompt_csv_path=str(
                        Path(validation_report.per_prompt_csv_path).resolve()
                    ),
                )
            )

    best_run = max(run_records, key=lambda item: (item.average_f1, -item.run_index))
    summary_json_path = output_dir / "sweep_report.json"
    summary_csv_path = output_dir / "sweep_runs.csv"
    report = HyperparameterSweepReport(
        output_dir=str(output_dir.resolve()),
        summary_json_path=str(summary_json_path.resolve()),
        summary_csv_path=str(summary_csv_path.resolve()),
        total_runs=len(run_records),
        best_run_index=best_run.run_index,
        best_average_f1=best_run.average_f1,
        best_config={
            "guidance_scale": best_run.guidance_scale,
            "num_inference_steps": best_run.num_inference_steps,
            "sampler": best_run.sampler,
            "seed": best_run.seed,
        },
        runs=tuple(run_records),
    )
    write_hyperparameter_sweep_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON generation config.",
    )
    parser.add_argument(
        "--output-dir", help="Optional sweep output directory override."
    )
    parser.add_argument(
        "--guidance-scales",
        help="Optional comma-separated guidance scales override.",
    )
    parser.add_argument(
        "--steps",
        help="Optional comma-separated inference step counts override.",
    )
    parser.add_argument(
        "--samplers",
        help="Optional comma-separated sampler names override.",
    )
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated seeds override.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: HyperparameterSweepConfig, args: argparse.Namespace
) -> HyperparameterSweepConfig:
    """Apply explicit CLI overrides on top of JSON config values."""
    updates: dict[str, Any] = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.guidance_scales is not None:
        updates["guidance_scales"] = _parse_csv_arg(args.guidance_scales, cast=float)
    if args.steps is not None:
        updates["num_inference_steps"] = _parse_csv_arg(args.steps, cast=int)
    if args.samplers is not None:
        updates["samplers"] = _parse_csv_arg(args.samplers, cast=str)
    if args.seeds is not None:
        updates["seeds"] = _parse_csv_arg(args.seeds, cast=int)

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return HyperparameterSweepConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        sweep_config = load_hyperparameter_sweep_config(args.config)
        sweep_config = apply_cli_overrides(sweep_config, args)
        generation_config = load_generation_config(args.config)
        validation_config = load_offline_validation_config(args.config)
        yolo_config = load_yolo_detection_config(args.config)

        report = run_hyperparameter_sweep(
            sweep_config,
            generation_config=generation_config,
            validation_config=validation_config,
            yolo_config=yolo_config,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "total_runs": report.total_runs,
                    "best_run_index": report.best_run_index,
                    "best_average_f1": report.best_average_f1,
                    "best_config": report.best_config,
                    "summary_csv_path": report.summary_csv_path,
                    "summary_json_path": report.summary_json_path,
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
