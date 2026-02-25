#!/usr/bin/env python3
"""Tune realism bias prompts to improve YOLO detectability."""

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
        load_offline_validation_config,
        run_offline_validation,
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
        resolve_device,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.offline_runner import (
        OfflineValidationConfig,
        load_offline_validation_config,
        run_offline_validation,
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
        resolve_device,
    )

_VARIANT_ALIASES: dict[str, str] = {
    "baseline": "baseline",
    "original": "baseline",
    "photorealistic": "photorealistic",
    "realistic": "photorealistic",
    "photo_realistic": "photorealistic",
    "photorealistic_simple_background": "photorealistic_simple_background",
    "realistic_simple_background": "photorealistic_simple_background",
    "simple_background": "photorealistic_simple_background",
    "background_simplified": "photorealistic_simple_background",
    "stylized": "stylized",
    "style": "stylized",
}
_SUPPORTED_VARIANTS: tuple[str, ...] = (
    "baseline",
    "photorealistic",
    "photorealistic_simple_background",
    "stylized",
)
_REALISTIC_VARIANTS: frozenset[str] = frozenset(
    {"baseline", "photorealistic", "photorealistic_simple_background"}
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


def _parse_csv_arg(raw: str) -> list[str]:
    values: list[str] = []
    for item in raw.split(","):
        normalized = item.strip()
        if normalized:
            values.append(normalized)
    if not values:
        raise ValueError("CSV override produced an empty list.")
    return values


def normalize_variant_name(variant: str) -> str:
    """Normalize variant aliases to canonical variant names."""
    normalized = variant.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        raise ValueError("variant cannot be empty")
    return _VARIANT_ALIASES.get(normalized, normalized)


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
        """Create generation pipeline."""


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
        """Run offline validation for one variant."""


class RealismBiasTuningConfig(BaseModel):
    """Configuration for realism-bias prompt variant tuning."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "experiments/realism_bias_tuning"
    variants: list[str] = Field(
        default_factory=lambda: [
            "baseline",
            "photorealistic",
            "photorealistic_simple_background",
            "stylized",
        ],
        validation_alias=AliasChoices("variants", "styles", "profiles"),
    )

    photorealistic_prefix: str = (
        "A photorealistic, high-resolution photograph with natural lighting and true-to-life textures."
    )
    photorealistic_directive: str = (
        "Ensure clear object boundaries, realistic scale, and high YOLO-detectable object visibility."
    )
    background_simplification_directive: str = (
        "Use a simple uncluttered background with minimal extra props and low visual noise."
    )
    stylized_prefix: str = (
        "A stylized artistic illustration with painterly rendering and expressive visual style."
    )
    stylized_directive: str = (
        "Prioritize stylized aesthetics over strict photorealistic appearance."
    )

    @field_validator("variants", mode="before")
    @classmethod
    def _coerce_variants_to_list(cls, value: Any) -> list[Any]:
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @field_validator("variants")
    @classmethod
    def _validate_variants(cls, value: list[str]) -> list[str]:
        normalized = [normalize_variant_name(str(item)) for item in value]
        deduped = _dedupe_preserve_order(normalized)
        if not deduped:
            raise ValueError("variants cannot be empty")
        unsupported = [item for item in deduped if item not in _SUPPORTED_VARIANTS]
        if unsupported:
            supported = ", ".join(_SUPPORTED_VARIANTS)
            unknown = ", ".join(unsupported)
            raise ValueError(f"Unsupported variants: {unknown}. Supported: {supported}")
        return deduped


@dataclass(frozen=True, slots=True)
class RealismBiasRunRecord:
    """Metrics for one realism-bias variant run."""

    run_index: int
    variant: str
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
        """Serialize run record to JSON-compatible shape."""
        return {
            "run_index": self.run_index,
            "variant": self.variant,
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
class RealismBiasComparisons:
    """Key F1 deltas requested for realism-bias analysis."""

    photorealistic_vs_baseline_delta_f1: float | None
    simple_background_vs_photorealistic_delta_f1: float | None
    stylized_vs_best_realistic_delta_f1: float | None
    best_realistic_variant: str | None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize comparison deltas to JSON-compatible shape."""
        return {
            "photorealistic_vs_baseline_delta_f1": self.photorealistic_vs_baseline_delta_f1,
            "simple_background_vs_photorealistic_delta_f1": self.simple_background_vs_photorealistic_delta_f1,
            "stylized_vs_best_realistic_delta_f1": self.stylized_vs_best_realistic_delta_f1,
            "best_realistic_variant": self.best_realistic_variant,
        }


@dataclass(frozen=True, slots=True)
class RealismBiasTuningReport:
    """Aggregate realism-bias tuning report."""

    output_dir: str
    summary_json_path: str
    summary_csv_path: str
    total_variants: int
    best_variant: str
    best_average_f1: float
    comparisons: RealismBiasComparisons
    runs: tuple[RealismBiasRunRecord, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize full report to JSON-compatible shape."""
        return {
            "output_dir": self.output_dir,
            "summary_json_path": self.summary_json_path,
            "summary_csv_path": self.summary_csv_path,
            "total_variants": self.total_variants,
            "best_variant": self.best_variant,
            "best_average_f1": self.best_average_f1,
            "comparisons": self.comparisons.to_json_dict(),
            "runs": [item.to_json_dict() for item in self.runs],
        }


def load_realism_bias_tuning_config(
    config_path: str | Path,
) -> RealismBiasTuningConfig:
    """Load realism-bias tuning config section from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("optimization"), dict):
        payload = payload["optimization"]
    if isinstance(payload.get("realism_bias_tuning"), dict):
        payload = payload["realism_bias_tuning"]

    return RealismBiasTuningConfig.model_validate(payload)


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


def build_variant_prompt_text(
    original_text: str,
    *,
    variant: str,
    config: RealismBiasTuningConfig,
) -> str:
    """Build variant prompt text for one realism profile."""
    base_text = original_text.strip()
    if variant == "baseline":
        return base_text

    if variant == "photorealistic":
        return " ".join(
            [
                config.photorealistic_prefix.strip(),
                f"Scene description: {base_text}",
                config.photorealistic_directive.strip(),
            ]
        ).strip()

    if variant == "photorealistic_simple_background":
        return " ".join(
            [
                config.photorealistic_prefix.strip(),
                f"Scene description: {base_text}",
                config.photorealistic_directive.strip(),
                config.background_simplification_directive.strip(),
            ]
        ).strip()

    if variant == "stylized":
        return " ".join(
            [
                config.stylized_prefix.strip(),
                f"Scene description: {base_text}",
                config.stylized_directive.strip(),
            ]
        ).strip()

    raise ValueError(f"Unsupported variant '{variant}'")


def build_variant_prompts(
    prompts: Sequence[PromptRecord],
    *,
    variant: str,
    config: RealismBiasTuningConfig,
) -> list[PromptRecord]:
    """Apply one realism-bias variant to all prompts."""
    return [
        PromptRecord(
            prompt_id=prompt.prompt_id,
            text=build_variant_prompt_text(
                prompt.text,
                variant=variant,
                config=config,
            ),
            expected_objects=prompt.expected_objects,
        )
        for prompt in prompts
    ]


def _find_run_by_variant(
    runs: Sequence[RealismBiasRunRecord],
    variant: str,
) -> RealismBiasRunRecord | None:
    for run in runs:
        if run.variant == variant:
            return run
    return None


def build_realism_bias_comparisons(
    runs: Sequence[RealismBiasRunRecord],
) -> RealismBiasComparisons:
    """Build requested realism-vs-style F1 comparisons."""
    baseline = _find_run_by_variant(runs, "baseline")
    photorealistic = _find_run_by_variant(runs, "photorealistic")
    simplified = _find_run_by_variant(runs, "photorealistic_simple_background")
    stylized = _find_run_by_variant(runs, "stylized")

    photorealistic_vs_baseline = None
    if baseline is not None and photorealistic is not None:
        photorealistic_vs_baseline = photorealistic.average_f1 - baseline.average_f1

    simple_background_vs_photorealistic = None
    if simplified is not None and photorealistic is not None:
        simple_background_vs_photorealistic = (
            simplified.average_f1 - photorealistic.average_f1
        )

    realistic_runs = [run for run in runs if run.variant in _REALISTIC_VARIANTS]
    best_realistic = (
        max(realistic_runs, key=lambda item: (item.average_f1, -item.run_index))
        if realistic_runs
        else None
    )

    stylized_vs_best_realistic = None
    if stylized is not None and best_realistic is not None:
        stylized_vs_best_realistic = stylized.average_f1 - best_realistic.average_f1

    return RealismBiasComparisons(
        photorealistic_vs_baseline_delta_f1=photorealistic_vs_baseline,
        simple_background_vs_photorealistic_delta_f1=simple_background_vs_photorealistic,
        stylized_vs_best_realistic_delta_f1=stylized_vs_best_realistic,
        best_realistic_variant=best_realistic.variant if best_realistic else None,
    )


def write_realism_bias_tuning_report(report: RealismBiasTuningReport) -> None:
    """Write realism-bias report JSON + CSV summaries."""
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
                "variant",
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


def run_realism_bias_tuning(
    tuning_config: RealismBiasTuningConfig,
    *,
    generation_config: BaselineGenerationConfig,
    validation_config: OfflineValidationConfig,
    yolo_config: YoloDetectionConfig,
    prompts: Sequence[PromptRecord] | None = None,
    yolo_model: YoloModelLike | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
    evaluation_runner: OfflineValidationRunner = run_offline_validation,
) -> RealismBiasTuningReport:
    """Run realism-bias prompt tuning and return aggregate report."""
    output_dir = Path(tuning_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = _resolve_prompt_records(
        validation_config=validation_config,
        generation_config=generation_config,
        prompts=prompts,
    )
    if not prompt_records:
        raise ValueError("No prompts available for realism bias tuning.")

    active_yolo_model = (
        yolo_model if yolo_model is not None else build_yolo_model(yolo_config)
    )
    device = resolve_device(generation_config.device)
    pipeline = pipeline_factory(generation_config, device)

    run_records: list[RealismBiasRunRecord] = []
    for run_index, variant in enumerate(tuning_config.variants, start=1):
        run_dir = output_dir / f"run_{run_index:02d}_{variant}"
        generated_output_dir = run_dir / "generated"
        report_path = run_dir / "report.json"
        per_prompt_csv_path = run_dir / "per_prompt_metrics.csv"

        runtime_prompts = build_variant_prompts(
            prompt_records,
            variant=variant,
            config=tuning_config,
        )
        runtime_generation_config = generation_config.model_copy(
            update={
                "output_dir": str(generated_output_dir),
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
            prompts=runtime_prompts,
            generation_pipeline=pipeline,
            yolo_model=active_yolo_model,
        )
        run_records.append(
            RealismBiasRunRecord(
                run_index=run_index,
                variant=variant,
                prompt_count=int(validation_report.prompt_count),
                average_precision=float(validation_report.average_precision),
                average_recall=float(validation_report.average_recall),
                average_f1=float(validation_report.average_f1),
                generation_runtime_sec=float(validation_report.generation_runtime_sec),
                evaluation_runtime_sec=float(validation_report.evaluation_runtime_sec),
                total_runtime_sec=float(validation_report.total_runtime_sec),
                generated_output_dir=str(generated_output_dir.resolve()),
                report_path=str(Path(validation_report.report_path).resolve()),
                per_prompt_csv_path=str(
                    Path(validation_report.per_prompt_csv_path).resolve()
                ),
            )
        )

    best_run = max(run_records, key=lambda item: (item.average_f1, -item.run_index))
    comparisons = build_realism_bias_comparisons(run_records)
    summary_json_path = output_dir / "realism_bias_report.json"
    summary_csv_path = output_dir / "realism_bias_runs.csv"
    report = RealismBiasTuningReport(
        output_dir=str(output_dir.resolve()),
        summary_json_path=str(summary_json_path.resolve()),
        summary_csv_path=str(summary_csv_path.resolve()),
        total_variants=len(run_records),
        best_variant=best_run.variant,
        best_average_f1=best_run.average_f1,
        comparisons=comparisons,
        runs=tuple(run_records),
    )
    write_realism_bias_tuning_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON generation config.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional realism-bias tuning output directory override.",
    )
    parser.add_argument(
        "--variants",
        help="Optional comma-separated variant override.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: RealismBiasTuningConfig,
    args: argparse.Namespace,
) -> RealismBiasTuningConfig:
    """Apply explicit CLI overrides on top of JSON config values."""
    updates: dict[str, Any] = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.variants is not None:
        updates["variants"] = _parse_csv_arg(args.variants)

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return RealismBiasTuningConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        tuning_config = load_realism_bias_tuning_config(args.config)
        tuning_config = apply_cli_overrides(tuning_config, args)
        generation_config = load_generation_config(args.config)
        validation_config = load_offline_validation_config(args.config)
        yolo_config = load_yolo_detection_config(args.config)

        report = run_realism_bias_tuning(
            tuning_config,
            generation_config=generation_config,
            validation_config=validation_config,
            yolo_config=yolo_config,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "total_variants": report.total_variants,
                    "best_variant": report.best_variant,
                    "best_average_f1": report.best_average_f1,
                    "comparisons": report.comparisons.to_json_dict(),
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
