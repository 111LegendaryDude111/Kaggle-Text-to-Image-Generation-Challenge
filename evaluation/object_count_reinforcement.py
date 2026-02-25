#!/usr/bin/env python3
"""Tune object-count reinforcement to improve recall while tracking false positives."""

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
    from evaluation.prompt_parser import (
        get_spacy_nlp,
        load_prompts_with_expected_objects,
    )
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
    from generation.prompt_optimizer import (
        build_count_reinforcement_clause,
        infer_object_counts,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.offline_runner import (
        OfflineValidationConfig,
        load_offline_validation_config,
        run_offline_validation,
    )
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import (
        get_spacy_nlp,
        load_prompts_with_expected_objects,
    )
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
    from generation.prompt_optimizer import (
        build_count_reinforcement_clause,
        infer_object_counts,
    )

_VARIANT_ALIASES: dict[str, str] = {
    "baseline": "baseline",
    "original": "baseline",
    "count_reinforced": "count_reinforced",
    "reinforced": "count_reinforced",
    "count": "count_reinforced",
    "exactly_n": "count_reinforced",
}
_SUPPORTED_VARIANTS: tuple[str, ...] = ("baseline", "count_reinforced")


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


class OfflinePromptResultLike(Protocol):
    """Shape required from per-prompt offline result objects."""

    prompt_id: str
    false_positives: int
    recall: float
    f1: float


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
    results: Sequence[OfflinePromptResultLike]


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


class ObjectCountReinforcementConfig(BaseModel):
    """Configuration for object-count reinforcement experiments."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "experiments/object_count_reinforcement"
    variants: list[str] = Field(
        default_factory=lambda: ["baseline", "count_reinforced"],
        validation_alias=AliasChoices("variants", "profiles"),
    )
    reinforcement_suffix: str = (
        "Ensure every required object is present and avoid missing requested objects."
    )
    fallback_count: int = Field(default=1, ge=1)
    use_spacy_count_inference: bool = True
    spacy_model: str = "en_core_web_sm"
    max_false_positive_examples: int = Field(default=10, ge=1, le=50)

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
class ObjectCountRunRecord:
    """Metrics for one count-reinforcement variant run."""

    run_index: int
    variant: str
    prompt_count: int
    average_precision: float
    average_recall: float
    average_f1: float
    total_false_positives: int
    average_false_positives_per_prompt: float
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
            "total_false_positives": self.total_false_positives,
            "average_false_positives_per_prompt": self.average_false_positives_per_prompt,
            "generation_runtime_sec": self.generation_runtime_sec,
            "evaluation_runtime_sec": self.evaluation_runtime_sec,
            "total_runtime_sec": self.total_runtime_sec,
            "generated_output_dir": self.generated_output_dir,
            "report_path": self.report_path,
            "per_prompt_csv_path": self.per_prompt_csv_path,
        }


@dataclass(frozen=True, slots=True)
class FalsePositiveDriftRecord:
    """Per-prompt false-positive drift from baseline to reinforced."""

    prompt_id: str
    baseline_false_positives: int
    reinforced_false_positives: int
    delta_false_positives: int
    baseline_recall: float
    reinforced_recall: float
    baseline_f1: float
    reinforced_f1: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize drift record to JSON-compatible shape."""
        return {
            "prompt_id": self.prompt_id,
            "baseline_false_positives": self.baseline_false_positives,
            "reinforced_false_positives": self.reinforced_false_positives,
            "delta_false_positives": self.delta_false_positives,
            "baseline_recall": self.baseline_recall,
            "reinforced_recall": self.reinforced_recall,
            "baseline_f1": self.baseline_f1,
            "reinforced_f1": self.reinforced_f1,
        }


@dataclass(frozen=True, slots=True)
class ObjectCountComparisons:
    """Key detection and false-positive deltas for count reinforcement."""

    baseline_variant: str | None
    reinforced_variant: str | None
    recall_delta: float | None
    f1_delta: float | None
    false_positive_delta: int | None
    average_false_positive_delta_per_prompt: float | None
    prompts_with_increased_false_positives: tuple[FalsePositiveDriftRecord, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize comparison summary to JSON-compatible shape."""
        return {
            "baseline_variant": self.baseline_variant,
            "reinforced_variant": self.reinforced_variant,
            "recall_delta": self.recall_delta,
            "f1_delta": self.f1_delta,
            "false_positive_delta": self.false_positive_delta,
            "average_false_positive_delta_per_prompt": self.average_false_positive_delta_per_prompt,
            "prompts_with_increased_false_positives": [
                item.to_json_dict()
                for item in self.prompts_with_increased_false_positives
            ],
        }


@dataclass(frozen=True, slots=True)
class ObjectCountReinforcementReport:
    """Aggregate count-reinforcement tuning report."""

    output_dir: str
    summary_json_path: str
    summary_csv_path: str
    total_variants: int
    best_variant: str
    best_average_f1: float
    comparisons: ObjectCountComparisons
    runs: tuple[ObjectCountRunRecord, ...]

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


def load_object_count_reinforcement_config(
    config_path: str | Path,
) -> ObjectCountReinforcementConfig:
    """Load object-count reinforcement config section from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("optimization"), dict):
        payload = payload["optimization"]
    if isinstance(payload.get("object_count_reinforcement"), dict):
        payload = payload["object_count_reinforcement"]

    return ObjectCountReinforcementConfig.model_validate(payload)


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


def _maybe_load_count_nlp(config: ObjectCountReinforcementConfig) -> Any:
    if not config.use_spacy_count_inference:
        return None
    try:
        return get_spacy_nlp(config.spacy_model)
    except Exception:
        return None


def infer_prompt_object_counts(
    prompt: PromptRecord,
    *,
    nlp: Any,
    fallback_count: int,
) -> dict[str, int]:
    """Infer object counts for one prompt with deterministic fallback."""
    expected_objects = list(prompt.expected_objects)
    if not expected_objects:
        return {}

    if nlp is None:
        return {label: fallback_count for label in expected_objects}

    try:
        inferred = infer_object_counts(prompt.text, nlp, expected_objects)
    except Exception:
        inferred = {}
    return {
        label: max(fallback_count, int(inferred.get(label, fallback_count)))
        for label in expected_objects
    }


def build_count_reinforced_prompt_text(
    prompt: PromptRecord,
    *,
    object_counts: dict[str, int],
    config: ObjectCountReinforcementConfig,
) -> str:
    """Inject explicit `exactly N` constraints into prompt text."""
    if not prompt.expected_objects:
        return prompt.text.strip()

    count_clause = build_count_reinforcement_clause(
        prompt.expected_objects, object_counts
    )
    suffix = config.reinforcement_suffix.strip()
    return " ".join(
        part for part in [prompt.text.strip(), count_clause, suffix] if part
    ).strip()


def build_variant_prompts(
    prompts: Sequence[PromptRecord],
    *,
    variant: str,
    config: ObjectCountReinforcementConfig,
    nlp: Any,
) -> list[PromptRecord]:
    """Build prompts for baseline or count-reinforced variant."""
    if variant == "baseline":
        return list(prompts)

    if variant != "count_reinforced":
        raise ValueError(f"Unsupported variant '{variant}'")

    rewritten: list[PromptRecord] = []
    for prompt in prompts:
        object_counts = infer_prompt_object_counts(
            prompt,
            nlp=nlp,
            fallback_count=config.fallback_count,
        )
        rewritten.append(
            PromptRecord(
                prompt_id=prompt.prompt_id,
                text=build_count_reinforced_prompt_text(
                    prompt,
                    object_counts=object_counts,
                    config=config,
                ),
                expected_objects=prompt.expected_objects,
            )
        )
    return rewritten


def _per_prompt_lookup(
    results: Sequence[OfflinePromptResultLike],
) -> dict[str, OfflinePromptResultLike]:
    return {item.prompt_id: item for item in results}


def build_object_count_comparisons(
    runs: Sequence[ObjectCountRunRecord],
    *,
    run_results_by_variant: dict[str, Sequence[OfflinePromptResultLike]],
    max_false_positive_examples: int,
) -> ObjectCountComparisons:
    """Build detection impact and false-positive drift summary."""
    baseline = next((run for run in runs if run.variant == "baseline"), None)
    reinforced = next((run for run in runs if run.variant == "count_reinforced"), None)
    if baseline is None or reinforced is None:
        return ObjectCountComparisons(
            baseline_variant=baseline.variant if baseline else None,
            reinforced_variant=reinforced.variant if reinforced else None,
            recall_delta=None,
            f1_delta=None,
            false_positive_delta=None,
            average_false_positive_delta_per_prompt=None,
            prompts_with_increased_false_positives=(),
        )

    baseline_lookup = _per_prompt_lookup(run_results_by_variant.get("baseline", []))
    reinforced_lookup = _per_prompt_lookup(
        run_results_by_variant.get("count_reinforced", [])
    )

    drifts: list[FalsePositiveDriftRecord] = []
    for prompt_id in sorted(
        set(baseline_lookup.keys()) | set(reinforced_lookup.keys())
    ):
        before = baseline_lookup.get(prompt_id)
        after = reinforced_lookup.get(prompt_id)
        baseline_fp = int(before.false_positives) if before is not None else 0
        reinforced_fp = int(after.false_positives) if after is not None else 0
        delta_fp = reinforced_fp - baseline_fp
        if delta_fp <= 0:
            continue

        drifts.append(
            FalsePositiveDriftRecord(
                prompt_id=prompt_id,
                baseline_false_positives=baseline_fp,
                reinforced_false_positives=reinforced_fp,
                delta_false_positives=delta_fp,
                baseline_recall=float(before.recall) if before is not None else 0.0,
                reinforced_recall=float(after.recall) if after is not None else 0.0,
                baseline_f1=float(before.f1) if before is not None else 0.0,
                reinforced_f1=float(after.f1) if after is not None else 0.0,
            )
        )

    drifts.sort(key=lambda item: (-item.delta_false_positives, item.prompt_id))
    top_drifts = tuple(drifts[:max_false_positive_examples])
    prompt_count = max(1, baseline.prompt_count)

    return ObjectCountComparisons(
        baseline_variant=baseline.variant,
        reinforced_variant=reinforced.variant,
        recall_delta=reinforced.average_recall - baseline.average_recall,
        f1_delta=reinforced.average_f1 - baseline.average_f1,
        false_positive_delta=reinforced.total_false_positives
        - baseline.total_false_positives,
        average_false_positive_delta_per_prompt=(
            (reinforced.total_false_positives - baseline.total_false_positives)
            / prompt_count
        ),
        prompts_with_increased_false_positives=top_drifts,
    )


def write_object_count_reinforcement_report(
    report: ObjectCountReinforcementReport,
) -> None:
    """Write object-count reinforcement report JSON + CSV summaries."""
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
                "total_false_positives",
                "average_false_positives_per_prompt",
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


def run_object_count_reinforcement(
    count_config: ObjectCountReinforcementConfig,
    *,
    generation_config: BaselineGenerationConfig,
    validation_config: OfflineValidationConfig,
    yolo_config: YoloDetectionConfig,
    prompts: Sequence[PromptRecord] | None = None,
    yolo_model: YoloModelLike | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
    evaluation_runner: OfflineValidationRunner = run_offline_validation,
) -> ObjectCountReinforcementReport:
    """Run count-reinforcement experiments and return aggregate report."""
    output_dir = Path(count_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = _resolve_prompt_records(
        validation_config=validation_config,
        generation_config=generation_config,
        prompts=prompts,
    )
    if not prompt_records:
        raise ValueError("No prompts available for object count reinforcement.")

    active_yolo_model = (
        yolo_model if yolo_model is not None else build_yolo_model(yolo_config)
    )
    device = resolve_device(generation_config.device)
    pipeline = pipeline_factory(generation_config, device)
    nlp = _maybe_load_count_nlp(count_config)

    run_records: list[ObjectCountRunRecord] = []
    run_results_by_variant: dict[str, Sequence[OfflinePromptResultLike]] = {}

    for run_index, variant in enumerate(count_config.variants, start=1):
        run_dir = output_dir / f"run_{run_index:02d}_{variant}"
        generated_output_dir = run_dir / "generated"
        report_path = run_dir / "report.json"
        per_prompt_csv_path = run_dir / "per_prompt_metrics.csv"

        runtime_prompts = build_variant_prompts(
            prompt_records,
            variant=variant,
            config=count_config,
            nlp=nlp,
        )
        runtime_generation_config = generation_config.model_copy(
            update={"output_dir": str(generated_output_dir)}
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
        run_results_by_variant[variant] = tuple(validation_report.results)
        total_false_positives = sum(
            int(item.false_positives) for item in validation_report.results
        )
        prompt_count = int(validation_report.prompt_count)
        run_records.append(
            ObjectCountRunRecord(
                run_index=run_index,
                variant=variant,
                prompt_count=prompt_count,
                average_precision=float(validation_report.average_precision),
                average_recall=float(validation_report.average_recall),
                average_f1=float(validation_report.average_f1),
                total_false_positives=total_false_positives,
                average_false_positives_per_prompt=(
                    total_false_positives / prompt_count if prompt_count > 0 else 0.0
                ),
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
    comparisons = build_object_count_comparisons(
        run_records,
        run_results_by_variant=run_results_by_variant,
        max_false_positive_examples=count_config.max_false_positive_examples,
    )
    summary_json_path = output_dir / "object_count_reinforcement_report.json"
    summary_csv_path = output_dir / "object_count_reinforcement_runs.csv"
    report = ObjectCountReinforcementReport(
        output_dir=str(output_dir.resolve()),
        summary_json_path=str(summary_json_path.resolve()),
        summary_csv_path=str(summary_csv_path.resolve()),
        total_variants=len(run_records),
        best_variant=best_run.variant,
        best_average_f1=best_run.average_f1,
        comparisons=comparisons,
        runs=tuple(run_records),
    )
    write_object_count_reinforcement_report(report)
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
        help="Optional object-count reinforcement output directory override.",
    )
    parser.add_argument(
        "--variants",
        help="Optional comma-separated variant override.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: ObjectCountReinforcementConfig,
    args: argparse.Namespace,
) -> ObjectCountReinforcementConfig:
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
    return ObjectCountReinforcementConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        count_config = load_object_count_reinforcement_config(args.config)
        count_config = apply_cli_overrides(count_config, args)
        generation_config = load_generation_config(args.config)
        validation_config = load_offline_validation_config(args.config)
        yolo_config = load_yolo_detection_config(args.config)

        report = run_object_count_reinforcement(
            count_config,
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
