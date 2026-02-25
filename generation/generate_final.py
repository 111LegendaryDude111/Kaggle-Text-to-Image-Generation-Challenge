#!/usr/bin/env python3
"""Freeze best config and regenerate final one-image-per-prompt outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

try:
    from evaluation.object_count_reinforcement import (
        ObjectCountReinforcementConfig,
        build_count_reinforced_prompt_text,
        infer_prompt_object_counts,
        load_object_count_reinforcement_config,
        normalize_variant_name as normalize_count_variant_name,
    )
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import (
        get_spacy_nlp,
        load_prompts_with_expected_objects,
    )
    from evaluation.realism_bias_tuning import (
        RealismBiasTuningConfig,
        build_variant_prompt_text,
        load_realism_bias_tuning_config,
        normalize_variant_name as normalize_realism_variant_name,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        GeneratedImageRecord,
        TextToImagePipeline,
        load_generation_config,
        run_baseline_generation,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.object_count_reinforcement import (
        ObjectCountReinforcementConfig,
        build_count_reinforced_prompt_text,
        infer_prompt_object_counts,
        load_object_count_reinforcement_config,
        normalize_variant_name as normalize_count_variant_name,
    )
    from evaluation.prompt_loader import PromptRecord
    from evaluation.prompt_parser import (
        get_spacy_nlp,
        load_prompts_with_expected_objects,
    )
    from evaluation.realism_bias_tuning import (
        RealismBiasTuningConfig,
        build_variant_prompt_text,
        load_realism_bias_tuning_config,
        normalize_variant_name as normalize_realism_variant_name,
    )
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        GeneratedImageRecord,
        TextToImagePipeline,
        load_generation_config,
        run_baseline_generation,
    )


class FinalImageGenerationConfig(BaseModel):
    """Configuration for freezing and regenerating final submission images."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "dreamlayer_export/final_images"
    frozen_config_path: str = "dreamlayer_export/frozen_generation_config.json"
    report_path: str = "dreamlayer_export/final_generation_report.json"
    clean_output_dir: bool = True

    prompt_file: str | None = None
    prompt_spacy_model: str = "en_core_web_sm"
    keep_unmapped_expected_objects: bool = False

    use_best_hyperparameters: bool = True
    hyperparameter_report_path: str = (
        "experiments/hyperparameter_sweep/sweep_report.json"
    )

    use_best_realism_variant: bool = True
    realism_report_path: str = (
        "experiments/realism_bias_tuning/realism_bias_report.json"
    )
    default_realism_variant: str = "baseline"

    use_best_object_count_variant: bool = True
    object_count_report_path: str = (
        "experiments/object_count_reinforcement/object_count_reinforcement_report.json"
    )
    default_object_count_variant: str = "baseline"

    require_reports: bool = False


@dataclass(frozen=True, slots=True)
class FrozenFinalConfig:
    """Frozen generation config selected for final regeneration."""

    generation_config: dict[str, Any]
    realism_variant: str
    object_count_variant: str
    source_reports: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize frozen config to JSON-compatible shape."""
        return {
            "generation_config": self.generation_config,
            "realism_variant": self.realism_variant,
            "object_count_variant": self.object_count_variant,
            "source_reports": self.source_reports,
        }


@dataclass(frozen=True, slots=True)
class FinalGenerationReport:
    """Final generation summary for DreamLayer export preparation."""

    prompt_count: int
    generated_count: int
    output_dir: str
    frozen_config_path: str
    report_path: str
    realism_variant: str
    object_count_variant: str
    generated_files: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize report to JSON-compatible shape."""
        return {
            "prompt_count": self.prompt_count,
            "generated_count": self.generated_count,
            "output_dir": self.output_dir,
            "frozen_config_path": self.frozen_config_path,
            "report_path": self.report_path,
            "realism_variant": self.realism_variant,
            "object_count_variant": self.object_count_variant,
            "generated_files": list(self.generated_files),
        }


def load_final_generation_config(config_path: str | Path) -> FinalImageGenerationConfig:
    """Load final generation config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("export"), dict):
        payload = payload["export"]
    if isinstance(payload.get("final_image_generation"), dict):
        payload = payload["final_image_generation"]
    else:
        payload = {}

    return FinalImageGenerationConfig.model_validate(payload)


def _load_optional_report(
    report_path: str | Path,
    *,
    require_reports: bool,
) -> dict[str, Any] | None:
    path = Path(report_path)
    if not path.exists():
        if require_reports:
            raise FileNotFoundError(f"Required report not found: {path}")
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        if require_reports:
            raise ValueError(f"Report must contain a JSON object: {path}")
        return None
    return payload


def resolve_frozen_final_config(
    *,
    base_generation_config: BaselineGenerationConfig,
    final_config: FinalImageGenerationConfig,
) -> tuple[BaselineGenerationConfig, FrozenFinalConfig]:
    """Resolve best hyperparameters and prompt variants into a frozen config."""
    runtime_generation_config = base_generation_config
    source_reports: dict[str, Any] = {}

    if final_config.use_best_hyperparameters:
        sweep_report = _load_optional_report(
            final_config.hyperparameter_report_path,
            require_reports=final_config.require_reports,
        )
        if sweep_report is not None and isinstance(
            sweep_report.get("best_config"), dict
        ):
            best_config = sweep_report["best_config"]
            updates: dict[str, Any] = {}
            for key in ("guidance_scale", "num_inference_steps", "sampler", "seed"):
                if key in best_config:
                    updates[key] = best_config[key]
            if updates:
                runtime_generation_config = runtime_generation_config.model_copy(
                    update=updates
                )
            source_reports["hyperparameter_report_path"] = str(
                Path(final_config.hyperparameter_report_path).resolve()
            )
            source_reports["hyperparameter_best_config"] = dict(best_config)

    realism_variant = normalize_realism_variant_name(
        final_config.default_realism_variant
    )
    if final_config.use_best_realism_variant:
        realism_report = _load_optional_report(
            final_config.realism_report_path,
            require_reports=final_config.require_reports,
        )
        if realism_report is not None and isinstance(
            realism_report.get("best_variant"), str
        ):
            realism_variant = normalize_realism_variant_name(
                realism_report["best_variant"]
            )
            source_reports["realism_report_path"] = str(
                Path(final_config.realism_report_path).resolve()
            )
            source_reports["realism_best_variant"] = realism_variant

    object_count_variant = normalize_count_variant_name(
        final_config.default_object_count_variant
    )
    if final_config.use_best_object_count_variant:
        object_count_report = _load_optional_report(
            final_config.object_count_report_path,
            require_reports=final_config.require_reports,
        )
        if object_count_report is not None and isinstance(
            object_count_report.get("best_variant"), str
        ):
            object_count_variant = normalize_count_variant_name(
                object_count_report["best_variant"]
            )
            source_reports["object_count_report_path"] = str(
                Path(final_config.object_count_report_path).resolve()
            )
            source_reports["object_count_best_variant"] = object_count_variant

    frozen = FrozenFinalConfig(
        generation_config=runtime_generation_config.model_dump(),
        realism_variant=realism_variant,
        object_count_variant=object_count_variant,
        source_reports=source_reports,
    )
    return runtime_generation_config, frozen


def write_frozen_final_config(
    frozen: FrozenFinalConfig,
    *,
    output_path: str | Path,
) -> str:
    """Write frozen final config JSON and return resolved path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(frozen.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return str(destination.resolve())


def _maybe_clean_output_dir(output_dir: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in output_dir.glob("*.png"):
        file_path.unlink()


def build_final_prompts(
    prompts: Sequence[PromptRecord],
    *,
    realism_variant: str,
    object_count_variant: str,
    realism_tuning_config: RealismBiasTuningConfig,
    count_tuning_config: ObjectCountReinforcementConfig,
) -> list[PromptRecord]:
    """Apply selected count and realism transformations to prompts."""
    transformed_prompts: list[PromptRecord] = []
    nlp = None
    if (
        object_count_variant == "count_reinforced"
        and count_tuning_config.use_spacy_count_inference
    ):
        try:
            nlp = get_spacy_nlp(count_tuning_config.spacy_model)
        except Exception:
            nlp = None

    for prompt in prompts:
        text = prompt.text.strip()

        if object_count_variant == "count_reinforced":
            object_counts = infer_prompt_object_counts(
                prompt,
                nlp=nlp,
                fallback_count=count_tuning_config.fallback_count,
            )
            text = build_count_reinforced_prompt_text(
                PromptRecord(
                    prompt_id=prompt.prompt_id,
                    text=text,
                    expected_objects=prompt.expected_objects,
                ),
                object_counts=object_counts,
                config=count_tuning_config,
            )

        if realism_variant != "baseline":
            text = build_variant_prompt_text(
                text,
                variant=realism_variant,
                config=realism_tuning_config,
            )

        transformed_prompts.append(
            PromptRecord(
                prompt_id=prompt.prompt_id,
                text=text,
                expected_objects=prompt.expected_objects,
            )
        )

    return transformed_prompts


def _validate_final_output(
    prompts: Sequence[PromptRecord],
    generated: Sequence[GeneratedImageRecord],
    *,
    output_dir: str | Path,
) -> tuple[str, ...]:
    expected_names = tuple(f"{prompt.prompt_id}.png" for prompt in prompts)
    expected_set = set(expected_names)

    generated_names = tuple(Path(item.output_path).name for item in generated)
    generated_set = set(generated_names)
    if expected_set != generated_set:
        missing = sorted(expected_set - generated_set)
        extra = sorted(generated_set - expected_set)
        raise RuntimeError(
            "Final generation output mismatch. " f"Missing: {missing}. Extra: {extra}."
        )

    output_png_names = {path.name for path in Path(output_dir).glob("*.png")}
    if output_png_names != expected_set:
        missing = sorted(expected_set - output_png_names)
        extra = sorted(output_png_names - expected_set)
        raise RuntimeError(
            "Output directory does not contain exactly one image per prompt. "
            f"Missing: {missing}. Extra: {extra}."
        )

    return tuple(sorted(expected_names))


def run_final_image_generation(
    final_config: FinalImageGenerationConfig,
    *,
    generation_config: BaselineGenerationConfig,
    prompts: Sequence[PromptRecord] | None = None,
    pipeline: TextToImagePipeline | None = None,
    realism_tuning_config: RealismBiasTuningConfig | None = None,
    count_tuning_config: ObjectCountReinforcementConfig | None = None,
) -> FinalGenerationReport:
    """Freeze config, regenerate clean output, and validate final image set."""
    runtime_generation_config, frozen = resolve_frozen_final_config(
        base_generation_config=generation_config,
        final_config=final_config,
    )
    frozen_config_path = write_frozen_final_config(
        frozen,
        output_path=final_config.frozen_config_path,
    )

    runtime_prompt_file = (
        final_config.prompt_file or runtime_generation_config.prompt_file
    )
    prompt_records = (
        list(prompts)
        if prompts is not None
        else load_prompts_with_expected_objects(
            runtime_prompt_file,
            spacy_model=final_config.prompt_spacy_model,
            keep_unmapped=final_config.keep_unmapped_expected_objects,
        )
    )
    if not prompt_records:
        raise ValueError("No prompts available for final image generation.")

    realism_cfg = (
        realism_tuning_config
        if realism_tuning_config is not None
        else RealismBiasTuningConfig()
    )
    count_cfg = (
        count_tuning_config
        if count_tuning_config is not None
        else ObjectCountReinforcementConfig()
    )
    transformed_prompts = build_final_prompts(
        prompt_records,
        realism_variant=frozen.realism_variant,
        object_count_variant=frozen.object_count_variant,
        realism_tuning_config=realism_cfg,
        count_tuning_config=count_cfg,
    )

    output_dir = Path(final_config.output_dir)
    _maybe_clean_output_dir(output_dir, enabled=final_config.clean_output_dir)

    runtime_generation_config = runtime_generation_config.model_copy(
        update={
            "output_dir": str(output_dir),
            "prompt_file": runtime_prompt_file,
        }
    )
    generated = run_baseline_generation(
        runtime_generation_config,
        prompts=transformed_prompts,
        pipeline=pipeline,
    )
    generated_files = _validate_final_output(
        transformed_prompts,
        generated,
        output_dir=output_dir,
    )

    report = FinalGenerationReport(
        prompt_count=len(transformed_prompts),
        generated_count=len(generated),
        output_dir=str(output_dir.resolve()),
        frozen_config_path=frozen_config_path,
        report_path=str(Path(final_config.report_path).resolve()),
        realism_variant=frozen.realism_variant,
        object_count_variant=frozen.object_count_variant,
        generated_files=generated_files,
    )
    report_path = Path(final_config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON generation config.",
    )
    parser.add_argument(
        "--output-dir", help="Optional final output directory override."
    )
    parser.add_argument(
        "--frozen-config-path",
        help="Optional path for frozen config JSON override.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional path for final generation report JSON override.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional prompt file override.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: FinalImageGenerationConfig,
    args: argparse.Namespace,
) -> FinalImageGenerationConfig:
    """Apply explicit CLI overrides on top of JSON config values."""
    updates: dict[str, Any] = {}
    for field_name in (
        "output_dir",
        "frozen_config_path",
        "report_path",
        "prompt_file",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return FinalImageGenerationConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        final_config = load_final_generation_config(args.config)
        final_config = apply_cli_overrides(final_config, args)
        generation_config = load_generation_config(args.config)
        realism_tuning_config = load_realism_bias_tuning_config(args.config)
        count_tuning_config = load_object_count_reinforcement_config(args.config)

        report = run_final_image_generation(
            final_config,
            generation_config=generation_config,
            realism_tuning_config=realism_tuning_config,
            count_tuning_config=count_tuning_config,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "prompt_count": report.prompt_count,
                    "generated_count": report.generated_count,
                    "output_dir": report.output_dir,
                    "frozen_config_path": report.frozen_config_path,
                    "report_path": report.report_path,
                    "realism_variant": report.realism_variant,
                    "object_count_variant": report.object_count_variant,
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
