#!/usr/bin/env python3
"""Generate multiple variants per prompt, evaluate, and keep best image."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import Field

try:
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        build_pipeline,
        compute_prompt_seed,
        create_torch_generator,
        resolve_device,
    )
    from generation.negative_prompt_strategy import (
        build_negative_prompt_for_prompt,
        maybe_load_nlp_for_negative_prompt_strategy,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.generate_baseline import (
        BaselineGenerationConfig,
        build_pipeline,
        compute_prompt_seed,
        create_torch_generator,
        resolve_device,
    )
    from generation.negative_prompt_strategy import (
        build_negative_prompt_for_prompt,
        maybe_load_nlp_for_negative_prompt_strategy,
    )


class PipelineOutput(Protocol):
    """Minimal output contract expected from a text-to-image pipeline."""

    images: Sequence[Any]


class TextToImagePipeline(Protocol):
    """Minimal callable contract for text-to-image pipelines."""

    def to(self, device: str) -> "TextToImagePipeline":
        """Move pipeline to target device."""

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        num_inference_steps: int,
        width: int,
        height: int,
        num_images_per_prompt: int,
        generator: Any,
    ) -> PipelineOutput:
        """Generate image(s) for a prompt."""


class VariantEvaluator(Protocol):
    """Evaluator contract for scoring candidate variants."""

    def __call__(
        self,
        *,
        prompt: PromptRecord,
        image_path: Path,
        variant_index: int,
        seed: int,
        guidance_scale: float,
        metadata: dict[str, Any],
    ) -> Any:
        """Return float score or dict with `score` field."""


class MultiSampleGenerationConfig(BaselineGenerationConfig):
    """Config for multisample generation and variant search."""

    num_variants: int = Field(default=4, ge=4, le=8)
    guidance_scale_step: float = 0.3
    variant_seed_step: int = Field(default=1, ge=1)

    final_output_dir: str = "generation/outputs/multisample/final"
    temp_output_dir: str = "generation/outputs/multisample/temp"
    metadata_path: str = "generation/outputs/multisample/metadata.json"
    keep_temp_images: bool = True

    evaluator_spec: str | None = None


@dataclass(frozen=True, slots=True)
class VariantScoreRecord:
    """Metadata and evaluation result for a generated candidate variant."""

    prompt_id: str
    prompt_text: str
    variant_index: int
    seed: int
    guidance_scale: float
    negative_prompt: str
    temp_image_path: str
    score: float
    evaluator_details: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize variant record to JSON-friendly dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "variant_index": self.variant_index,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "temp_image_path": self.temp_image_path,
            "score": self.score,
            "evaluator_details": self.evaluator_details,
        }


@dataclass(frozen=True, slots=True)
class PromptSelectionRecord:
    """Best-variant selection metadata for one prompt."""

    prompt_id: str
    best_variant_index: int
    best_score: float
    best_temp_image_path: str
    final_image_path: str

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize selection record to JSON-friendly dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "best_variant_index": self.best_variant_index,
            "best_score": self.best_score,
            "best_temp_image_path": self.best_temp_image_path,
            "final_image_path": self.final_image_path,
        }


@dataclass(frozen=True, slots=True)
class MultiSampleRunReport:
    """Run summary for multisample generation."""

    prompt_count: int
    num_variants: int
    metadata_path: str
    selections: tuple[PromptSelectionRecord, ...]
    variants: tuple[VariantScoreRecord, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize run report to JSON-friendly dictionary."""
        return {
            "prompt_count": self.prompt_count,
            "num_variants": self.num_variants,
            "metadata_path": self.metadata_path,
            "selections": [record.to_json_dict() for record in self.selections],
            "variants": [record.to_json_dict() for record in self.variants],
        }


def load_multisample_config(config_path: str | Path) -> MultiSampleGenerationConfig:
    """Load multisample config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Generation config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation config JSON must be an object")

    if isinstance(payload.get("multisample"), dict):
        payload = payload["multisample"]
    elif isinstance(payload.get("baseline"), dict):
        payload = payload["baseline"]

    return MultiSampleGenerationConfig.model_validate(payload)


def default_zero_evaluator(**_: Any) -> dict[str, Any]:
    """Deterministic fallback evaluator when no external evaluator is configured."""
    return {"score": 0.0, "reason": "no evaluator configured"}


def _parse_evaluator_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("evaluator spec must be in `module:function` format")
    module_name, function_name = spec.split(":", 1)
    if not module_name.strip() or not function_name.strip():
        raise ValueError("invalid evaluator spec; empty module/function name")
    return module_name.strip(), function_name.strip()


def load_variant_evaluator(spec: str | None) -> VariantEvaluator:
    """Load evaluator callable from spec or return default fallback evaluator."""
    if spec is None or not spec.strip():
        return default_zero_evaluator

    module_name, function_name = _parse_evaluator_spec(spec)
    module = importlib.import_module(module_name)
    candidate = getattr(module, function_name, None)
    if not callable(candidate):
        raise TypeError(f"Evaluator `{spec}` is not callable")
    return candidate  # type: ignore[return-value]


def normalize_evaluator_output(output: Any) -> tuple[float, dict[str, Any]]:
    """Normalize evaluator output to `(score, details)`."""
    if isinstance(output, (float, int)):
        return float(output), {}

    if isinstance(output, dict):
        if "score" not in output:
            raise ValueError("Evaluator dict output must include a `score` field.")
        score = float(output["score"])
        details = {str(key): value for key, value in output.items() if key != "score"}
        return score, details

    if hasattr(output, "score"):
        score = float(getattr(output, "score"))
        details = (
            dict(getattr(output, "__dict__", {}))
            if hasattr(output, "__dict__")
            else {}
        )
        details.pop("score", None)
        return score, details

    raise TypeError(
        "Evaluator output must be float/int, dict with `score`, or object with `.score`."
    )


def compute_guidance_schedule(
    *,
    base_guidance_scale: float,
    guidance_scale_step: float,
    num_variants: int,
) -> list[float]:
    """Build symmetric guidance schedule centered around base guidance."""
    center = (num_variants - 1) / 2
    schedule: list[float] = []
    for variant_index in range(num_variants):
        offset = (variant_index - center) * guidance_scale_step
        schedule.append(max(0.0, base_guidance_scale + offset))
    return schedule


def _resolve_negative_prompt(
    prompt: PromptRecord,
    config: MultiSampleGenerationConfig,
    strategy_nlp: Any,
) -> str:
    if not config.use_structured_negative_prompt:
        return config.negative_prompt.strip()
    return build_negative_prompt_for_prompt(
        prompt,
        base_negative_prompt=config.negative_prompt,
        config=config.negative_prompt_strategy,
        nlp=strategy_nlp,
    ).strip()


def generate_variants_for_prompt(
    *,
    prompt: PromptRecord,
    prompt_index: int,
    pipeline: TextToImagePipeline,
    config: MultiSampleGenerationConfig,
    device: str,
    evaluator: VariantEvaluator,
    temp_output_dir: Path,
    strategy_nlp: Any,
) -> tuple[list[VariantScoreRecord], PromptSelectionRecord]:
    """Generate, evaluate, and select best variant for one prompt."""
    prompt_base_seed = compute_prompt_seed(config.seed, prompt_index, config.seed_strategy)
    negative_prompt = _resolve_negative_prompt(prompt, config, strategy_nlp)
    guidance_schedule = compute_guidance_schedule(
        base_guidance_scale=config.guidance_scale,
        guidance_scale_step=config.guidance_scale_step,
        num_variants=config.num_variants,
    )

    prompt_temp_dir = temp_output_dir / prompt.prompt_id
    prompt_temp_dir.mkdir(parents=True, exist_ok=True)

    variants: list[VariantScoreRecord] = []
    for variant_index, variant_guidance in enumerate(guidance_schedule):
        variant_seed = prompt_base_seed + (variant_index * config.variant_seed_step)
        output = pipeline(
            prompt=prompt.text,
            negative_prompt=negative_prompt or None,
            guidance_scale=variant_guidance,
            num_inference_steps=config.num_inference_steps,
            width=config.width,
            height=config.height,
            num_images_per_prompt=1,
            generator=create_torch_generator(device, variant_seed),
        )
        if not output.images:
            raise RuntimeError(
                f"Pipeline returned no image for prompt_id={prompt.prompt_id}, variant={variant_index}."
            )

        temp_image_path = prompt_temp_dir / f"{prompt.prompt_id}_v{variant_index + 1:02d}.png"
        output.images[0].save(temp_image_path, format="PNG")

        evaluator_output = evaluator(
            prompt=prompt,
            image_path=temp_image_path,
            variant_index=variant_index,
            seed=variant_seed,
            guidance_scale=variant_guidance,
            metadata={
                "prompt_id": prompt.prompt_id,
                "prompt_text": prompt.text,
                "variant_index": variant_index,
                "seed": variant_seed,
                "guidance_scale": variant_guidance,
                "negative_prompt": negative_prompt,
            },
        )
        score, evaluator_details = normalize_evaluator_output(evaluator_output)
        variants.append(
            VariantScoreRecord(
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.text,
                variant_index=variant_index,
                seed=variant_seed,
                guidance_scale=variant_guidance,
                negative_prompt=negative_prompt,
                temp_image_path=str(temp_image_path),
                score=score,
                evaluator_details=evaluator_details,
            )
        )

    best_variant = max(variants, key=lambda record: record.score)
    final_output_dir = Path(config.final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_image_path = final_output_dir / f"{prompt.prompt_id}.png"
    shutil.copy2(best_variant.temp_image_path, final_image_path)

    if not config.keep_temp_images:
        shutil.rmtree(prompt_temp_dir, ignore_errors=True)

    selection = PromptSelectionRecord(
        prompt_id=prompt.prompt_id,
        best_variant_index=best_variant.variant_index,
        best_score=best_variant.score,
        best_temp_image_path=best_variant.temp_image_path,
        final_image_path=str(final_image_path),
    )
    return variants, selection


def write_multisample_metadata(
    *,
    report: MultiSampleRunReport,
    metadata_path: str | Path,
) -> None:
    """Persist multisample metadata report to JSON."""
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def run_multisample_generation(
    config: MultiSampleGenerationConfig,
    *,
    prompts: Sequence[PromptRecord] | None = None,
    pipeline: TextToImagePipeline | None = None,
    evaluator: VariantEvaluator | None = None,
) -> MultiSampleRunReport:
    """Run multi-sampling, evaluate variants, and keep best image per prompt."""
    prompt_records = list(prompts) if prompts is not None else load_prompt_file(config.prompt_file)
    if not prompt_records:
        raise ValueError("No prompts found for multisample generation.")

    device = resolve_device(config.device)
    active_pipeline = pipeline if pipeline is not None else build_pipeline(config, device)
    active_evaluator = evaluator if evaluator is not None else load_variant_evaluator(config.evaluator_spec)
    temp_output_dir = Path(config.temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    strategy_nlp = (
        maybe_load_nlp_for_negative_prompt_strategy(config.negative_prompt_strategy)
        if config.use_structured_negative_prompt
        else None
    )

    all_variants: list[VariantScoreRecord] = []
    selections: list[PromptSelectionRecord] = []
    for prompt_index, prompt in enumerate(prompt_records):
        variants, selection = generate_variants_for_prompt(
            prompt=prompt,
            prompt_index=prompt_index,
            pipeline=active_pipeline,
            config=config,
            device=device,
            evaluator=active_evaluator,
            temp_output_dir=temp_output_dir,
            strategy_nlp=strategy_nlp,
        )
        all_variants.extend(variants)
        selections.append(selection)

    report = MultiSampleRunReport(
        prompt_count=len(prompt_records),
        num_variants=config.num_variants,
        metadata_path=str(Path(config.metadata_path).resolve()),
        selections=tuple(selections),
        variants=tuple(all_variants),
    )
    write_multisample_metadata(report=report, metadata_path=config.metadata_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON generation config.",
    )
    parser.add_argument("--prompt-file", help="Optional prompt file override.")
    parser.add_argument("--final-output-dir", help="Optional final output directory override.")
    parser.add_argument("--temp-output-dir", help="Optional temporary output directory override.")
    parser.add_argument("--metadata-path", help="Optional metadata JSON path override.")
    parser.add_argument("--num-variants", type=int, help="Number of variants per prompt (4-8).")
    parser.add_argument(
        "--evaluator",
        help="Evaluator spec in `module:function` format.",
    )
    parser.add_argument(
        "--keep-temp-images",
        action="store_true",
        help="Keep temporary variant images after selecting best variant.",
    )
    parser.add_argument(
        "--cleanup-temp-images",
        action="store_true",
        help="Delete temporary variant images after selecting best variant.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: MultiSampleGenerationConfig,
    args: argparse.Namespace,
) -> MultiSampleGenerationConfig:
    """Apply explicit CLI overrides on top of config values."""
    updates: dict[str, Any] = {}
    if args.prompt_file is not None:
        updates["prompt_file"] = args.prompt_file
    if args.final_output_dir is not None:
        updates["final_output_dir"] = args.final_output_dir
    if args.temp_output_dir is not None:
        updates["temp_output_dir"] = args.temp_output_dir
    if args.metadata_path is not None:
        updates["metadata_path"] = args.metadata_path
    if args.num_variants is not None:
        updates["num_variants"] = args.num_variants
    if args.evaluator is not None:
        updates["evaluator_spec"] = args.evaluator
    if args.keep_temp_images and args.cleanup_temp_images:
        raise ValueError("Cannot pass both --keep-temp-images and --cleanup-temp-images.")
    if args.keep_temp_images:
        updates["keep_temp_images"] = True
    if args.cleanup_temp_images:
        updates["keep_temp_images"] = False

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return MultiSampleGenerationConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        config = load_multisample_config(args.config)
        config = apply_cli_overrides(config, args)
        report = run_multisample_generation(config)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "prompt_count": report.prompt_count,
                    "num_variants": report.num_variants,
                    "metadata_path": report.metadata_path,
                    "final_output_dir": str(Path(config.final_output_dir).resolve()),
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
