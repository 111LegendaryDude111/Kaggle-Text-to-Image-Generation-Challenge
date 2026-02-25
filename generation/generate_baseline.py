#!/usr/bin/env python3
"""Generate one baseline image per prompt and save as `{prompt_id}.png`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.negative_prompt_strategy import (
        NegativePromptStrategyConfig,
        build_negative_prompt_for_prompt,
        maybe_load_nlp_for_negative_prompt_strategy,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.negative_prompt_strategy import (
        NegativePromptStrategyConfig,
        build_negative_prompt_for_prompt,
        maybe_load_nlp_for_negative_prompt_strategy,
    )


_SAMPLER_ALIASES: dict[str, str] = {
    "default": "default",
    "model_default": "default",
    "ddim": "ddim",
    "pndm": "pndm",
    "lms": "lms",
    "euler": "euler",
    "euler_a": "euler_a",
    "euler_ancestral": "euler_a",
    "heun": "heun",
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp2m": "dpmpp_2m",
    "dpm_solver_multistep": "dpmpp_2m",
    "dpmpp_1s": "dpmpp_1s",
    "dpm_solver_singlestep": "dpmpp_1s",
    "deis": "deis",
    "unipc": "unipc",
    "dpm2": "dpm2",
    "dpm2_a": "dpm2_a",
}
_SUPPORTED_SAMPLERS: frozenset[str] = frozenset(_SAMPLER_ALIASES.values())


class PipelineOutput(Protocol):
    """Minimal output contract expected from a text-to-image pipeline."""

    images: Sequence[Any]


class TextToImagePipeline(Protocol):
    """Minimal callable contract expected from diffusers pipeline."""

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
        """Generate image(s) for one prompt."""


class BaselineGenerationConfig(BaseModel):
    """Configuration for deterministic baseline generation."""

    model_config = ConfigDict(extra="ignore")

    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_variant: str | None = None
    use_safetensors: bool = True

    prompt_file: str = "prompts/DreamLayer-Prompt-Kaggle.txt"
    output_dir: str = "generation/outputs/baseline"

    seed: int = Field(default=42, ge=0)
    seed_strategy: Literal["fixed", "incremental"] = "incremental"

    guidance_scale: float = Field(default=7.5, ge=0.0)
    num_inference_steps: int = Field(default=30, ge=1, le=200)
    sampler: str = "default"
    width: int = Field(default=1024, ge=64)
    height: int = Field(default=1024, ge=64)
    negative_prompt: str = ""
    use_structured_negative_prompt: bool = True
    negative_prompt_strategy: NegativePromptStrategyConfig = Field(
        default_factory=NegativePromptStrategyConfig
    )

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_name cannot be empty")
        return normalized

    @field_validator("width", "height")
    @classmethod
    def _validate_resolution_multiple_of_8(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("resolution must be a multiple of 8")
        return value

    @field_validator("sampler")
    @classmethod
    def _validate_sampler(cls, value: str) -> str:
        normalized = normalize_sampler_name(value)
        if normalized not in _SUPPORTED_SAMPLERS:
            supported = ", ".join(sorted(_SUPPORTED_SAMPLERS))
            raise ValueError(
                f"Unsupported sampler '{value}'. Supported samplers: {supported}"
            )
        return normalized


@dataclass(frozen=True, slots=True)
class GeneratedImageRecord:
    """Metadata for one generated prompt image."""

    prompt_id: str
    prompt_text: str
    output_path: str
    seed: int


def load_generation_config(config_path: str | Path) -> BaselineGenerationConfig:
    """Load baseline generation config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Generation config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("baseline"), dict):
        payload = payload["baseline"]
    if not isinstance(payload, dict):
        raise ValueError("generation config JSON must be an object")

    return BaselineGenerationConfig.model_validate(payload)


def resolve_device(requested_device: Literal["auto", "cpu", "cuda", "mps"]) -> str:
    """Resolve runtime device with availability checks."""
    import torch

    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested_device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested but not available.")
    return requested_device


def resolve_torch_dtype(
    dtype_name: Literal["auto", "float32", "float16", "bfloat16"],
    device: str,
) -> Any:
    """Resolve torch dtype from config value and selected device."""
    import torch

    if dtype_name == "auto":
        return torch.float16 if device in {"cuda", "mps"} else torch.float32

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    resolved_dtype = dtype_map[dtype_name]
    if device == "cpu" and resolved_dtype == torch.float16:
        return torch.float32
    return resolved_dtype


def normalize_sampler_name(sampler: str) -> str:
    """Normalize sampler aliases to canonical scheduler names."""
    normalized = sampler.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace(".", "_")
    normalized = "_".join(segment for segment in normalized.split("_") if segment)
    if not normalized:
        raise ValueError("sampler cannot be empty")
    return _SAMPLER_ALIASES.get(normalized, normalized)


def apply_sampler_to_pipeline(
    pipeline: TextToImagePipeline,
    *,
    sampler: str,
) -> TextToImagePipeline:
    """Attach a scheduler to a diffusers pipeline based on sampler name."""
    normalized_sampler = normalize_sampler_name(sampler)
    if normalized_sampler == "default":
        return pipeline

    from diffusers import (
        DDIMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )

    scheduler_by_sampler: dict[str, type[Any]] = {
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "heun": HeunDiscreteScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "dpmpp_1s": DPMSolverSinglestepScheduler,
        "deis": DEISMultistepScheduler,
        "unipc": UniPCMultistepScheduler,
        "dpm2": KDPM2DiscreteScheduler,
        "dpm2_a": KDPM2AncestralDiscreteScheduler,
    }

    scheduler_cls = scheduler_by_sampler.get(normalized_sampler)
    if scheduler_cls is None:
        supported = ", ".join(sorted(scheduler_by_sampler.keys()))
        raise ValueError(
            f"Unsupported sampler '{sampler}'. Supported samplers: {supported}, default"
        )

    pipeline_any: Any = pipeline
    current_scheduler = getattr(pipeline_any, "scheduler", None)
    if current_scheduler is None or not hasattr(current_scheduler, "config"):
        raise RuntimeError(
            "Pipeline does not expose a configurable scheduler; cannot apply sampler."
        )
    pipeline_any.scheduler = scheduler_cls.from_config(current_scheduler.config)
    return pipeline


def build_pipeline(
    config: BaselineGenerationConfig, device: str
) -> TextToImagePipeline:
    """Create and place the diffusion pipeline on target device."""
    from diffusers import AutoPipelineForText2Image

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(config.torch_dtype, device),
        "use_safetensors": config.use_safetensors,
    }
    if config.model_variant:
        model_kwargs["variant"] = config.model_variant

    pipeline = AutoPipelineForText2Image.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    pipeline = apply_sampler_to_pipeline(pipeline, sampler=config.sampler)
    pipeline = pipeline.to(device)

    set_progress_bar_config = getattr(pipeline, "set_progress_bar_config", None)
    if callable(set_progress_bar_config):
        set_progress_bar_config(disable=True)

    return pipeline


def compute_prompt_seed(
    base_seed: int,
    prompt_index: int,
    seed_strategy: Literal["fixed", "incremental"],
) -> int:
    """Derive deterministic seed for one prompt."""
    if seed_strategy == "fixed":
        return base_seed
    return base_seed + prompt_index


def create_torch_generator(device: str, seed: int) -> Any:
    """Build deterministic torch generator for diffusion sampling."""
    import torch

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def generate_images_for_prompts(
    prompts: Sequence[PromptRecord],
    pipeline: TextToImagePipeline,
    config: BaselineGenerationConfig,
    output_dir: Path,
    device: str,
) -> list[GeneratedImageRecord]:
    """Generate and persist one PNG image per prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[GeneratedImageRecord] = []
    strategy_nlp = (
        maybe_load_nlp_for_negative_prompt_strategy(config.negative_prompt_strategy)
        if config.use_structured_negative_prompt
        else None
    )

    for index, prompt in enumerate(prompts):
        prompt_seed = compute_prompt_seed(config.seed, index, config.seed_strategy)
        runtime_negative_prompt = config.negative_prompt.strip()
        if config.use_structured_negative_prompt:
            runtime_negative_prompt = build_negative_prompt_for_prompt(
                prompt,
                base_negative_prompt=config.negative_prompt,
                config=config.negative_prompt_strategy,
                nlp=strategy_nlp,
            )

        output = pipeline(
            prompt=prompt.text,
            negative_prompt=runtime_negative_prompt or None,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            width=config.width,
            height=config.height,
            num_images_per_prompt=1,
            generator=create_torch_generator(device, prompt_seed),
        )

        if not output.images:
            raise RuntimeError(
                f"Pipeline returned no image for prompt_id={prompt.prompt_id}"
            )

        image = output.images[0]
        image_path = output_dir / f"{prompt.prompt_id}.png"
        image.save(image_path, format="PNG")

        records.append(
            GeneratedImageRecord(
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.text,
                output_path=str(image_path),
                seed=prompt_seed,
            )
        )

    return records


def run_baseline_generation(
    config: BaselineGenerationConfig,
    *,
    prompts: Sequence[PromptRecord] | None = None,
    pipeline: TextToImagePipeline | None = None,
) -> list[GeneratedImageRecord]:
    """Run baseline generation end-to-end for all prompts."""
    prompt_records = (
        list(prompts) if prompts is not None else load_prompt_file(config.prompt_file)
    )
    if not prompt_records:
        raise ValueError("No prompts found for baseline generation.")

    device = resolve_device(config.device)
    active_pipeline = (
        pipeline if pipeline is not None else build_pipeline(config, device)
    )
    output_dir = Path(config.output_dir)
    return generate_images_for_prompts(
        prompt_records, active_pipeline, config, output_dir, device
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to JSON generation config.",
    )
    parser.add_argument("--prompt-file", help="Optional prompt file override.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument("--seed", type=int, help="Optional seed override.")
    parser.add_argument(
        "--seed-strategy",
        choices=("fixed", "incremental"),
        help="Optional seed strategy override.",
    )
    parser.add_argument(
        "--guidance-scale", type=float, help="Optional guidance scale override."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Optional diffusion step count override.",
    )
    parser.add_argument(
        "--sampler",
        help="Optional sampler override (e.g., default, euler_a, dpmpp_2m).",
    )
    parser.add_argument("--width", type=int, help="Optional image width override.")
    parser.add_argument("--height", type=int, help="Optional image height override.")
    return parser.parse_args()


def apply_cli_overrides(
    config: BaselineGenerationConfig, args: argparse.Namespace
) -> BaselineGenerationConfig:
    """Apply explicit CLI overrides on top of config file values."""
    updates: dict[str, Any] = {}
    override_fields = (
        "prompt_file",
        "output_dir",
        "seed",
        "seed_strategy",
        "guidance_scale",
        "num_inference_steps",
        "sampler",
        "width",
        "height",
    )
    for field_name in override_fields:
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return BaselineGenerationConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        config = load_generation_config(args.config)
        config = apply_cli_overrides(config, args)
        results = run_baseline_generation(config)
        payload = {
            "status": "ok",
            "generated_count": len(results),
            "output_dir": str(Path(config.output_dir).resolve()),
            "seed": config.seed,
            "seed_strategy": config.seed_strategy,
            "guidance_scale": config.guidance_scale,
            "sampler": config.sampler,
        }
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
