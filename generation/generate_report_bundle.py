#!/usr/bin/env python3
"""Generate and validate DreamLayer report bundle artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

try:
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.generate_baseline import BaselineGenerationConfig, load_generation_config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_loader import PromptRecord, load_prompt_file
    from generation.generate_baseline import BaselineGenerationConfig, load_generation_config


_RESULTS_COLUMNS: tuple[str, str, str] = ("run_id", "prompt", "filenames")


class ReportBundleGenerationConfig(BaseModel):
    """Configuration for DreamLayer report bundle generation."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "dreamlayer_export"
    source_image_dir: str = "dreamlayer_export/final_images"
    results_csv_path: str = "dreamlayer_export/results.csv"
    config_dreamlayer_path: str = "dreamlayer_export/config-dreamlayer.json"

    prompt_file: str | None = None

    copy_images_to_output_dir: bool = True
    clean_output_images: bool = True
    enforce_exact_one_image_per_prompt: bool = True

    validate_integrity: bool = True
    confirm_no_manual_edits: bool = True

    final_generation_report_path: str = "dreamlayer_export/final_generation_report.json"
    frozen_config_path: str = "dreamlayer_export/frozen_generation_config.json"


@dataclass(frozen=True, slots=True)
class BundleRow:
    """One results.csv row record."""

    run_id: str
    prompt: str
    filenames: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "prompt": self.prompt,
            "filenames": self.filenames,
        }


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Hash and size information for one bundle artifact."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ReportBundleGenerationReport:
    """Summary for a generated DreamLayer bundle."""

    prompt_count: int
    image_count: int
    output_dir: str
    results_csv_path: str
    config_dreamlayer_path: str
    integrity_valid: bool
    no_manual_edits_confirmed: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "prompt_count": self.prompt_count,
            "image_count": self.image_count,
            "output_dir": self.output_dir,
            "results_csv_path": self.results_csv_path,
            "config_dreamlayer_path": self.config_dreamlayer_path,
            "integrity_valid": self.integrity_valid,
            "no_manual_edits_confirmed": self.no_manual_edits_confirmed,
        }


def load_report_bundle_generation_config(
    config_path: str | Path,
) -> ReportBundleGenerationConfig:
    """Load report bundle generation config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("export"), dict):
        payload = payload["export"]
    if isinstance(payload.get("report_bundle_generation"), dict):
        payload = payload["report_bundle_generation"]
    else:
        payload = {}

    return ReportBundleGenerationConfig.model_validate(payload)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path, output_dir: Path) -> str:
    resolved = path.resolve()
    output_resolved = output_dir.resolve()
    try:
        return resolved.relative_to(output_resolved).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_manifest_path(manifest_path: str, output_dir: Path) -> Path:
    candidate = Path(manifest_path)
    if candidate.is_absolute():
        return candidate
    return output_dir / candidate


def _resolve_prompt_records(
    bundle_config: ReportBundleGenerationConfig,
    generation_config: BaselineGenerationConfig,
    prompts: Sequence[PromptRecord] | None,
) -> tuple[list[PromptRecord], str]:
    if prompts is not None:
        prompt_file = bundle_config.prompt_file or generation_config.prompt_file
        return list(prompts), prompt_file

    prompt_file = bundle_config.prompt_file or generation_config.prompt_file
    return load_prompt_file(prompt_file), prompt_file


def _validate_source_images(
    source_image_dir: Path,
    expected_filenames: Sequence[str],
    *,
    enforce_exact_one_image_per_prompt: bool,
) -> None:
    if not source_image_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_image_dir}")

    expected_set = set(expected_filenames)
    source_pngs = {path.name for path in source_image_dir.glob("*.png")}

    missing = sorted(expected_set - source_pngs)
    if missing:
        raise RuntimeError(
            "Missing final images required for report bundle: "
            f"{missing}"
        )

    if enforce_exact_one_image_per_prompt:
        extras = sorted(source_pngs - expected_set)
        if extras:
            raise RuntimeError(
                "Source image directory contains unexpected PNGs. "
                f"Extra files: {extras}"
            )


def _prepare_bundle_images(
    *,
    source_image_dir: Path,
    output_dir: Path,
    expected_filenames: Sequence[str],
    copy_images_to_output_dir: bool,
    clean_output_images: bool,
) -> Path:
    if not copy_images_to_output_dir:
        return source_image_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    source_resolved = source_image_dir.resolve()
    output_resolved = output_dir.resolve()
    if clean_output_images and source_resolved != output_resolved:
        for path in output_dir.glob("*.png"):
            path.unlink()

    for filename in expected_filenames:
        source_path = source_image_dir / filename
        destination_path = output_dir / filename

        if source_path.resolve() == destination_path.resolve():
            continue

        shutil.copy2(source_path, destination_path)

    return output_dir


def _build_rows(
    prompts: Sequence[PromptRecord],
    *,
    image_relative_prefix: str,
) -> list[BundleRow]:
    rows: list[BundleRow] = []
    for prompt in prompts:
        filename = f"{prompt.prompt_id}.png"
        if image_relative_prefix:
            filename = f"{image_relative_prefix}/{filename}"

        rows.append(
            BundleRow(
                run_id=prompt.prompt_id,
                prompt=prompt.text,
                filenames=filename,
            )
        )
    return rows


def _write_results_csv(rows: Sequence[BundleRow], results_csv_path: Path) -> FileDigest:
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with results_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_RESULTS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())

    return FileDigest(
        path="",
        sha256=_sha256_file(results_csv_path),
        size_bytes=results_csv_path.stat().st_size,
    )


def _collect_image_digests(
    *,
    rows: Sequence[BundleRow],
    output_dir: Path,
) -> list[FileDigest]:
    digests: list[FileDigest] = []
    for row in rows:
        image_path = output_dir / row.filenames
        if not image_path.exists():
            raise FileNotFoundError(f"Bundle image not found: {image_path}")

        digests.append(
            FileDigest(
                path="",
                sha256=_sha256_file(image_path),
                size_bytes=image_path.stat().st_size,
            )
        )
    return digests


def _build_optional_artifact_digest(path: str | Path) -> dict[str, Any] | None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None

    return {
        "path": str(artifact_path.resolve()),
        "sha256": _sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def validate_report_bundle_from_config(
    config_dreamlayer_path: str | Path,
) -> tuple[bool, bool]:
    """Validate bundle artifacts against config-dreamlayer.json contract."""
    config_path = Path(config_dreamlayer_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config-dreamlayer.json not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config-dreamlayer.json must contain a JSON object")

    bundle_payload = payload.get("bundle")
    if not isinstance(bundle_payload, dict):
        raise ValueError("config-dreamlayer.json missing 'bundle' object")

    output_dir_raw = bundle_payload.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValueError("config-dreamlayer.json missing bundle.output_dir")
    output_dir = Path(output_dir_raw)

    results_contract = payload.get("results_contract")
    if not isinstance(results_contract, dict):
        raise ValueError("config-dreamlayer.json missing 'results_contract' object")

    expected_columns = results_contract.get("columns")
    if expected_columns != list(_RESULTS_COLUMNS):
        raise ValueError("results_contract.columns does not match expected schema")

    expected_rows_raw = results_contract.get("rows")
    if not isinstance(expected_rows_raw, list):
        raise ValueError("results_contract.rows must be a list")
    expected_rows: list[dict[str, str]] = []
    for row in expected_rows_raw:
        if not isinstance(row, dict):
            raise ValueError("results_contract.rows must contain objects")
        expected_rows.append({column: str(row.get(column, "")) for column in _RESULTS_COLUMNS})

    integrity_payload = payload.get("integrity")
    if not isinstance(integrity_payload, dict):
        raise ValueError("config-dreamlayer.json missing 'integrity' object")

    results_info = integrity_payload.get("results_csv")
    if not isinstance(results_info, dict):
        raise ValueError("integrity.results_csv must be an object")

    results_manifest_path = results_info.get("path")
    if not isinstance(results_manifest_path, str):
        raise ValueError("integrity.results_csv.path must be a string")

    results_csv_path = _resolve_manifest_path(results_manifest_path, output_dir)
    if not results_csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv_path}")

    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(_RESULTS_COLUMNS):
            raise RuntimeError(
                "results.csv header mismatch. "
                f"Expected {list(_RESULTS_COLUMNS)}, got {reader.fieldnames}."
            )
        actual_rows = [
            {column: str(row.get(column, "")) for column in _RESULTS_COLUMNS}
            for row in reader
        ]

    if actual_rows != expected_rows:
        raise RuntimeError("results.csv content does not match generated contract")

    actual_results_sha = _sha256_file(results_csv_path)
    if actual_results_sha != results_info.get("sha256"):
        raise RuntimeError("results.csv SHA256 mismatch; manual edits detected")

    actual_results_size = results_csv_path.stat().st_size
    if actual_results_size != int(results_info.get("size_bytes", -1)):
        raise RuntimeError("results.csv size mismatch; manual edits detected")

    images_payload = integrity_payload.get("images")
    if not isinstance(images_payload, list):
        raise ValueError("integrity.images must be a list")

    for image_info in images_payload:
        if not isinstance(image_info, dict):
            raise ValueError("integrity.images items must be objects")

        manifest_path = image_info.get("path")
        if not isinstance(manifest_path, str):
            raise ValueError("image digest path must be a string")

        image_path = _resolve_manifest_path(manifest_path, output_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Bundle image missing: {image_path}")

        expected_sha = image_info.get("sha256")
        if not isinstance(expected_sha, str):
            raise ValueError("image digest sha256 must be a string")

        actual_sha = _sha256_file(image_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Image SHA256 mismatch for {image_path.name}")

        expected_size = int(image_info.get("size_bytes", -1))
        actual_size = image_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"Image size mismatch for {image_path.name}")

    return True, True


def run_report_bundle_generation(
    bundle_config: ReportBundleGenerationConfig,
    *,
    generation_config: BaselineGenerationConfig,
    prompts: Sequence[PromptRecord] | None = None,
) -> ReportBundleGenerationReport:
    """Generate results.csv + config-dreamlayer.json and validate integrity."""
    prompt_records, prompt_file = _resolve_prompt_records(
        bundle_config,
        generation_config,
        prompts,
    )
    if not prompt_records:
        raise ValueError("No prompts available for report bundle generation.")

    expected_filenames = tuple(f"{prompt.prompt_id}.png" for prompt in prompt_records)

    source_image_dir = Path(bundle_config.source_image_dir)
    output_dir = Path(bundle_config.output_dir)
    _validate_source_images(
        source_image_dir,
        expected_filenames,
        enforce_exact_one_image_per_prompt=bundle_config.enforce_exact_one_image_per_prompt,
    )

    bundle_image_dir = _prepare_bundle_images(
        source_image_dir=source_image_dir,
        output_dir=output_dir,
        expected_filenames=expected_filenames,
        copy_images_to_output_dir=bundle_config.copy_images_to_output_dir,
        clean_output_images=bundle_config.clean_output_images,
    )

    output_resolved = output_dir.resolve()
    image_resolved = bundle_image_dir.resolve()
    image_relative_prefix = ""
    try:
        relative_dir = image_resolved.relative_to(output_resolved)
        if str(relative_dir) != ".":
            image_relative_prefix = relative_dir.as_posix()
    except ValueError:
        image_relative_prefix = ""

    rows = _build_rows(prompt_records, image_relative_prefix=image_relative_prefix)

    results_csv_path = Path(bundle_config.results_csv_path)
    results_digest = _write_results_csv(rows, results_csv_path)
    results_digest = FileDigest(
        path=_manifest_path(results_csv_path, output_dir),
        sha256=results_digest.sha256,
        size_bytes=results_digest.size_bytes,
    )

    image_digests = _collect_image_digests(rows=rows, output_dir=output_dir)
    image_digests = [
        FileDigest(
            path=_manifest_path(output_dir / row.filenames, output_dir),
            sha256=digest.sha256,
            size_bytes=digest.size_bytes,
        )
        for row, digest in zip(rows, image_digests, strict=True)
    ]

    config_payload: dict[str, Any] = {
        "schema_version": "dreamlayer_export_v1",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "generator": "generation/generate_report_bundle.py",
        "bundle": {
            "output_dir": str(output_dir.resolve()),
            "source_image_dir": str(source_image_dir.resolve()),
            "image_directory": str(bundle_image_dir.resolve()),
            "prompt_file": str(Path(prompt_file).resolve()),
            "prompt_count": len(prompt_records),
            "image_count": len(rows),
            "copy_images_to_output_dir": bundle_config.copy_images_to_output_dir,
            "results_columns": list(_RESULTS_COLUMNS),
            "generation_settings": {
                "model_name": generation_config.model_name,
                "seed": generation_config.seed,
                "guidance_scale": generation_config.guidance_scale,
                "num_inference_steps": generation_config.num_inference_steps,
                "sampler": generation_config.sampler,
            },
        },
        "results_contract": {
            "columns": list(_RESULTS_COLUMNS),
            "row_count": len(rows),
            "rows": [row.to_dict() for row in rows],
        },
        "integrity": {
            "results_csv": asdict(results_digest),
            "images": [asdict(digest) for digest in image_digests],
        },
        "source_artifacts": {
            "final_generation_report": _build_optional_artifact_digest(
                bundle_config.final_generation_report_path
            ),
            "frozen_generation_config": _build_optional_artifact_digest(
                bundle_config.frozen_config_path
            ),
        },
    }

    config_dreamlayer_path = Path(bundle_config.config_dreamlayer_path)
    config_dreamlayer_path.parent.mkdir(parents=True, exist_ok=True)
    config_dreamlayer_path.write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    integrity_valid = True
    no_manual_edits_confirmed = True
    if bundle_config.validate_integrity or bundle_config.confirm_no_manual_edits:
        integrity_valid, no_manual_edits_confirmed = validate_report_bundle_from_config(
            config_dreamlayer_path
        )

    return ReportBundleGenerationReport(
        prompt_count=len(prompt_records),
        image_count=len(rows),
        output_dir=str(output_dir.resolve()),
        results_csv_path=str(results_csv_path.resolve()),
        config_dreamlayer_path=str(config_dreamlayer_path.resolve()),
        integrity_valid=integrity_valid,
        no_manual_edits_confirmed=no_manual_edits_confirmed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation_config.json",
        help="Path to generation config JSON.",
    )
    parser.add_argument("--output-dir", help="Optional bundle output directory override.")
    parser.add_argument("--source-image-dir", help="Optional source image directory override.")
    parser.add_argument("--results-csv-path", help="Optional results.csv output path override.")
    parser.add_argument(
        "--config-dreamlayer-path",
        help="Optional config-dreamlayer.json output path override.",
    )
    parser.add_argument("--prompt-file", help="Optional prompt file override.")
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy images into output_dir before writing bundle files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing bundle files using config-dreamlayer.json.",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: ReportBundleGenerationConfig,
    args: argparse.Namespace,
) -> ReportBundleGenerationConfig:
    """Apply explicit CLI overrides on top of JSON config values."""
    updates: dict[str, Any] = {}
    for field_name in (
        "output_dir",
        "source_image_dir",
        "results_csv_path",
        "config_dreamlayer_path",
        "prompt_file",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    if args.no_copy_images:
        updates["copy_images_to_output_dir"] = False

    if not updates:
        return config

    merged = config.model_dump()
    merged.update(updates)
    return ReportBundleGenerationConfig.model_validate(merged)


def main() -> int:
    args = parse_args()
    try:
        bundle_config = load_report_bundle_generation_config(args.config)
        bundle_config = apply_cli_overrides(bundle_config, args)

        if args.validate_only:
            integrity_valid, no_manual_edits_confirmed = validate_report_bundle_from_config(
                bundle_config.config_dreamlayer_path
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "integrity_valid": integrity_valid,
                        "no_manual_edits_confirmed": no_manual_edits_confirmed,
                        "config_dreamlayer_path": str(
                            Path(bundle_config.config_dreamlayer_path).resolve()
                        ),
                    },
                    indent=2,
                )
            )
            return 0

        generation_config = load_generation_config(args.config)
        report = run_report_bundle_generation(
            bundle_config,
            generation_config=generation_config,
        )
        print(json.dumps({"status": "ok", **report.to_json_dict()}, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
