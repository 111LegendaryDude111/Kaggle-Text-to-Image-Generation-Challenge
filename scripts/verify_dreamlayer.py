#!/usr/bin/env python3
"""Verify DreamLayer installation from source repository."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/DreamLayer-AI/DreamLayer"
EXPECTED_FILES = (
    "README.md",
    "KaggleDreamLayer.ipynb",
    "install_mac_dependencies.sh",
    "start_dream_layer.sh",
    "dream_layer_backend/requirements.txt",
)


@dataclass(frozen=True)
class DreamLayerReport:
    install_path: str
    git_head: str
    expected_files_present: bool
    checked_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="third_party/DreamLayer",
        help="DreamLayer installation path.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git repository URL used for auto-install.",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Clone DreamLayer automatically if not installed.",
    )
    return parser.parse_args()


def clone_repo(repo_url: str, install_path: Path) -> None:
    install_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(install_path)],
        check=True,
        text=True,
    )


def get_git_head(repo_path: Path) -> str:
    cmd = ["git", "-C", str(repo_path), "rev-parse", "HEAD"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def verify_install(install_path: Path) -> DreamLayerReport:
    missing = [name for name in EXPECTED_FILES if not (install_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"DreamLayer files missing: {', '.join(missing)}")

    return DreamLayerReport(
        install_path=str(install_path.resolve()),
        git_head=get_git_head(install_path),
        expected_files_present=True,
        checked_files=list(EXPECTED_FILES),
    )


def main() -> int:
    args = parse_args()
    install_path = Path(args.path)

    try:
        if not install_path.exists():
            if not args.auto_install:
                raise FileNotFoundError(
                    f"DreamLayer not found at '{install_path}'. "
                    "Run with --auto-install to clone automatically."
                )
            if shutil.which("git") is None:
                raise RuntimeError("git is required for DreamLayer auto-install.")
            clone_repo(args.repo_url, install_path)

        report = verify_install(install_path)
        print(json.dumps({"status": "ok", "report": asdict(report)}, indent=2))
        return 0
    except Exception as exc:
        payload = {"status": "failed", "reason": str(exc), "path": str(install_path)}
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
