#!/usr/bin/env python3
"""GPU compatibility smoke test for deterministic torch execution."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class GpuTestReport:
    python_version: str
    torch_version: str
    cuda_available: bool
    mps_available: bool
    selected_device: str
    elapsed_sec: float
    checksum: float


def _sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        return
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def run_test(device: str) -> GpuTestReport:
    torch.manual_seed(42)

    a = torch.randn((512, 512), device=device, dtype=torch.float32)
    b = torch.randn((512, 512), device=device, dtype=torch.float32)

    started = time.perf_counter()
    product = a @ b
    _sync_if_needed(device)
    elapsed_sec = time.perf_counter() - started

    return GpuTestReport(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        mps_available=bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        selected_device=device,
        elapsed_sec=elapsed_sec,
        checksum=float(product.sum().item()),
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA/MPS device is not available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = pick_device()

    if args.require_gpu and device == "cpu":
        payload = {
            "status": "failed",
            "reason": "No GPU backend available (CUDA/MPS unavailable).",
        }
        print(json.dumps(payload, indent=2))
        return 2

    report = run_test(device)
    payload = {"status": "ok", "report": asdict(report)}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
