#!/usr/bin/env python3
"""Run a local YOLO inference smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


@dataclass(frozen=True)
class YoloSmokeReport:
    weights: str
    image_path: str
    num_results: int
    num_boxes: int
    labels: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="models/yolo/yolov8n.pt",
        help="Path to local YOLO weights (.pt).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow ultralytics to auto-download weights if local file is missing.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    return parser.parse_args()


def build_test_image() -> Path:
    canvas = Image.fromarray(np.full((640, 640, 3), 255, dtype=np.uint8))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(120, 260), (520, 520)], outline="black", width=6)
    draw.ellipse([(260, 180), (360, 280)], outline="black", width=6)

    output = Path(tempfile.gettempdir()) / "t2i_f1_yolo_smoke.jpg"
    canvas.save(output)
    return output


def resolve_weights(path: Path, allow_download: bool) -> str:
    if path.exists():
        return str(path)
    if allow_download:
        return "yolov8n.pt"
    raise FileNotFoundError(
        f"Weights file not found at '{path}'. "
        "Provide local weights or rerun with --allow-download."
    )


def main() -> int:
    args = parse_args()
    image_path = build_test_image()

    try:
        weights = resolve_weights(Path(args.weights), args.allow_download)
        model = YOLO(weights)
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )

        first = results[0]
        labels: list[str] = []
        if len(first.boxes) > 0:
            names = first.names
            for cls_id in first.boxes.cls.tolist():
                labels.append(str(names[int(cls_id)]))

        report = YoloSmokeReport(
            weights=weights,
            image_path=str(image_path),
            num_results=len(results),
            num_boxes=len(first.boxes),
            labels=sorted(set(labels)),
        )
        print(json.dumps({"status": "ok", "report": asdict(report)}, indent=2))
        return 0
    except Exception as exc:
        payload = {"status": "failed", "reason": str(exc)}
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
