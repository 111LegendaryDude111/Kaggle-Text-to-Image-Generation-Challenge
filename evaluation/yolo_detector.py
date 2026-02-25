#!/usr/bin/env python3
"""Detect objects with YOLO and normalize labels for F1 evaluation."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from evaluation.prompt_parser import normalize_to_yolo_label
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.prompt_parser import normalize_to_yolo_label

_WHITESPACE_RE = re.compile(r"[\s/_-]+")
_DROP_CHARS_RE = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE_RE = re.compile(r"\s+")


class BoxesLike(Protocol):
    """Minimal YOLO boxes contract used by the detector adapter."""

    cls: Any
    conf: Any
    xyxy: Any

    def __len__(self) -> int:
        """Return number of detected boxes."""


class YoloResultLike(Protocol):
    """Minimal YOLO result contract used by the detector adapter."""

    names: Mapping[int, str] | Sequence[str]
    boxes: BoxesLike


class YoloModelLike(Protocol):
    """Minimal callable model contract for Ultralytics YOLO inference."""

    def predict(
        self,
        *,
        source: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        device: str | None,
        verbose: bool,
    ) -> Sequence[YoloResultLike]:
        """Run YOLO inference for one image path."""


class YoloDetectionConfig(BaseModel):
    """Configuration for YOLO object detection."""

    model_config = ConfigDict(extra="ignore")

    weights: str = "models/yolo/yolov8n.pt"
    allow_download: bool = False

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32)
    max_det: int = Field(default=300, ge=1)
    device: str | None = None
    verbose: bool = False

    keep_unmapped_labels: bool = False
    label_aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("weights cannot be empty")
        return normalized

    @field_validator("label_aliases")
    @classmethod
    def _validate_label_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, raw_target in value.items():
            key = _normalize_free_text(str(raw_key))
            target = _normalize_free_text(str(raw_target))
            if key and target:
                normalized[key] = target
        return normalized


@dataclass(frozen=True, slots=True)
class YoloDetection:
    """Single normalized object detection."""

    label: str
    confidence: float
    class_id: int
    bbox_xyxy: tuple[float, float, float, float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize detection to JSON-compatible shape."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
        }


@dataclass(frozen=True, slots=True)
class YoloDetectionResult:
    """Detection output for one image."""

    image_path: str
    num_raw_detections: int
    confidence_threshold: float
    detected_labels: tuple[str, ...]
    detections: tuple[YoloDetection, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize detection result to JSON-compatible shape."""
        return {
            "image_path": self.image_path,
            "num_raw_detections": self.num_raw_detections,
            "confidence_threshold": self.confidence_threshold,
            "detected_labels": list(self.detected_labels),
            "detections": [item.to_json_dict() for item in self.detections],
        }


def _normalize_free_text(value: str) -> str:
    normalized = _WHITESPACE_RE.sub(" ", value.strip().lower())
    normalized = _DROP_CHARS_RE.sub("", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        rendered = value.tolist()
        if isinstance(rendered, list):
            return rendered
        return [rendered]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _to_bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = _to_list(value)
    if len(raw) != 4:
        return None
    return (
        float(raw[0]),
        float(raw[1]),
        float(raw[2]),
        float(raw[3]),
    )


def normalize_detected_label(label: str, config: YoloDetectionConfig) -> str | None:
    """Normalize a detected label to YOLO-compatible evaluation vocabulary."""
    normalized_raw = _normalize_free_text(label)
    if not normalized_raw:
        return None

    alias_resolved = config.label_aliases.get(normalized_raw, normalized_raw)
    mapped = normalize_to_yolo_label(alias_resolved)
    if mapped is not None:
        return mapped
    if config.keep_unmapped_labels:
        return alias_resolved
    return None


def _resolve_name(names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    if isinstance(names, Mapping):
        by_int = names.get(class_id)
        if by_int is not None:
            return str(by_int)
        by_str = names.get(str(class_id))  # type: ignore[arg-type]
        if by_str is not None:
            return str(by_str)
        return str(class_id)

    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _resolve_weights(config: YoloDetectionConfig) -> str:
    weights_path = Path(config.weights)
    if weights_path.exists():
        return str(weights_path)
    if config.allow_download:
        return config.weights
    raise FileNotFoundError(
        f"YOLO weights not found at '{weights_path}'. "
        "Provide local weights or set allow_download=True."
    )


def build_yolo_model(config: YoloDetectionConfig) -> YoloModelLike:
    """Construct a YOLO model instance from config."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("ultralytics is not installed in the environment.") from exc

    return YOLO(_resolve_weights(config))


def detect_objects_in_image(
    image_path: str | Path,
    *,
    model: YoloModelLike,
    config: YoloDetectionConfig,
) -> YoloDetectionResult:
    """Run YOLO inference for one image and return normalized detections."""
    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Image file not found: {source_path}")

    prediction = model.predict(
        source=str(source_path),
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        imgsz=config.imgsz,
        max_det=config.max_det,
        device=config.device,
        verbose=config.verbose,
    )
    if not prediction:
        raise RuntimeError(f"YOLO returned no results for image: {source_path}")

    first_result = prediction[0]
    class_ids = [int(value) for value in _to_list(first_result.boxes.cls)]
    confidences = [float(value) for value in _to_list(first_result.boxes.conf)]
    raw_boxes = _to_list(first_result.boxes.xyxy)
    num_raw = len(first_result.boxes)

    detections: list[YoloDetection] = []
    for index, (class_id, confidence) in enumerate(zip(class_ids, confidences)):
        if confidence < config.confidence_threshold:
            continue

        resolved_name = _resolve_name(first_result.names, class_id)
        normalized_label = normalize_detected_label(resolved_name, config=config)
        if normalized_label is None:
            continue

        bbox_xyxy = None
        if index < len(raw_boxes):
            bbox_xyxy = _to_bbox(raw_boxes[index])

        detections.append(
            YoloDetection(
                label=normalized_label,
                confidence=confidence,
                class_id=class_id,
                bbox_xyxy=bbox_xyxy,
            )
        )

    deduped_labels = tuple(_dedupe_preserve_order([item.label for item in detections]))
    return YoloDetectionResult(
        image_path=str(source_path),
        num_raw_detections=num_raw,
        confidence_threshold=config.confidence_threshold,
        detected_labels=deduped_labels,
        detections=tuple(detections),
    )


def detect_objects_for_images(
    image_paths: Sequence[str | Path],
    *,
    model: YoloModelLike,
    config: YoloDetectionConfig,
) -> list[YoloDetectionResult]:
    """Run YOLO detection for multiple images."""
    return [
        detect_objects_in_image(image_path, model=model, config=config)
        for image_path in image_paths
    ]


def load_yolo_detection_config(config_path: str | Path) -> YoloDetectionConfig:
    """Load YOLO detection config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")

    if isinstance(payload.get("evaluation"), dict):
        payload = payload["evaluation"]
    if isinstance(payload.get("yolo"), dict):
        payload = payload["yolo"]
    elif isinstance(payload.get("yolo_detector"), dict):
        payload = payload["yolo_detector"]

    return YoloDetectionConfig.model_validate(payload)
