from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.yolo_detector import (
    YoloDetectionConfig,
    detect_objects_in_image,
    load_yolo_detection_config,
    normalize_detected_label,
)


class FakeTensor:
    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        return self._data


class FakeBoxes:
    def __init__(self, cls: list[float], conf: list[float], xyxy: list[list[float]]) -> None:
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)
        self.xyxy = FakeTensor(xyxy)

    def __len__(self) -> int:
        return len(self.conf.tolist())


class FakeResult:
    def __init__(self, names: dict[int, str], boxes: FakeBoxes) -> None:
        self.names = names
        self.boxes = boxes


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [self._result]


def test_detect_objects_in_image_filters_confidence_and_normalizes_labels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "0001.png"
    image_path.write_bytes(b"fake")

    fake_result = FakeResult(
        names={0: "dogs", 1: "traffic-light", 2: "dragon"},
        boxes=FakeBoxes(
            cls=[0.0, 0.0, 1.0, 2.0],
            conf=[0.95, 0.49, 0.80, 0.92],
            xyxy=[
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        ),
    )
    model = FakeModel(fake_result)
    config = YoloDetectionConfig(confidence_threshold=0.5, keep_unmapped_labels=False)

    result = detect_objects_in_image(image_path, model=model, config=config)

    assert result.num_raw_detections == 4
    assert result.detected_labels == ("dog", "traffic light")
    assert [item.label for item in result.detections] == ["dog", "traffic light"]
    assert [round(item.confidence, 2) for item in result.detections] == [0.95, 0.8]
    assert model.calls[0]["conf"] == 0.5


def test_normalize_detected_label_respects_aliases_and_keep_unmapped() -> None:
    config = YoloDetectionConfig(
        label_aliases={"tv-monitor": "television"},
        keep_unmapped_labels=False,
    )
    assert normalize_detected_label("TV monitor", config) == "tv"
    assert normalize_detected_label("mystery-object", config) is None

    keep_config = YoloDetectionConfig(keep_unmapped_labels=True)
    assert normalize_detected_label("mystery-object", keep_config) == "mystery object"


def test_load_yolo_detection_config_reads_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "yolo": {
                        "weights": "models/yolo/custom.pt",
                        "confidence_threshold": 0.33,
                        "keep_unmapped_labels": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_yolo_detection_config(config_path)

    assert config.weights == "models/yolo/custom.pt"
    assert config.confidence_threshold == 0.33
    assert config.keep_unmapped_labels is True
