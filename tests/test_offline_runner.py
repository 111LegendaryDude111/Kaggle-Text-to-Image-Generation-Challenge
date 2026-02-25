from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.offline_runner import (
    OfflineValidationConfig,
    load_offline_validation_config,
    run_offline_validation,
)
from evaluation.prompt_loader import PromptRecord
from evaluation.yolo_detector import YoloDetectionConfig
from generation.generate_baseline import BaselineGenerationConfig


class FakePipelineOutput:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "FakePipeline":
        return self

    def __call__(self, **kwargs: Any) -> FakePipelineOutput:
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(255, 255, 255))
        return FakePipelineOutput(image)


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


class FakeYoloModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        filename = Path(str(kwargs["source"])).name
        if filename == "0001.png":
            return [
                FakeResult(
                    names={0: "dog", 1: "chair"},
                    boxes=FakeBoxes(
                        cls=[0.0, 1.0],
                        conf=[0.9, 0.8],
                        xyxy=[[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]],
                    ),
                )
            ]
        return [
            FakeResult(
                names={0: "chair", 1: "cat"},
                boxes=FakeBoxes(
                    cls=[0.0, 1.0],
                    conf=[0.2, 0.85],
                    xyxy=[[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]],
                ),
            )
        ]


def test_run_offline_validation_end_to_end(tmp_path: Path) -> None:
    prompts = [
        PromptRecord(prompt_id="0001", text="A cat and a dog.", expected_objects=("cat", "dog")),
        PromptRecord(prompt_id="0002", text="A single chair.", expected_objects=("chair",)),
    ]
    generation_config = BaselineGenerationConfig(
        model_name="unit-test-model",
        output_dir=str(tmp_path / "unused_output"),
        seed=10,
        seed_strategy="incremental",
        guidance_scale=7.0,
        num_inference_steps=10,
        width=64,
        height=64,
        device="cpu",
        torch_dtype="float32",
        use_structured_negative_prompt=False,
    )
    yolo_config = YoloDetectionConfig(confidence_threshold=0.5, keep_unmapped_labels=False)
    validation_config = OfflineValidationConfig(
        generated_output_dir=str(tmp_path / "generated"),
        report_path=str(tmp_path / "offline_report.json"),
        per_prompt_csv_path=str(tmp_path / "per_prompt.csv"),
    )
    fake_pipeline = FakePipeline()
    fake_yolo = FakeYoloModel()

    report = run_offline_validation(
        validation_config,
        generation_config=generation_config,
        yolo_config=yolo_config,
        prompts=prompts,
        generation_pipeline=fake_pipeline,
        yolo_model=fake_yolo,
    )

    assert report.prompt_count == 2
    assert round(report.average_precision, 3) == 0.25
    assert round(report.average_recall, 3) == 0.25
    assert round(report.average_f1, 3) == 0.25
    assert (tmp_path / "generated" / "0001.png").exists()
    assert (tmp_path / "generated" / "0002.png").exists()
    assert Path(validation_config.report_path).exists()
    assert Path(validation_config.per_prompt_csv_path).exists()

    loaded_json = json.loads(Path(validation_config.report_path).read_text(encoding="utf-8"))
    assert loaded_json["prompt_count"] == 2
    assert len(loaded_json["results"]) == 2

    with Path(validation_config.per_prompt_csv_path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "0001"
    assert rows[1]["prompt_id"] == "0002"


def test_load_offline_validation_config_reads_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "offline_runner": {
                        "report_path": "experiments/custom/report.json",
                        "per_prompt_csv_path": "experiments/custom/per_prompt.csv",
                        "spacy_model": "en_core_web_md",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_offline_validation_config(config_path)

    assert config.report_path == "experiments/custom/report.json"
    assert config.per_prompt_csv_path == "experiments/custom/per_prompt.csv"
    assert config.spacy_model == "en_core_web_md"
