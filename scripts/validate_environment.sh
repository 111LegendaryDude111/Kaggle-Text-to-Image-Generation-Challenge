#!/usr/bin/env bash
set -euo pipefail

python3 scripts/gpu_compatibility_test.py
python3 scripts/verify_dreamlayer.py
python3 scripts/yolo_inference_test.py --weights models/yolo/yolov8n.pt
