#!/usr/bin/env bash
# Download YOLOv8n + MiDaS-small ONNX weights used by this project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"
cd "$MODELS"

if [[ ! -f yolov8n.onnx ]]; then
  echo "Downloading YOLOv8n ONNX..."
  curl -L --fail -o YOLOv8n.zip \
    "https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8n.zip"
  unzip -o YOLOv8n.zip
  cp YOLOv8n/ONNX/yolov8n.onnx ./yolov8n.onnx
  rm -rf YOLOv8n YOLOv8n.zip
fi

if [[ ! -f midas_small.onnx ]]; then
  echo "Downloading MiDaS-small ONNX..."
  curl -L --fail -o midas_small.onnx \
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
fi

ls -lh yolov8n.onnx midas_small.onnx
echo "Next: bash scripts/build_engines.sh"
