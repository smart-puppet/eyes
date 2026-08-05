#!/usr/bin/env bash
# Build FP16 TensorRT engines for YOLOv8n + MiDaS-small on this Jetson.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

YOLO_ONNX="$MODELS/yolov8n.onnx"
MIDAS_ONNX="$MODELS/midas_small.onnx"
YOLO_ENGINE="$MODELS/yolov8n_fp16.engine"
MIDAS_ENGINE="$MODELS/midas_small_fp16.engine"

# Keep workspace modest on 8GB Orin Nano.
WORKSPACE="${WORKSPACE:-512M}"

build_one() {
  local onnx="$1"
  local engine="$2"
  local name="$3"
  if [[ ! -f "$onnx" ]]; then
    echo "Missing ONNX: $onnx" >&2
    exit 1
  fi
  if [[ -f "$engine" ]]; then
    echo "Skipping $name (exists: $engine)"
    return 0
  fi
  echo "=== Building $name FP16 engine ==="
  "$TRTEXEC" \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    --fp16 \
    --memPoolSize=workspace:"$WORKSPACE" \
    --builderOptimizationLevel=3 \
    --avgRuns=10 \
    --duration=0 \
    --warmUp=100 \
    --iterations=30
  echo "Wrote $engine"
}

build_one "$YOLO_ONNX" "$YOLO_ENGINE" "YOLOv8n"
# Free builder memory between engines on 8GB devices.
sync
build_one "$MIDAS_ONNX" "$MIDAS_ENGINE" "MiDaS-small"

echo "Done."
ls -lh "$YOLO_ENGINE" "$MIDAS_ENGINE"
