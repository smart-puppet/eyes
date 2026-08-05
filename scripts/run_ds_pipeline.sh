#!/usr/bin/env bash
# Run the DeepStream nvinfer YOLO + depth pipeline
# (default: DA-V2 Metric Indoor Small INT8).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/deepstream"

export PATH="/opt/nvidia/deepstream/deepstream/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/deepstream:${GST_PLUGIN_PATH:-}"

exec /usr/bin/python3 ds_pipeline.py "$@"
