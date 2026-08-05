#!/usr/bin/env bash
# Eyes debug web: MJPEG camera / boxes+distance toggle (port 8091).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/debug_web"

export PATH="/opt/nvidia/deepstream/deepstream/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

exec /usr/bin/python3 app.py "$@"
