#!/usr/bin/env bash
# Install NVIDIA DeepStream 7.1 on Jetson Orin (JetPack 6.1 / 6.2, L4T 36.4–36.5).
#
# Officially validated on JetPack 6.1. On JetPack 6.2 it is commonly used; see
# docs/DEEPSTREAM_INSTALL.md for known JP 6.2 workarounds (nvvideoconvert).
#
# Usage:
#   bash scripts/install_deepstream.sh            # download (if needed) + install
#   bash scripts/install_deepstream.sh --download-only
#   bash scripts/install_deepstream.sh --install-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
DEB_NAME="deepstream-7.1_7.1.0-1_arm64.deb"
DEB_PATH="$THIRD_PARTY/$DEB_NAME"
# Guest-downloadable NGC API URL (no NGC API key required for this public resource).
NGC_URL="https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/7.1/files/${DEB_NAME}"

DOWNLOAD_ONLY=0
INSTALL_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --download-only) DOWNLOAD_ONLY=1 ;;
    --install-only) INSTALL_ONLY=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

need_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

check_platform() {
  if [[ ! -f /etc/nv_tegra_release ]]; then
    echo "error: this installer targets Jetson (/etc/nv_tegra_release missing)" >&2
    exit 1
  fi
  echo "=== Platform ==="
  head -1 /etc/nv_tegra_release || true
  dpkg -l nvidia-l4t-core 2>/dev/null | awk '/^ii/ {print $2, $3}' || true
  dpkg -l nvidia-jetpack 2>/dev/null | awk '/^ii/ {print $2, $3}' || true
  uname -m
}

download_deb() {
  mkdir -p "$THIRD_PARTY"
  if [[ -f "$DEB_PATH" ]]; then
    local size
    size=$(stat -c%s "$DEB_PATH" 2>/dev/null || echo 0)
    # Package is ~602 MiB; treat tiny files as failed/partial downloads.
    if (( size > 500000000 )); then
      echo "Using existing package: $DEB_PATH ($(numfmt --to=iec "$size"))"
      return 0
    fi
    echo "Removing incomplete download ($size bytes)"
    rm -f "$DEB_PATH"
  fi
  echo "=== Downloading $DEB_NAME (~602 MiB) ==="
  curl -L --fail --retry 3 --retry-delay 2 -o "$DEB_PATH.partial" "$NGC_URL"
  mv "$DEB_PATH.partial" "$DEB_PATH"
  ls -lh "$DEB_PATH"
  file "$DEB_PATH"
}

install_deps() {
  echo "=== Installing DeepStream prerequisites ==="
  need_sudo apt-get update
  need_sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libssl3 \
    libssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libyaml-cpp-dev \
    python3-gi \
    python3-gst-1.0 \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0
}

install_deb() {
  if [[ ! -f "$DEB_PATH" ]]; then
    echo "error: missing $DEB_PATH — run without --install-only first" >&2
    exit 1
  fi
  echo "=== Installing $DEB_NAME ==="
  # apt-get install ./file.deb resolves dependencies better than dpkg -i alone.
  need_sudo apt-get install -y "$DEB_PATH" || {
    echo "Fixing broken deps then retrying..."
    need_sudo apt-get -f install -y
    need_sudo dpkg -i "$DEB_PATH"
  }
  need_sudo ldconfig

  # DeepStream ships an optional rtpjitterbuffer patch helper.
  if [[ -x /opt/nvidia/deepstream/deepstream/update_rtpmanager.sh ]]; then
    echo "=== Applying rtpjitterbuffer update script (optional) ==="
    need_sudo /opt/nvidia/deepstream/deepstream/update_rtpmanager.sh || true
  fi
}

verify_install() {
  echo "=== Verify ==="
  if command -v deepstream-app >/dev/null 2>&1; then
    deepstream-app --version-all || deepstream-app --help | head -5
  else
    # Some installs put binaries under /opt and rely on PATH via /etc/profile.d
    export PATH="/opt/nvidia/deepstream/deepstream/bin:$PATH"
    if command -v deepstream-app >/dev/null 2>&1; then
      deepstream-app --version-all || true
    else
      echo "warning: deepstream-app not on PATH; check /opt/nvidia/deepstream/" >&2
      ls -la /opt/nvidia/deepstream/ || true
    fi
  fi
  gst-inspect-1.0 nvinfer 2>&1 | head -8 || true
  echo
  echo "DeepStream root: /opt/nvidia/deepstream/deepstream-7.1 (or deepstream symlink)"
  echo "Sample configs:  /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
  echo "Python bindings: see docs/DEEPSTREAM_INSTALL.md (pyds)"
}

main() {
  check_platform
  if [[ "$INSTALL_ONLY" -eq 0 ]]; then
    download_deb
  fi
  if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    echo "Download complete. Re-run without --download-only (with sudo) to install."
    exit 0
  fi
  install_deps
  install_deb
  verify_install
  echo "Done."
}

main "$@"
