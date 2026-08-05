# DeepStream install on this Jetson

## Recommendation: version

| Component | This machine | DeepStream to install |
|-----------|--------------|------------------------|
| Board | Jetson Orin Nano Super (8 GB) | |
| JetPack | 6.2.2 | |
| L4T | 36.5.0 | |
| CUDA / TensorRT | 12.6 / 10.3 | |
| **DeepStream** | (was missing) | **7.1.0** (`deepstream-7.1_7.1.0-1_arm64.deb`) |

DeepStream **7.1** is the matching SDK for the JetPack 6.1 / 6.2 line. It is
officially validated on JetPack 6.1; on JetPack 6.2 it is widely used with a
known `nvvideoconvert` caveat (see below). DeepStream 8.x / 9.x target newer
JetPack / OS baselines and are **not** the right choice for L4T 36.5.

## One-command install

```bash
cd /home/cvincent/Projects/01-Puppet/eyes
bash scripts/install_deepstream.sh
```

The script will:

1. Detect Jetson / L4T
2. Download `deepstream-7.1_7.1.0-1_arm64.deb` (~602 MiB) from NVIDIA NGC into
   `third_party/` (guest download, no NGC API key)
3. Install GStreamer / SSL / YAML prerequisites via `apt`
4. Install the `.deb` with `apt-get install ./…`
5. Run `ldconfig` and print `deepstream-app` / `nvinfer` verification

Requires `sudo` for the apt/dpkg steps. Download-only (no sudo):

```bash
bash scripts/install_deepstream.sh --download-only
sudo bash scripts/install_deepstream.sh --install-only
```

## Manual steps (same as the script)

```bash
# 1) Prerequisites
sudo apt-get update
sudo apt-get install -y \
  libssl3 libssl-dev libgstreamer1.0-0 gstreamer1.0-tools \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 \
  libjansson4 libyaml-cpp-dev python3-gi python3-gst-1.0

# 2) Download Jetson Debian package
mkdir -p third_party && cd third_party
curl -L --fail -o deepstream-7.1_7.1.0-1_arm64.deb \
  "https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/7.1/files/deepstream-7.1_7.1.0-1_arm64.deb"

# 3) Install
sudo apt-get install -y ./deepstream-7.1_7.1.0-1_arm64.deb
sudo ldconfig

# 4) Verify
deepstream-app --version-all
gst-inspect-1.0 nvinfer | head
```

Official reference:
https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html

## JetPack 6.2 caveat (`nvvideoconvert`)

On JP 6.2 you may see `nvbufsurftransform` / mem-copy failures with
`nvvideoconvert`. NVIDIA’s FAQ workaround for DS 7.1 on JP 6.2 is to prefer
`nvvidconv` (L4T plugin) in custom graphs, or apply the documented
compatibility patch from the DeepStream JP 6.2 FAQ. Our project configs under
`deepstream/` should use `nvvidconv` where possible.

## Python bindings (`pyds`)

After the SDK is installed:

```bash
cd /opt/nvidia/deepstream/deepstream/lib
python3 -m pip install --user ./pyds-*-py3-none-linux_aarch64.whl
# or build from sources/python if the wheel name differs
python3 -c "import pyds; print('pyds ok')"
```

## After the `.deb` install (fix missing deps)

If `deepstream-app` fails with `libgstrtspserver-1.0.so.0: cannot open shared object file`, install the leftover GStreamer deps (safe to re-run):

```bash
sudo apt-get install -y libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly gstreamer1.0-libav python3-gi python3-gst-1.0
sudo ldconfig
```

Harmless `gst-inspect` warnings you can ignore unless you need those features:

- `librivermax.so.0` — Rivermax (optional, high-speed NIC)
- `libtritonserver.so` — Triton / `nvinferserver` (optional)

`nvinfer` (TensorRT) does **not** need those.

## Post-install smoke test

```bash
deepstream-app --version-all
gst-inspect-1.0 nvinfer | head

# Optional: raise clocks (Orin Nano Super: check nvpmodel -q for mode ids)
sudo nvpmodel -m 0   # or -m 2 for MAXN SUPER on some Orin Nano images
sudo jetson_clocks

cd /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app
deepstream-app -c source1_usb_dec_infer_resnet.txt
```

Project-specific configs (YOLO + MiDaS) live in `deepstream/`. The live app is:

```bash
bash scripts/run_ds_pipeline.sh
# headless smoke:
bash scripts/run_ds_pipeline.sh --fakesink
```

Graph: USB MJPG → `nvstreammux` → `nvinfer`(YOLOv8n FP16) → `nvinfer`(MiDaS-small FP16, tensor meta) → probe (distance) → `nvdsosd` → display.

Measured on this Orin Nano @ 640×480: ~**29.5 FPS** with both models (camera-limited).

YOLO bbox parser: `deepstream/libnvdsinfer_yolov8_parser.so` (Ultralytics raw
`[1,84,8400]` layout — not the EfficientNMS `[N,6]` DeepStream-Yolo expects).

```bash
cd deepstream && make
```

## Uninstall

```bash
sudo apt-get remove --purge deepstream-7.1
sudo ldconfig
```
