# How distances are computed

## Default: Depth Anything V2 Metric Indoor Base (INT8)

The DeepStream pipeline defaults to **DA-V2 Metric Indoor Base**, quantized to
INT8 (`models/dav2_metric_indoor_base_518_int8.engine`).

- Trained for **indoor metric depth** (Hypersim), range roughly **0–20 m**.
- Tensor values are **metres** already (nearer = smaller).
- No `depth_scale` / `--calib-closest-m` needed for the default path.
- OSD shows `Xm` (not `~Xm`).

Closest obstacle = low-percentile sample over the **native depth grid**
(DA-V2 is 518×518). Floor / BEV stay at that size; the Eye overlay only
nearest-neighbor upscales the free mask onto the camera frame. Per-box
distance = 25th percentile in the box centre crop, with YOLO boxes scaled
into the depth grid.

Floor / BEV stay on **NumPy** at the native 518×518 grid (CuPy copies were
a net loss at that size). CuPy is used only if the depth grid is at least
400k cells. Morphology stays on CPU OpenCV.

```bash
python3 -m pip install --user 'cupy-cuda12x>=13,<14'
```

A log line `traversability CuPy available …` appears on the first capture.

```bash
bash scripts/run_ds_pipeline.sh
# headless smoke:
bash scripts/run_ds_pipeline.sh --fakesink
```

### Optional: Small INT8 (faster, less detail)

```bash
bash scripts/run_ds_pipeline.sh --depth-config config_infer_dav2_metric.txt
```

Or Eye: `--depth models/dav2_metric_indoor_small_518_int8.engine --depth-config deepstream/config_infer_dav2_metric.txt`

## Fallback: MiDaS-small (relative)

MiDaS does **not** output metres. Larger ≈ nearer. Absolute values need a
single global scale:

```text
meters ≈ depth_scale / relative_depth
```

```bash
bash scripts/run_ds_pipeline.sh \
  --midas-config config_infer_midas.txt \
  --relative-depth \
  --calib-closest-m 0.40
```

## Rebuild DA-V2 INT8 (optional)

```bash
# ONNX export (needs .venv-export with torch)
.venv-export/bin/python scripts/export_dav2_metric_onnx.py \
  --encoder vitb \
  --weights third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth \
  --output models/dav2_metric_indoor_base_518.onnx

# INT8 engine (system TensorRT)
/usr/bin/python3 scripts/build_dav2_int8_engine.py \
  --onnx models/dav2_metric_indoor_base_518.onnx \
  --engine models/dav2_metric_indoor_base_518_int8.engine
```
