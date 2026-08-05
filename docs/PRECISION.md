# Precision notes (FP16 vs INT8)

## Short answer

**Stay on FP16 as the default.** INT8 is optional for YOLO only if you need
more headroom after measuring; it is **not** a clear win for MiDaS-small here.

## Why FP16 is the sweet spot on Orin Nano

| | FP16 (current) | INT8 |
|--|----------------|------|
| YOLO GPU time (measured) | ~5.5 ms | often ~3–4 ms after good calib |
| MiDaS-small GPU time | ~6.4 ms | smaller gain; depth quality risks |
| Accuracy | near-FP32 for these nets | YOLO usually fine; depth often noisier |
| Build / ops cost | one `trtexec --fp16` | needs calibration images + rebuild |
| Memory | higher | lower (helps on 8 GB) |

End-to-end latency today is ~**45 ms**, dominated by **CPU preprocess/postprocess
and H2D/D2H**, not by the ~12 ms of combined GPU compute. Cutting YOLO from
5.5 → ~3.5 ms saves ~2 ms — useful, but not transformative until the CPU path
is tightened (or moved into DeepStream/`nvvideoconvert`/`nvstreammux`).

Orin has INT8 Tensor Cores, so INT8 is *capable*. For this dual-network app:

1. **YOLO** — INT8 can make sense later (detection is quantization-tolerant if
   you calibrate on representative webcam frames). Expect a few points of mAP
   loss; verify on your scene.
2. **MiDaS-small** — prefer **FP16**. Monocular depth is sensitive to
   quantization; bad calib shows up as unstable distances. Keep FP16 unless
   you prove INT8 depth error is acceptable.

Recommended production mix if you chase more speed later:

- YOLO → INT8 (calibrated)
- MiDaS → FP16
- Or skip MiDaS some frames (`--depth-every N`) before touching INT8

## If you still want INT8 for YOLO

```bash
# Collect ~200–500 representative frames first, then:
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolov8n.onnx \
  --saveEngine=models/yolov8n_int8.engine \
  --int8 --fp16 \
  --calib=<calibration_cache_or_use_builder_with_images> \
  --memPoolSize=workspace:512M
```

DeepStream `nvinfer` `network-mode=1` is INT8; `network-mode=2` is FP16
(our configs use FP16).
