#!/usr/bin/env python3
"""Build INT8 (+FP16) TensorRT engine for DA-V2 Metric Indoor Base."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trt import cudart  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_bgr(path: str, size: int) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths: list[str], size: int, cache_file: Path, batch_size: int = 1):
        super().__init__()
        self.image_paths = image_paths
        self.size = size
        self.cache_file = Path(cache_file)
        self.batch_size = batch_size
        self.index = 0
        self.nbytes = batch_size * 3 * size * size * 4
        self.device_ptr = cudart.malloc(self.nbytes)
        self.batch = np.empty((batch_size, 3, size, size), dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.image_paths):
            return None
        for i in range(self.batch_size):
            self.batch[i] = preprocess_bgr(self.image_paths[self.index + i], self.size)[0]
        self.index += self.batch_size
        host = np.ascontiguousarray(self.batch)
        print(f"  calib batch {self.index}/{len(self.image_paths)}", flush=True)
        cudart.memcpy_htod(self.device_ptr, host, self.nbytes)
        return [self.device_ptr]

    def read_calibration_cache(self):
        if self.cache_file.is_file():
            print(f"Using calib cache {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)
        print(f"Wrote calib cache {self.cache_file}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, default=ROOT / "models" / "dav2_metric_indoor_base_518.onnx")
    p.add_argument(
        "--engine",
        type=Path,
        default=ROOT / "models" / "dav2_metric_indoor_base_518_int8.engine",
    )
    p.add_argument("--calib-dir", type=Path, default=ROOT / "models" / "calib")
    p.add_argument("--size", type=int, default=518)
    p.add_argument("--workspace-mb", type=int, default=1024)
    p.add_argument("--max-calib-images", type=int, default=64)
    args = p.parse_args()

    images = sorted(glob.glob(str(args.calib_dir / "*.jpg")))
    images += sorted(glob.glob(str(args.calib_dir / "*.png")))
    images = images[: args.max_calib_images]
    if len(images) < 8:
        raise SystemExit(f"Need >=8 calib images in {args.calib_dir}, found {len(images)}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    print(f"Parsing {args.onnx} ...")
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    cache = args.engine.with_suffix(".calib.cache")
    calibrator = ImageCalibrator(images, size=args.size, cache_file=cache)
    if hasattr(config, "set_int8_calibrator"):
        config.set_int8_calibrator(calibrator)
    else:
        config.int8_calibrator = calibrator

    print(f"Building INT8 engine with {len(images)} images (workspace={args.workspace_mb}MB)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Engine build failed", file=sys.stderr)
        return 1
    args.engine.write_bytes(serialized)
    print(f"Wrote {args.engine} ({args.engine.stat().st_size / 1e6:.1f} MB)")
    cudart.free(calibrator.device_ptr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
