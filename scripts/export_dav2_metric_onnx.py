#!/usr/bin/env python3
"""Export Depth Anything V2 Metric Indoor Base (Hypersim) to ONNX for TensorRT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Depth-Anything-V2" / "metric_depth"
sys.path.insert(0, str(ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    p.add_argument("--max-depth", type=float, default=20.0, help="Indoor Hypersim max depth")
    p.add_argument("--size", type=int, default=518, help="Square input size (multiple of 14)")
    p.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "checkpoints" / "depth_anything_v2_metric_hypersim_vitb.pth",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "dav2_metric_indoor_base_518.onnx",
    )
    args = p.parse_args()

    if args.size % 14 != 0:
        raise SystemExit(f"--size must be multiple of 14, got {args.size}")
    if not args.weights.is_file():
        raise SystemExit(f"missing weights: {args.weights}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    print(f"Loading {args.weights} on CPU...")
    model = DepthAnythingV2(**{**model_configs[args.encoder], "max_depth": args.max_depth})
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX → {args.output}  input=1x3x{args.size}x{args.size}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            input_names=["input"],
            output_names=["depth"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    # Quick shape check
    import onnx

    m = onnx.load(str(args.output))
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print("IN ", i.name, dims)
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print("OUT", o.name, dims)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
