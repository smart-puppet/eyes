#!/usr/bin/env python3
"""Low-latency YOLO + MiDaS-small TensorRT pipeline for the Microdia webcam.

DeepStream SDK is not installed on this Jetson (JetPack 6.2 / L4T 36.5). This
app uses the same TensorRT engines DeepStream ``nvinfer`` would load, with a
latency-oriented capture → detect → depth → overlay loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera import find_video_device, DEFAULT_PRODUCT_ID, DEFAULT_VENDOR_ID
from trt import (
    TrtEngine,
    attach_distances,
    depth_to_colormap,
    postprocess_yolo,
    preprocess_midas,
    preprocess_yolo,
    resize_depth_to_frame,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO = ROOT / "models" / "yolo26n_fp16.engine"
DEFAULT_MIDAS = ROOT / "models" / "midas_small_fp16.engine"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="V4L2 device (default: auto-detect)")
    p.add_argument("--vendor", default=DEFAULT_VENDOR_ID)
    p.add_argument("--product", default=DEFAULT_PRODUCT_ID)
    p.add_argument("--width", type=int, default=640, help="Capture width (lower = less latency)")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    p.add_argument("--midas", type=Path, default=DEFAULT_MIDAS)
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--depth-scale",
        type=float,
        default=800.0,
        help="meters ≈ depth_scale / MiDaS_relative (calibrate on a known object)",
    )
    p.add_argument(
        "--depth-every",
        type=int,
        default=1,
        help="Run MiDaS every N frames (1 = every frame; >1 reduces latency/load)",
    )
    p.add_argument("--show-depth", action="store_true", help="Show depth colormap side panel")
    p.add_argument("--max-det", type=int, default=30)
    return p.parse_args()


def draw_detections(frame: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        if det.distance_m is not None and np.isfinite(det.distance_m):
            text = f"{det.label} {det.conf:.2f} {det.distance_m:.2f}m"
        else:
            text = f"{det.label} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 220, 80), -1)
        cv2.putText(
            frame,
            text,
            (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    for path in (args.yolo, args.midas):
        if not path.is_file():
            print(
                f"error: missing engine {path}\n"
                f"Run: bash scripts/build_engines.sh",
                file=sys.stderr,
            )
            return 1

    try:
        device = args.device or find_video_device(args.vendor, args.product)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Camera {device}")
    print(f"YOLO   {args.yolo}")
    print(f"MiDaS  {args.midas}")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"error: failed to open {device}", file=sys.stderr)
        return 1

    # Minimize capture queueing for lower end-to-end latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    yolo = TrtEngine(args.yolo)
    midas = TrtEngine(args.midas)
    yolo_h, yolo_w = yolo.input_shape[2], yolo.input_shape[3]
    midas_h = midas.input_shape[2]

    window = "YOLO + MiDaS"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_depth = None
    frame_i = 0
    fps_t0 = time.perf_counter()
    fps_n = 0
    fps = 0.0
    lat_ms = 0.0

    print("Running — press q to quit")
    try:
        while True:
            # Grab+retrieve separately so we discard stale buffered frames.
            if not cap.grab():
                print("error: grab failed", file=sys.stderr)
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                print("error: retrieve failed", file=sys.stderr)
                break

            t0 = time.perf_counter()
            yolo_in, info = preprocess_yolo(frame, size=yolo_w)
            yolo_out = yolo.infer(yolo_in)
            dets = postprocess_yolo(
                next(iter(yolo_out.values())),
                info,
                conf_thres=args.conf,
                iou_thres=args.iou,
                max_det=args.max_det,
            )

            if frame_i % max(1, args.depth_every) == 0:
                midas_in = preprocess_midas(frame, size=midas_h)
                midas_out = midas.infer(midas_in)
                depth_small = next(iter(midas_out.values()))
                last_depth = resize_depth_to_frame(depth_small, frame.shape[:2])

            if last_depth is not None:
                attach_distances(dets, last_depth, args.depth_scale)

            draw_detections(frame, dets)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            fps_n += 1
            if time.perf_counter() - fps_t0 >= 1.0:
                fps = fps_n / (time.perf_counter() - fps_t0)
                fps_n = 0
                fps_t0 = time.perf_counter()

            cv2.putText(
                frame,
                f"{fps:.1f} FPS  {lat_ms:.1f} ms  dets={len(dets)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )

            if args.show_depth and last_depth is not None:
                depth_bgr = depth_to_colormap(last_depth)
                vis = np.hstack([frame, depth_bgr])
            else:
                vis = frame

            cv2.imshow(window, vis)
            frame_i += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        yolo.close()
        midas.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
