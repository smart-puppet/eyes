#!/usr/bin/env python3
"""Capture frames from the Microdia webcam and display them with OpenCV."""

from __future__ import annotations

import argparse
import sys

import cv2

from camera import DEFAULT_PRODUCT_ID, DEFAULT_VENDOR_ID, find_video_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live frames from the Microdia Webcam Vitade AF (0c45:6366)."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="V4L2 device path (default: auto-detect USB 0c45:6366)",
    )
    parser.add_argument("--vendor", default=DEFAULT_VENDOR_ID, help="USB vendor id")
    parser.add_argument("--product", default=DEFAULT_PRODUCT_ID, help="USB product id")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        device = args.device or find_video_device(args.vendor, args.product)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Opening {device} ({args.vendor}:{args.product})")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"error: failed to open {device}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Streaming {actual_w}x{actual_h} @ {actual_fps:.1f} FPS — press q to quit")

    window = "Webcam"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("error: failed to read frame", file=sys.stderr)
                break

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
