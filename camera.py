"""Webcam capture utilities for the Microdia / Innomaker UVC camera."""

from __future__ import annotations

import os
from pathlib import Path


# Microdia Webcam Vitade AF / Innomaker-U20CAM-1080p-S1
DEFAULT_VENDOR_ID = "0c45"
DEFAULT_PRODUCT_ID = "6366"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def find_video_device(
    vendor_id: str = DEFAULT_VENDOR_ID,
    product_id: str = DEFAULT_PRODUCT_ID,
) -> str:
    """Return the /dev/videoN path for a USB camera matching vendor:product.

    Prefers the capture node (video0-style) over metadata-only siblings by
    picking the lowest-numbered matching device.
    """
    vendor_id = vendor_id.lower()
    product_id = product_id.lower()
    matches: list[tuple[int, str]] = []

    sys_v4l = Path("/sys/class/video4linux")
    if not sys_v4l.is_dir():
        raise FileNotFoundError("No V4L2 devices found under /sys/class/video4linux")

    for node in sorted(sys_v4l.iterdir()):
        if not node.name.startswith("video"):
            continue
        try:
            index = int(node.name.removeprefix("video"))
        except ValueError:
            continue

        device_link = node / "device"
        if not device_link.exists():
            continue

        # Walk up the sysfs tree to find USB idVendor / idProduct.
        cur = device_link.resolve()
        for _ in range(8):
            vid = _read_text(cur / "idVendor")
            pid = _read_text(cur / "idProduct")
            if vid is not None and pid is not None:
                if vid.lower() == vendor_id and pid.lower() == product_id:
                    dev_path = f"/dev/{node.name}"
                    if os.path.exists(dev_path):
                        matches.append((index, dev_path))
                break
            parent = cur.parent
            if parent == cur:
                break
            cur = parent

    if not matches:
        raise FileNotFoundError(
            f"No V4L2 device found for USB {vendor_id}:{product_id}"
        )

    matches.sort(key=lambda item: item[0])
    return matches[0][1]
