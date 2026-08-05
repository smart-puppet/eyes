"""Depth Anything V2 Metric helpers (metres; nearer = smaller)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .midas import preprocess_midas, resize_depth_to_frame
from .yolo import Detection

# Same ImageNet norm / NCHW layout as MiDaS; only the spatial size differs.
preprocess_dav2 = preprocess_midas


def sample_metric_depth(depth: np.ndarray, det: Detection, center_frac: float = 0.35) -> float:
    """Low-percentile metric depth in the box centre (metres)."""
    h, w = depth.shape[:2]
    bw = max(det.x2 - det.x1, 1.0)
    bh = max(det.y2 - det.y1, 1.0)
    cx = 0.5 * (det.x1 + det.x2)
    cy = 0.5 * (det.y1 + det.y2)
    rw = bw * center_frac * 0.5
    rh = bh * center_frac * 0.5
    x1 = int(max(0, cx - rw))
    y1 = int(max(0, cy - rh))
    x2 = int(min(w - 1, cx + rw))
    y2 = int(min(h - 1, cy + rh))
    patch = depth[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return float("nan")
    finite = patch[np.isfinite(patch) & (patch > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 25.0))


def attach_metric_distances(
    detections: list[Detection],
    depth_map: np.ndarray,
) -> list[Detection]:
    for det in detections:
        metres = sample_metric_depth(depth_map, det)
        det.distance_m = metres if np.isfinite(metres) else None
    return detections


def closest_scene_metric(
    depth: np.ndarray,
    border_frac: float = 0.08,
    percentile: float = 1.0,
) -> Tuple[float, Tuple[int, int] | None]:
    """Closest full-frame metric sample → (metres, xy)."""
    h, w = depth.shape[:2]
    if h < 8 or w < 8:
        return float("nan"), None
    border_frac = float(np.clip(border_frac, 0.0, 0.4))
    my = int(h * border_frac)
    mx = int(w * border_frac)
    y1, y2 = my, max(my + 1, h - my)
    x1, x2 = mx, max(mx + 1, w - mx)
    roi = depth[y1:y2, x1:x2]
    finite = np.isfinite(roi) & (roi > 1e-6)
    if not finite.any():
        return float("nan"), None
    vals = roi[finite]
    target = float(np.percentile(vals, np.clip(percentile, 0.0, 100.0)))
    mask = finite & (roi <= target)
    if not mask.any():
        mask = finite
    masked = np.where(mask, roi, np.inf)
    flat = int(np.argmin(masked))
    ly, lx = np.unravel_index(flat, roi.shape)
    metres = float(roi[ly, lx])
    return metres, (x1 + int(lx), y1 + int(ly))
