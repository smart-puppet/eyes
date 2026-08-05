"""MiDaS-small depth helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .yolo import Detection


# ImageNet normalization used by official MiDaS small ONNX.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_midas(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) * (1.0 / 255.0)
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x.astype(np.float32))


def resize_depth_to_frame(depth: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    h, w = frame_shape
    if depth.ndim == 3:
        depth = depth[0]
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[finite], 5), np.percentile(d[finite], 95)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm_u8, cv2.COLORMAP_MAGMA)


def sample_relative_depth(depth: np.ndarray, det: Detection, center_frac: float = 0.35) -> float:
    """Median MiDaS value in the central crop of a detection box (higher => closer)."""
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
    return float(np.median(patch))


def relative_to_meters(relative_depth: float, depth_scale: float) -> float:
    """Convert MiDaS relative inverse-depth to approximate meters.

    MiDaS outputs larger values for nearer surfaces. Tune ``depth_scale`` against a
    known object distance: meters ≈ depth_scale / relative_depth.
    """
    if not np.isfinite(relative_depth) or relative_depth <= 1e-6:
        return float("nan")
    return float(depth_scale / relative_depth)


def attach_distances(
    detections: list[Detection],
    depth_map: np.ndarray,
    depth_scale: float,
) -> list[Detection]:
    for det in detections:
        rel = sample_relative_depth(depth_map, det)
        det.distance_m = relative_to_meters(rel, depth_scale)
    return detections
