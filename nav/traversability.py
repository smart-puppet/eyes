"""Fuse semantic floor + metric depth + YOLO into traversability scene JSON."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from trt.yolo import Detection


NO_GO_LABELS = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "refrigerator",
    "potted plant",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


@dataclass
class SceneSummary:
    payload: Dict[str, Any]
    free_mask: np.ndarray  # HxW bool in frame coords
    bev: np.ndarray  # uh x uw uint8 (0 free, 100 occupied, 255 unknown)


def _bearing(cx: float, frame_w: int) -> str:
    t = cx / max(frame_w, 1)
    if t < 0.33:
        return "left"
    if t > 0.66:
        return "right"
    return "center"


def depth_floor_mask(
    depth: np.ndarray,
    *,
    camera_h_m: float = 0.12,
    band_m: float = 0.10,
    fy: Optional[float] = None,
) -> np.ndarray:
    """Mark pixels whose pinhole height matches the floor.

    Fast-SCNN is Cityscapes (road / sidewalk / terrain). Indoor wood and tile
    often miss that palette, so metric depth is the apartment ground prior —
    no extra TensorRT net.
    """
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2 or z.size == 0:
        return np.zeros_like(z, dtype=bool)
    h, w = z.shape[:2]
    if fy is None:
        fy = 0.9 * float(max(w, 1))
    cy = (h - 1) * 0.45
    rows = np.arange(h, dtype=np.float32)[:, None]
    valid = np.isfinite(z) & (z > 0.28) & (z < 4.0)
    y_cam = (rows - cy) * z / max(float(fy), 1e-3)
    # Close vertical surfaces also sit near camera height; only trust the plane farther out.
    plane = valid & (z >= 0.75) & (np.abs(y_cam - camera_h_m) <= band_m)
    # Floor: depth increases toward the top of the image. A wall stays roughly constant.
    shift = max(6, h // 12)
    z_up = np.full_like(z, np.nan)
    z_up[shift:, :] = z[:-shift, :]
    rising = (
        valid
        & (rows >= (h * 0.36))
        & np.isfinite(z_up)
        & ((z_up - z) > 0.05)
    )
    return plane | rising


def _sector_ranges(
    depth: np.ndarray,
    free: np.ndarray,
    *,
    d_min: float,
    d_max: float,
) -> Dict[str, float]:
    h, w = depth.shape[:2]
    bands = {
        "left": (0, w // 3),
        "center": (w // 3, 2 * w // 3),
        "right": (2 * w // 3, w),
    }
    # Lower 55% of frame = ground-ish FOV for ranging.
    y0 = int(h * 0.45)
    out: Dict[str, float] = {}
    for name, (x0, x1) in bands.items():
        roi_d = depth[y0:h, x0:x1]
        roi_f = free[y0:h, x0:x1]
        valid = np.isfinite(roi_d) & (roi_d > d_min) & (roi_d < d_max)
        # Distance to obstacles, not to the floor (wood at ~1 m is not a wall).
        obstacle = valid & ~roi_f
        if obstacle.any():
            out[name] = float(np.percentile(roi_d[obstacle], 10.0))
        else:
            out[name] = float(d_max)
    return out


def clean_floor_mask(floor: np.ndarray) -> np.ndarray:
    """Fill Fast-SCNN speckles/holes on the ground. CPU-only, no extra TRT."""
    mask = np.asarray(floor, dtype=np.uint8)
    if mask.ndim != 2 or mask.size == 0:
        return np.asarray(floor, dtype=bool)
    h, w = mask.shape[:2]
    y0 = int(h * 0.32)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ground = cv2.morphologyEx(mask[y0:], cv2.MORPH_CLOSE, close_k)
    ground = cv2.morphologyEx(ground, cv2.MORPH_OPEN, open_k)
    mask = mask.copy()
    mask[y0:] = ground
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, open_k)
    return mask.astype(bool)


class FloorSmoother:
    """Blend the last floor mask so one noisy capture does not punch a hole."""

    def __init__(self, alpha: float = 0.58, thresh: float = 0.38) -> None:
        self.alpha = float(alpha)
        self.thresh = float(thresh)
        self._ema: Optional[np.ndarray] = None
        self._ts = 0.0

    def reset(self) -> None:
        self._ema = None
        self._ts = 0.0

    def update(self, floor: np.ndarray, *, now: Optional[float] = None) -> np.ndarray:
        stamp = time.time() if now is None else float(now)
        cleaned = clean_floor_mask(floor).astype(np.float32)
        if self._ema is None or self._ema.shape != cleaned.shape or stamp - self._ts > 2.5:
            self._ema = cleaned
        else:
            self._ema = self.alpha * cleaned + (1.0 - self.alpha) * self._ema
        self._ts = stamp
        return self._ema >= self.thresh


def _apply_yolo_nogo(
    free: np.ndarray,
    detections: Sequence[Detection],
    frame_hw: Tuple[int, int],
) -> np.ndarray:
    """Block the floor *contact* of obstacles, not the whole tall bbox.

    A standing person / mislabeled fridge otherwise paints a vertical rectangle
    of no-go across good floor in front of the camera.
    """
    h, w = frame_hw
    out = free.copy()
    for det in detections:
        if det.label not in NO_GO_LABELS:
            continue
        x1 = int(np.clip(det.x1, 0, w - 1))
        y1 = int(np.clip(det.y1, 0, h - 1))
        x2 = int(np.clip(det.x2, 0, w - 1))
        y2 = int(np.clip(det.y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        box_h = y2 - y1
        y_contact = y1 + int(box_h * 0.5)
        pad = max(2, (x2 - x1) // 12)
        xa = max(0, x1 - pad)
        xb = min(w, x2 + pad)
        out[y_contact:y2, xa:xb] = False
    return out


def _bev_from_depth_free(
    depth: np.ndarray,
    free: np.ndarray,
    *,
    res_m: float = 0.05,
    width_m: float = 3.0,
    depth_m: float = 4.0,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """Crude pinhole projection of lower FOV into robot-centric BEV.

    Grid: rows = forward distance, cols = left/right. Origin at bottom-center.
    Values: 0=free, 100=occupied, 255=unknown.
    """
    h, w = depth.shape[:2]
    if cx is None:
        cx = (w - 1) * 0.5
    if cy is None:
        cy = (h - 1) * 0.5
    gw = int(round(width_m / res_m))
    gh = int(round(depth_m / res_m))
    bev = np.full((gh, gw), 255, dtype=np.uint8)

    y0 = int(h * 0.4)
    ys, xs = np.mgrid[y0:h, 0:w]
    z = depth[y0:h, 0:w]
    valid = np.isfinite(z) & (z > 0.2) & (z < depth_m)
    if not valid.any():
        return bev

    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    z_v = z[valid].astype(np.float32)
    free_v = free[y0:h, 0:w][valid]

    x_m = (xs_v - cx) * z_v / fx
    # Assume camera roughly horizontal; y image → ignore height for BEV occupancy.
    forward = z_v

    col = ((x_m + width_m * 0.5) / res_m).astype(np.int32)
    row = (gh - 1 - (forward / res_m).astype(np.int32))
    inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
    col, row, free_v = col[inside], row[inside], free_v[inside]

    # Occupied first, then free (free wins if both — prefer optimistic for planner later).
    occ = ~free_v
    bev[row[occ], col[occ]] = 100
    bev[row[free_v], col[free_v]] = 0
    return bev


def _encode_bev_rle(bev: np.ndarray) -> Dict[str, Any]:
    flat = bev.reshape(-1)
    runs: List[List[int]] = []
    if flat.size == 0:
        return {"encoding": "rle", "data": []}
    val = int(flat[0])
    count = 1
    for v in flat[1:]:
        iv = int(v)
        if iv == val:
            count += 1
        else:
            runs.append([val, count])
            val, count = iv, 1
    runs.append([val, count])
    return {"encoding": "rle", "data": runs}


def build_scene(
    *,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    detections: Sequence[Detection],
    frame_hw: Tuple[int, int],
    d_min: float = 0.25,
    d_max: float = 4.0,
    closest_m: float = float("nan"),
) -> SceneSummary:
    # Full-frame nearest depth is usually the floor under a low camera; ignore it.
    _ = closest_m
    h, w = frame_hw
    scnn = clean_floor_mask(floor_mask)
    geo = depth_floor_mask(depth_m)
    floor = clean_floor_mask(scnn | geo)
    free = floor & np.isfinite(depth_m) & (depth_m >= d_min) & (depth_m <= d_max)
    free = _apply_yolo_nogo(free, detections, (h, w))

    sectors = _sector_ranges(depth_m, free, d_min=d_min, d_max=d_max)
    y0 = int(h * 0.45)
    ahead = free[y0:h, :]
    floor_ahead_pct = float(ahead.mean()) if ahead.size else 0.0

    obstacle = np.isfinite(depth_m) & (depth_m >= d_min) & (depth_m <= d_max) & ~free
    occ = obstacle[y0:h, :]
    if occ.any():
        closest_m = float(np.percentile(depth_m[y0:h][occ], 5.0))
    else:
        closest_m = float(d_max)

    objects: List[Dict[str, Any]] = []
    for det in detections:
        cx = 0.5 * (det.x1 + det.x2)
        objects.append(
            {
                "label": det.label,
                "conf": round(float(det.conf), 2),
                "dist_m": None
                if det.distance_m is None or not np.isfinite(det.distance_m)
                else round(float(det.distance_m), 2),
                "bearing": _bearing(cx, w),
            }
        )
    objects.sort(key=lambda o: (o["dist_m"] is None, o["dist_m"] or 99.0))

    bev = _bev_from_depth_free(depth_m, free)
    center = sectors.get("center", float("nan"))
    left = sectors.get("left", float("nan"))
    right = sectors.get("right", float("nan"))

    def _fmt(v: float) -> str:
        return "n/a" if not np.isfinite(v) else f"{v:.1f}m"

    if np.isfinite(center) and center < 0.9:
        if np.isfinite(left) and (not np.isfinite(right) or left <= right):
            hint = f"center blocked at {_fmt(center)}; left freer ({_fmt(left)})"
        elif np.isfinite(right):
            hint = f"center blocked at {_fmt(center)}; right freer ({_fmt(right)})"
        else:
            hint = f"center blocked at {_fmt(center)}"
    elif floor_ahead_pct < 0.15:
        hint = "little free floor ahead (path may be blocked or unclear)"
    else:
        hint = f"path mostly clear; closest {_fmt(closest_m)}"

    if objects:
        top = objects[0]
        lab = str(top.get("label") or "object").replace("_", " ")
        dist = top.get("dist_m")
        br = top.get("bearing") or "center"
        if dist is not None:
            hint = f"{hint}; nearest object: {lab} at {dist}m ({br})"
        else:
            hint = f"{hint}; nearest object: {lab} ({br})"

    def _round_sec(v: float) -> Optional[float]:
        return None if not np.isfinite(v) else round(float(v), 2)

    payload: Dict[str, Any] = {
        "ts": time.time(),
        "closest_m": None if not np.isfinite(closest_m) else round(float(closest_m), 2),
        "sectors": {k: _round_sec(v) for k, v in sectors.items()},
        "floor_ahead_pct": round(floor_ahead_pct, 2),
        "objects": objects[:12],
        "costmap": {
            "res_m": 0.05,
            "w": int(bev.shape[1]),
            "h": int(bev.shape[0]),
            **_encode_bev_rle(bev),
        },
        "hint": hint,
    }
    return SceneSummary(payload=payload, free_mask=free, bev=bev)


def overlay_traversability(
    frame: np.ndarray,
    free_mask: np.ndarray,
    bev: np.ndarray,
    hint: str,
) -> np.ndarray:
    vis = frame.copy()
    tint = np.zeros_like(vis)
    tint[free_mask] = (40, 180, 60)
    vis = cv2.addWeighted(vis, 0.75, tint, 0.25, 0)

    # BEV inset top-right.
    inset_h, inset_w = 120, 160
    bev_color = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.uint8)
    bev_color[bev == 0] = (60, 200, 80)
    bev_color[bev == 100] = (40, 40, 220)
    bev_color[bev == 255] = (50, 50, 50)
    bev_big = cv2.resize(bev_color, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
    x0 = vis.shape[1] - inset_w - 10
    y0 = 10
    vis[y0 : y0 + inset_h, x0 : x0 + inset_w] = bev_big
    cv2.rectangle(vis, (x0, y0), (x0 + inset_w, y0 + inset_h), (200, 200, 200), 1)

    cv2.putText(
        vis,
        hint[:70],
        (10, vis.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 240, 255),
        1,
        cv2.LINE_AA,
    )
    return vis
