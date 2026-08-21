"""Fuse semantic floor + metric depth + YOLO into traversability scene JSON.

Floor / BEV stay on NumPy at DA-V2 native 518×518 (CuPy H2D is a net loss).
CuPy is used only when the depth grid is at least 400k cells. Public helpers
still take and return NumPy arrays.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from trt.yolo import Detection

logger = logging.getLogger(__name__)

try:
  import cupy as cp

  _CP = cp
except Exception:  # pragma: no cover - optional GPU accel
  _CP = None

_LOGGED_BACKEND = False
_USE_CUPY = False

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


def using_cupy() -> bool:
  return _CP is not None and _USE_CUPY


def _xp():
  return _CP if _USE_CUPY and _CP is not None else np


def _begin_xp(shape: Any) -> None:
  """CuPy only pays off on large maps; 518×518 stays on CPU NumPy."""
  global _USE_CUPY
  try:
    n = int(shape[0]) * int(shape[1])
  except (TypeError, IndexError, ValueError):
    n = 0
  _USE_CUPY = _CP is not None and n >= 400_000


def _as_xp(arr: Any, dtype: Any = None):
  xp = _xp()
  if dtype is None:
    return xp.asarray(arr)
  return xp.asarray(arr, dtype=dtype)


def _to_np(arr: Any) -> np.ndarray:
  if _CP is not None and isinstance(arr, _CP.ndarray):
    return _CP.asnumpy(arr)
  return np.asarray(arr)


def _any(arr: Any) -> bool:
  got = arr.any()
  if isinstance(got, (np.ndarray,)) or (_CP is not None and isinstance(got, _CP.ndarray)):
    return bool(got.item()) if getattr(got, "shape", ()) == () else bool(got)
  return bool(got)


def _scalar(arr: Any) -> float:
  if _CP is not None and isinstance(arr, _CP.ndarray):
    return float(_CP.asnumpy(arr).reshape(-1)[0])
  if isinstance(arr, np.ndarray):
    return float(arr.reshape(-1)[0])
  return float(arr)


def _log_backend_once() -> None:
  global _LOGGED_BACKEND
  if _LOGGED_BACKEND:
    return
  _LOGGED_BACKEND = True
  if _CP is not None:
    logger.info("traversability CuPy available (NumPy below 400k cells, CuPy above)")
  else:
    logger.info("traversability using NumPy (install cupy-cuda12x for large-grid GPU floor/BEV)")


def warmup_traversability(hw: Tuple[int, int] = (518, 518)) -> None:
  """Prime floor/BEV (and CuPy kernels when the grid is large) before MQTT capture."""
  h, w = hw
  dummy = np.full((h, w), 1.8, dtype=np.float32)
  dummy[int(h * 0.6) :, :] = 2.2
  t0 = time.perf_counter()
  depth_floor_mask(dummy)
  build_scene(depth_m=dummy, detections=[], frame_hw=(h, w))
  logger.info(
    "traversability warmup %.0fms cupy=%s size=%sx%s",
    (time.perf_counter() - t0) * 1000.0,
    using_cupy(),
    w,
    h,
  )


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


def _band_percentile(
  depth: Any,
  y0f: float,
  y1f: float,
  x0f: float,
  x1f: float,
  *,
  d_min: float,
  d_max: float,
  pct: float,
) -> float:
  xp = _xp()
  h, w = depth.shape[:2]
  y0, y1 = int(h * y0f), max(int(h * y0f) + 1, int(h * y1f))
  x0, x1 = int(w * x0f), max(int(w * x0f) + 1, int(w * x1f))
  roi = depth[y0:y1, x0:x1]
  valid = xp.isfinite(roi) & (roi > d_min) & (roi < d_max)
  if not _any(valid):
    return float("nan")
  return _scalar(xp.percentile(roi[valid], pct))


def wall_ahead_m(
  depth: np.ndarray,
  *,
  d_min: float = 0.25,
  d_max: float = 4.0,
) -> float:
  """Distance to a close vertical face filling the camera, or NaN.

  Floor gets farther toward the top of the image. A dresser / Kallax / wall
  stays about the same distance in the middle and the bottom.
  """
  _begin_xp(np.shape(depth))
  z = _as_xp(depth, dtype=_xp().float32)
  if z.ndim != 2 or z.size == 0:
    return float("nan")
  lower = _band_percentile(z, 0.68, 0.95, 0.22, 0.78, d_min=d_min, d_max=d_max, pct=20.0)
  mid = _band_percentile(z, 0.36, 0.60, 0.22, 0.78, d_min=d_min, d_max=d_max, pct=20.0)
  if not (np.isfinite(lower) and np.isfinite(mid)):
    return float("nan")
  if mid < 1.15 and lower < 1.15 and abs(mid - lower) < 0.28:
    return float(min(mid, lower))
  return float("nan")


def _vertical_face_mask_xp(z: Any) -> Any:
  xp = _xp()
  if z.ndim != 2 or z.size == 0:
    return xp.zeros_like(z, dtype=bool)
  h, w = z.shape[:2]
  y0 = int(h * 0.32)
  band = z[y0:, :]
  finite = xp.isfinite(band)
  if not _any(finite):
    return xp.zeros((h, w), dtype=bool)
  filled = xp.where(finite, band, xp.nan)
  ctx = np.errstate(all="ignore") if xp is np else contextlib.nullcontext()
  with ctx:
    col_std = xp.nanstd(filled, axis=0)
    col_med = xp.nanmedian(filled, axis=0)
  wall_cols = (
    xp.isfinite(col_std)
    & xp.isfinite(col_med)
    & (col_std < 0.20)
    & (col_med > 0.28)
    & (col_med < 1.35)
  )
  mask = xp.zeros((h, w), dtype=bool)
  mask[int(h * 0.22) :, wall_cols] = True
  return mask


def vertical_face_mask(depth: np.ndarray) -> np.ndarray:
  """Columns whose depth barely changes down the frame are furniture or walls."""
  _begin_xp(np.shape(depth))
  z = _as_xp(depth, dtype=_xp().float32)
  return _to_np(_vertical_face_mask_xp(z))


def _depth_floor_mask_xp(
  z: Any,
  *,
  camera_h_m: float = 0.12,
  band_m: float = 0.10,
  fy: Optional[float] = None,
) -> Any:
  xp = _xp()
  if z.ndim != 2 or z.size == 0:
    return xp.zeros_like(z, dtype=bool)
  h, w = z.shape[:2]
  if fy is None:
    fy = 0.9 * float(max(w, 1))
  cy = (h - 1) * 0.45
  rows = xp.arange(h, dtype=xp.float32)[:, None]
  valid = xp.isfinite(z) & (z > 0.22) & (z < 4.0)
  y_cam = (rows - cy) * z / max(float(fy), 1e-3)
  plane = valid & (z >= 1.35) & (xp.abs(y_cam - camera_h_m) <= band_m)
  shift = max(8, h // 8)
  z_up = xp.full_like(z, xp.nan)
  z_up[shift:, :] = z[:-shift, :]
  rising = (
    valid
    & (rows >= (h * 0.42))
    & xp.isfinite(z_up)
    & ((z_up - z) > 0.10)
    & (z_up > z * 1.15)
  )
  return (plane | rising) & ~_vertical_face_mask_xp(z)


def depth_floor_mask(
  depth: np.ndarray,
  *,
  camera_h_m: float = 0.12,
  band_m: float = 0.10,
  fy: Optional[float] = None,
) -> np.ndarray:
  """Mark pixels whose pinhole height matches the floor.

  Indoor wood and tile are a metric-depth ground prior — no extra TensorRT
  net. Close vertical furniture must stay blocked.
  """
  _begin_xp(np.shape(depth))
  _log_backend_once()
  z = _as_xp(depth, dtype=_xp().float32)
  return _to_np(_depth_floor_mask_xp(z, camera_h_m=camera_h_m, band_m=band_m, fy=fy))


def _sector_ranges(
  depth: Any,
  free: Any,
  *,
  d_min: float,
  d_max: float,
) -> Dict[str, float]:
  xp = _xp()
  h, w = depth.shape[:2]
  bands = {
    "left": (0, w // 3),
    "center": (w // 3, 2 * w // 3),
    "right": (2 * w // 3, w),
  }
  y0 = int(h * 0.45)
  out: Dict[str, float] = {}
  for name, (x0, x1) in bands.items():
    roi_d = depth[y0:h, x0:x1]
    roi_f = free[y0:h, x0:x1]
    valid = xp.isfinite(roi_d) & (roi_d > d_min) & (roi_d < d_max)
    obstacle = valid & ~roi_f
    if _any(obstacle):
      out[name] = _scalar(xp.percentile(roi_d[obstacle], 10.0))
    else:
      out[name] = float(d_max)
  return out


def _clean_floor_mask_xp(floor: Any) -> Any:
  host = _to_np(floor).astype(np.uint8)
  if host.ndim != 2 or host.size == 0:
    return _as_xp(np.asarray(floor, dtype=bool), dtype=bool)
  h, _w = host.shape[:2]
  y0 = int(h * 0.32)
  close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
  open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  ground = cv2.morphologyEx(host[y0:], cv2.MORPH_CLOSE, close_k)
  ground = cv2.morphologyEx(ground, cv2.MORPH_OPEN, open_k)
  host = host.copy()
  host[y0:] = ground
  host = cv2.morphologyEx(host, cv2.MORPH_CLOSE, open_k)
  return _as_xp(host.astype(bool), dtype=bool)


def clean_floor_mask(floor: np.ndarray) -> np.ndarray:
  """Fill speckles/holes on the ground. OpenCV kernels, CPU."""
  return _to_np(_clean_floor_mask_xp(floor))


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
  mask_hw: Tuple[int, int],
  box_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
  """Block the floor *contact* of obstacles, not the whole tall bbox.

  A standing person / mislabeled fridge otherwise paints a vertical rectangle
  of no-go across good floor in front of the camera. Boxes may be in frame
  pixels while the mask stays at native depth size.
  """
  h, w = mask_hw
  bh, bw = box_hw if box_hw is not None else (h, w)
  sx = w / max(float(bw), 1.0)
  sy = h / max(float(bh), 1.0)
  out = free.copy()
  for det in detections:
    if det.label not in NO_GO_LABELS:
      continue
    x1 = int(np.clip(det.x1 * sx, 0, w - 1))
    y1 = int(np.clip(det.y1 * sy, 0, h - 1))
    x2 = int(np.clip(det.x2 * sx, 0, w - 1))
    y2 = int(np.clip(det.y2 * sy, 0, h - 1))
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
  depth: Any,
  free: Any,
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
  xp = _xp()
  h, w = depth.shape[:2]
  if cx is None:
    cx = (w - 1) * 0.5
  if cy is None:
    cy = (h - 1) * 0.5
  gw = int(round(width_m / res_m))
  gh = int(round(depth_m / res_m))
  bev = xp.full((gh, gw), 255, dtype=xp.uint8)

  y0 = int(h * 0.4)
  ys, xs = xp.mgrid[y0:h, 0:w]
  z = depth[y0:h, 0:w]
  valid = xp.isfinite(z) & (z > 0.2) & (z < depth_m)
  if not _any(valid):
    return _to_np(bev)

  xs_v = xs[valid].astype(xp.float32)
  z_v = z[valid].astype(xp.float32)
  free_v = free[y0:h, 0:w][valid]

  x_m = (xs_v - cx) * z_v / fx
  forward = z_v

  col = ((x_m + width_m * 0.5) / res_m).astype(xp.int32)
  row = (gh - 1 - (forward / res_m).astype(xp.int32))
  inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
  col, row, free_v = col[inside], row[inside], free_v[inside]

  occ = ~free_v
  bev[row[free_v], col[free_v]] = 0
  bev[row[occ], col[occ]] = 100
  return _to_np(bev)


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
  detections: Sequence[Detection],
  frame_hw: Tuple[int, int],
  d_min: float = 0.25,
  d_max: float = 4.0,
  closest_m: float = float("nan"),
  floor_mask: Optional[np.ndarray] = None,
) -> SceneSummary:
  _begin_xp(np.shape(depth_m))
  _log_backend_once()
  xp = _xp()
  _ = closest_m
  fh, fw = frame_hw
  z = _as_xp(depth_m, dtype=xp.float32)
  dh, dw = int(z.shape[0]), int(z.shape[1])
  if floor_mask is not None:
    geo = _as_xp(floor_mask, dtype=bool)
  else:
    geo = _depth_floor_mask_xp(z)
  floor = _clean_floor_mask_xp(geo) & ~_vertical_face_mask_xp(z)
  free_xp = floor & xp.isfinite(z) & (z >= d_min) & (z <= d_max)
  free = _apply_yolo_nogo(_to_np(free_xp), detections, (dh, dw), box_hw=(fh, fw))
  free_xp = _as_xp(free, dtype=bool)

  sectors = _sector_ranges(z, free_xp, d_min=d_min, d_max=d_max)
  y0 = int(dh * 0.45)
  ahead = free_xp[y0:dh, :]
  floor_ahead_pct = float(_scalar(ahead.astype(xp.float32).mean())) if ahead.size else 0.0

  obstacle = xp.isfinite(z) & (z >= d_min) & (z <= d_max) & ~free_xp
  occ = obstacle[y0:dh, :]
  if _any(occ):
    closest_m = _scalar(xp.percentile(z[y0:dh][occ], 5.0))
  else:
    closest_m = float(d_max)
  wall_m = wall_ahead_m(z, d_min=d_min, d_max=d_max)
  if np.isfinite(wall_m) and wall_m < closest_m:
    closest_m = float(wall_m)
    center = sectors.get("center", float("nan"))
    if not np.isfinite(center) or wall_m < center:
      sectors["center"] = float(wall_m)

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
        "bearing": _bearing(cx, fw),
      }
    )
  objects.sort(key=lambda o: (o["dist_m"] is None, o["dist_m"] or 99.0))

  focal = 0.9 * float(max(dw, 1))
  bev = _bev_from_depth_free(z, free_xp, fx=focal, fy=focal)
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
    "wall_ahead_m": None if not np.isfinite(wall_m) else round(float(wall_m), 2),
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
  fh, fw = vis.shape[:2]
  mask = np.asarray(free_mask, dtype=bool)
  if mask.shape[:2] != (fh, fw):
    mask = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST).astype(bool)
  tint = np.zeros_like(vis)
  tint[mask] = (40, 180, 60)
  vis = cv2.addWeighted(vis, 0.75, tint, 0.25, 0)

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
