from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav.traversability import (  # noqa: E402
    FloorSmoother,
    build_scene,
    clean_floor_mask,
    depth_floor_mask,
)


def test_clean_floor_mask_fills_small_ground_hole() -> None:
  floor = np.ones((80, 120), dtype=bool)
  floor[50:58, 55:65] = False
  cleaned = clean_floor_mask(floor)
  assert cleaned[53, 60]


def test_floor_smoother_keeps_recent_floor() -> None:
  smooth = FloorSmoother(alpha=0.55, thresh=0.38)
  full = np.ones((40, 60), dtype=bool)
  hole = full.copy()
  hole[20:30, 20:40] = False
  first = smooth.update(full, now=1.0)
  second = smooth.update(hole, now=1.1)
  assert first[25, 30]
  assert second[25, 30]


def _ground_plane(h: int = 48, w: int = 64, camera_h_m: float = 0.12) -> np.ndarray:
  fy = 0.9 * float(w)
  cy = (h - 1) * 0.45
  rows = np.arange(h, dtype=np.float32)[:, None]
  z = camera_h_m * fy / np.maximum(rows - cy, 1e-3)
  z = np.broadcast_to(z, (h, w)).copy()
  z[: int(cy) + 1] = 2.5
  return z


def test_depth_floor_mask_labels_ground_plane() -> None:
  depth = _ground_plane()
  mask = depth_floor_mask(depth)
  assert mask[-3, 32]
  assert not mask[2, 32]


def test_build_scene_treats_wood_floor_as_free_without_scnn() -> None:
  h, w = 48, 64
  depth = _ground_plane(h, w)
  empty_scnn = np.zeros((h, w), dtype=bool)
  scene = build_scene(
    depth_m=depth,
    floor_mask=empty_scnn,
    detections=[],
    frame_hw=(h, w),
    closest_m=0.96,
  )
  assert scene.payload["floor_ahead_pct"] > 0.3
  assert scene.payload["closest_m"] > 2.0
  assert scene.payload["sectors"]["center"] > 2.0
  assert scene.free_mask[-4, w // 2]


def test_build_scene_reports_wall_not_floor() -> None:
  h, w = 48, 64
  depth = np.full((h, w), 0.45, dtype=np.float32)
  empty_scnn = np.zeros((h, w), dtype=bool)
  scene = build_scene(
    depth_m=depth,
    floor_mask=empty_scnn,
    detections=[],
    frame_hw=(h, w),
  )
  assert scene.payload["closest_m"] < 0.7
  assert scene.payload["sectors"]["center"] < 0.7
