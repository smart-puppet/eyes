"""Cityscapes Fast-SCNN helpers for apartment floor / wall remap."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# Cityscapes trainId palette (19 classes) — Fast-SCNN PINTO outputs class indices.
# Indoor wood/tile often lands on road, sidewalk, or terrain — not a second net.
CITYSCAPES_FLOOR = {0, 1, 9}  # road, sidewalk, terrain
CITYSCAPES_WALL = {2, 3, 4}  # building, wall, fence

# ImageNet-ish / Cityscapes common preprocess for PINTO Fast-SCNN demos.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_fast_scnn(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) * (1.0 / 255.0)
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x.astype(np.float32))


def logits_to_labels(output: np.ndarray) -> np.ndarray:
    """Accept [1,1,H,W] class ids, [1,C,H,W] logits, or [H,W] → int32 HxW labels."""
    arr = np.asarray(output)
    if arr.ndim == 4:
        # PINTO Fast-SCNN engines often emit already-argmaxed [1,1,H,W].
        if arr.shape[1] == 1:
            arr = arr[0, 0]
        else:
            arr = arr.argmax(axis=1)[0]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] < arr.shape[1]:
            arr = arr.argmax(axis=0)
        else:
            arr = arr[0]
    return arr.astype(np.int32, copy=False)


def remap_apartment(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (floor_mask, wall_mask, other_mask) bool HxW."""
    floor = np.isin(labels, list(CITYSCAPES_FLOOR))
    wall = np.isin(labels, list(CITYSCAPES_WALL))
    other = ~(floor | wall)
    return floor, wall, other


def resize_mask_to_frame(mask: np.ndarray, frame_hw: Tuple[int, int]) -> np.ndarray:
    h, w = frame_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
