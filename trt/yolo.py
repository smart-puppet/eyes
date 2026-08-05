"""YOLOv8 detection preprocess / postprocess helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    distance_m: float | None = None

    @property
    def label(self) -> str:
        if 0 <= self.cls_id < len(COCO_CLASSES):
            return COCO_CLASSES[self.cls_id]
        return str(self.cls_id)


@dataclass
class LetterboxInfo:
    ratio: float
    pad_w: float
    pad_h: float
    orig_w: int
    orig_h: int


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = image.shape[:2]
    th, tw = new_shape
    ratio = min(th / h, tw / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (tw - nw) / 2
    pad_h = (th - nh) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, LetterboxInfo(ratio=ratio, pad_w=pad_w, pad_h=pad_h, orig_w=w, orig_h=h)


def preprocess_yolo(bgr: np.ndarray, size: int = 640) -> Tuple[np.ndarray, LetterboxInfo]:
    img, info = letterbox(bgr, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) * (1.0 / 255.0)
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), info


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess_yolo(
    output: np.ndarray,
    info: LetterboxInfo,
    conf_thres: float = 0.4,
    iou_thres: float = 0.45,
    max_det: int = 50,
) -> List[Detection]:
    """Parse Ultralytics YOLOv8 output shaped [1, 84, 8400]."""
    pred = output[0] if output.ndim == 3 else output
    # [84, N] -> [N, 84]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    cls_ids = scores.argmax(axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]
    mask = confs >= conf_thres
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
    if boxes.size == 0:
        return []

    boxes = _xywh_to_xyxy(boxes)
    # Undo letterbox
    boxes[:, [0, 2]] -= info.pad_w
    boxes[:, [1, 3]] -= info.pad_h
    boxes[:, :4] /= info.ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h - 1)

    detections: List[Detection] = []
    for cls in np.unique(cls_ids):
        idxs = np.where(cls_ids == cls)[0]
        keep = _nms(boxes[idxs], confs[idxs], iou_thres)
        for k in keep:
            i = idxs[k]
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(confs[i]),
                    cls_id=int(cls_ids[i]),
                )
            )

    detections.sort(key=lambda d: d.conf, reverse=True)
    return detections[:max_det]


def postprocess_yolo_end2end(
    output: np.ndarray,
    info: LetterboxInfo,
    conf_thres: float = 0.4,
    max_det: int = 50,
) -> List[Detection]:
    """Parse DeepStream-Yolo / YOLO26 end2end output shaped [1, N, 6] or [N, 6].

    Each row is ``x1, y1, x2, y2, score, class`` in letterboxed network pixels.
    No NMS (model already emits per-anchor max class).
    """
    pred = output
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[-1] < 6:
        raise ValueError(f"expected [N,6] end2end output, got {output.shape}")

    boxes = pred[:, :4].astype(np.float32, copy=True)
    confs = pred[:, 4].astype(np.float32, copy=False)
    cls_ids = pred[:, 5].astype(np.int32, copy=False)
    mask = confs >= conf_thres
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
    if boxes.size == 0:
        return []

    boxes[:, [0, 2]] -= info.pad_w
    boxes[:, [1, 3]] -= info.pad_h
    boxes[:, :4] /= info.ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h - 1)

    order = np.argsort(-confs)[:max_det]
    detections: List[Detection] = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(confs[i]),
                cls_id=int(cls_ids[i]),
            )
        )
    return detections
