#!/usr/bin/env python3
"""Eye: MJPEG camera, on-demand capture, drive pad, and live play speeds.

Video is OpenCV capture; YOLO / DA-V2 / Fast-SCNN run on Capture (HTTP or
robot/nav/capture). Drive arrows publish to robot/drive/*. Play speed sliders
write brain/config/play.speeds and robot/play/speeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, Generator, List, Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
from mic_volume import get_default_mic, set_default_mic
from play_speeds import read_play_speeds, write_play_speeds
from languages import list_language_profiles, parse_language_id
sys.path.insert(0, str(ROOT))

from camera import DEFAULT_PRODUCT_ID, DEFAULT_VENDOR_ID, find_video_device  # noqa: E402
from nav.fast_scnn import (  # noqa: E402
    logits_to_labels,
    preprocess_fast_scnn,
    remap_apartment,
    resize_mask_to_frame,
)
from nav.mqtt_scene import ScenePublisher  # noqa: E402
from nav.traversability import (  # noqa: E402
    FloorSmoother,
    build_scene,
    overlay_traversability,
)
from trt import (  # noqa: E402
    TrtEngine,
    attach_metric_distances,
    closest_scene_metric,
    postprocess_yolo,
    postprocess_yolo_end2end,
    preprocess_dav2,
    preprocess_yolo,
    resize_depth_to_frame,
)

logger = logging.getLogger("eyes")

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_YOLO = ROOT / "models" / "yolov8n_fp16.engine"
DEFAULT_DEPTH = ROOT / "models" / "dav2_metric_indoor_small_518_int8.engine"
DEFAULT_SCNN = ROOT / "models" / "fast_scnn_256x640_fp16.engine"
HB_PERIOD_S = 0.15
LOG_MAX_BRAIN = 5000
LOG_MAX_DRIVE = 500
LOG_TOPIC = "robot/log/+"
DEFAULT_LANGUAGE = "de_1"
DEFAULT_BRAIN_CONFIG = Path(__file__).resolve().parents[2] / "brain" / "config"


def _brain_config_dir() -> Path:
  raw = _cfg.get("brain_config") or os.environ.get("PUPPET_CONFIG_DIR")
  if raw:
    return Path(raw)
  return DEFAULT_BRAIN_CONFIG


def _parse_language_code(text: str) -> Optional[str]:
  first = (text or "").strip().splitlines()
  if not first:
    return None
  return parse_language_id(first[0])


def _read_saved_language(config_dir: Path) -> tuple[str, bool]:
  path = config_dir / "language.active"
  if not path.is_file():
    return DEFAULT_LANGUAGE, False
  try:
    parsed = _parse_language_code(path.read_text(encoding="utf-8"))
  except OSError:
    return DEFAULT_LANGUAGE, False
  if parsed is None:
    return DEFAULT_LANGUAGE, True
  return parsed, True


def _write_saved_language(config_dir: Path, language: str) -> Path:
  code = parse_language_id(str(language or ""))
  known = {item["id"] for item in list_language_profiles(config_dir)}
  if not code or (known and code not in known):
    raise ValueError("language must be a listed profile such as en_1, fr_1, de_1, or de_2")
  config_dir.mkdir(parents=True, exist_ok=True)
  path = config_dir / "language.active"
  tmp = path.with_name("language.active.tmp")
  tmp.write_text(code + "\n", encoding="utf-8")
  tmp.replace(path)
  return path


def _lan_ipv4s() -> List[str]:
    skip_prefixes = ("127.", "172.17.", "172.18.")
    found: List[str] = []
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            text=True,
            timeout=2,
        )
        for line in out.splitlines():
            parts = line.split()
            if "inet" not in parts:
                continue
            ip = parts[parts.index("inet") + 1].split("/")[0]
            if any(ip.startswith(p) for p in skip_prefixes):
                continue
            if ip not in found:
                found.append(ip)
    except Exception:
        pass
    if found:
        return found
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
        if primary and not any(primary.startswith(p) for p in skip_prefixes):
            found.append(primary)
    except Exception:
        pass
    return found


def _http_url(host: str, port: int) -> str:
    if int(port) == 80:
        return f"http://{host}"
    return f"http://{host}:{port}"


def _print_access_urls(host: str, port: int) -> None:
    urls = [_http_url("127.0.0.1", port), _http_url("localhost", port), _http_url("puppet.local", port)]
    if host in ("0.0.0.0", "::", ""):
        for ip in _lan_ipv4s():
            urls.append(_http_url(ip, port))
    elif host not in ("127.0.0.1", "localhost"):
        urls.append(_http_url(host, port))
    seen: set[str] = set()
    logger.info("Eye listening — open:")
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        logger.info("  %s", u)


def _draw_detections(frame: np.ndarray, detections) -> None:
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


def _detections_to_objects(dets, frame_w: int) -> List[Dict[str, Any]]:
    """Serialize YOLO detections for MQTT / LLM (plain JSON types)."""
    objects: List[Dict[str, Any]] = []
    for det in dets:
        cx = 0.5 * (det.x1 + det.x2)
        t = cx / max(frame_w, 1)
        if t < 0.33:
            bearing = "left"
        elif t > 0.66:
            bearing = "right"
        else:
            bearing = "center"
        dist = det.distance_m
        objects.append(
            {
                "label": det.label,
                "conf": round(float(det.conf), 2),
                "dist_m": None
                if dist is None or not np.isfinite(dist)
                else round(float(dist), 2),
                "bearing": bearing,
            }
        )
    objects.sort(key=lambda o: (o["dist_m"] is None, o["dist_m"] or 99.0))
    return objects[:12]


class EyesStreamer:
    """Background MJPEG + one-shot YOLO/DA-V2/Fast-SCNN on Capture."""

    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        fps: int,
        yolo_path: Path,
        depth_path: Path,
        scnn_path: Path,
        conf: float,
        iou: float,
        jpeg_quality: int,
        overlay: bool,
        view: str = "camera",
        scene_pub: Optional[ScenePublisher] = None,
        seg_every: int = 2,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.yolo_path = yolo_path
        self.depth_path = depth_path
        self.scnn_path = scnn_path
        self.conf = conf
        self.iou = iou
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.scene_pub = scene_pub
        self.seg_every = max(1, int(seg_every))

        self._lock = threading.Lock()
        # view: camera | boxes | traverse — selects what Capture computes
        if overlay and view == "camera":
            view = "boxes"
        self._view = view if view in ("camera", "boxes", "traverse") else "camera"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._jpeg = b""
        self._fps = 0.0
        self._infer_ms = 0.0
        self._num_dets = 0
        self._closest_m = float("nan")
        self._hint = ""
        self._error: Optional[str] = None
        self._frame_i = 0
        self._last_scene: Dict[str, Any] = {}
        self._hold_vis: Optional[np.ndarray] = None
        self._capturing = False
        self._capture_pending = threading.Event()
        self._capture_done = threading.Event()
        self._capture_error: Optional[str] = None
        self._pending_req_id: Optional[str] = None
        self._floor = FloorSmoother()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="eyes-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._capture_pending.set()  # unblock waiters
        self._capture_done.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def set_overlay(self, enabled: bool) -> None:
        self.set_view("boxes" if enabled else "camera")

    def set_view(self, view: str) -> None:
        if view not in ("camera", "boxes", "traverse"):
            raise ValueError(f"bad view: {view}")
        with self._lock:
            self._view = view
            if view == "camera":
                self._hold_vis = None

    def request_capture(
        self,
        timeout: float = 60.0,
        *,
        view: Optional[str] = None,
        req_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one inference+publish cycle on the next camera frame."""
        if view is not None:
            self.set_view(view)
        with self._lock:
            if self._view == "camera":
                return {
                    "ok": False,
                    "error": "select Boxes or Traversability before capturing",
                }
            if self._capturing or self._capture_pending.is_set():
                return {"ok": False, "error": "capture already in progress"}
            self._capture_error = None
            self._pending_req_id = req_id
            self._capturing = True
        self._capture_done.clear()
        self._capture_pending.set()
        if not self._capture_done.wait(timeout):
            with self._lock:
                self._capturing = False
                self._pending_req_id = None
                self._capture_pending.clear()
            return {"ok": False, "error": "capture timeout"}
        with self._lock:
            self._capturing = False
            err = self._capture_error
            closest = self._closest_m
            result = {
                "ok": err is None,
                "view": self._view,
                "infer_ms": round(self._infer_ms, 1),
                "dets": self._num_dets,
                "closest_m": None if not np.isfinite(closest) else round(float(closest), 2),
                "hint": self._hint,
                "scene": dict(self._last_scene),
                "req_id": req_id,
                "error": err,
            }
        return result

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            closest = self._closest_m
            pub = self.scene_pub.snapshot() if self.scene_pub else {}
            return {
                "overlay": self._view != "camera",
                "view": self._view,
                "fps": round(self._fps, 1),
                "infer_ms": round(self._infer_ms, 1),
                "dets": self._num_dets,
                "closest_m": None if not np.isfinite(closest) else round(float(closest), 2),
                "hint": self._hint,
                "device": self.device,
                "frame": self._frame_i,
                "error": self._error,
                "capturing": self._capturing or self._capture_pending.is_set(),
                "manual": True,
                "yolo": str(self.yolo_path.name),
                "depth": str(self.depth_path.name),
                "scnn": str(self.scnn_path.name),
                "scene": self._last_scene,
                **pub,
            }

    def jpeg_bytes(self) -> bytes:
        with self._lock:
            return self._jpeg

    def _encode(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return b""
        return buf.tobytes()

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            with self._lock:
                self._error = f"failed to open {self.device}"
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        yolo: Optional[TrtEngine] = None
        depth_engine: Optional[TrtEngine] = None
        scnn: Optional[TrtEngine] = None
        yolo_w = 640
        depth_h = 518
        scnn_h, scnn_w = 256, 640
        engines_loaded = False

        fps_t0 = time.perf_counter()
        fps_n = 0
        try:
            while self._running:
                if not cap.grab():
                    time.sleep(0.01)
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                do_capture = self._capture_pending.is_set()
                if do_capture:
                    self._capture_pending.clear()
                    # Drop buffered frames so capture uses a fresh image.
                    for _ in range(3):
                        if not cap.grab():
                            break
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        with self._lock:
                            self._capture_error = "camera frame grab failed"
                            self._capturing = False
                        self._capture_done.set()
                        continue

                with self._lock:
                    view = self._view
                    hold_vis = self._hold_vis

                closest_m = float("nan")
                num_dets = 0
                infer_ms = 0.0
                hint = ""
                scene_payload: Dict[str, Any] = {}
                capture_err: Optional[str] = None
                vis = frame

                if do_capture:
                    if view == "camera":
                        capture_err = "select Boxes or Traversability before capturing"
                    else:
                        if not engines_loaded:
                            try:
                                yolo = TrtEngine(self.yolo_path)
                                depth_engine = TrtEngine(self.depth_path)
                                yolo_w = int(yolo.input_shape[3])
                                depth_h = int(depth_engine.input_shape[2])
                                if view == "traverse" or self.scnn_path.is_file():
                                    scnn = TrtEngine(self.scnn_path)
                                    scnn_h = int(scnn.input_shape[2])
                                    scnn_w = int(scnn.input_shape[3])
                                engines_loaded = True
                                with self._lock:
                                    self._error = None
                            except Exception as exc:  # noqa: BLE001
                                capture_err = f"engine load failed: {exc}"
                                with self._lock:
                                    self._error = capture_err
                                engines_loaded = False
                                yolo = depth_engine = scnn = None

                        if capture_err is None and yolo is not None and depth_engine is not None:
                            try:
                                t0 = time.perf_counter()
                                yolo_in, info = preprocess_yolo(frame, size=yolo_w)
                                yolo_out = yolo.infer(yolo_in)
                                raw = next(iter(yolo_out.values()))
                                if raw.ndim >= 2 and int(raw.shape[-1]) == 6:
                                    yolo_branch = "end2end"
                                    # Top raw score before threshold (helps debug empty scenes).
                                    pred = raw[0] if raw.ndim == 3 else raw
                                    raw_scores = pred[:, 4].astype(np.float32, copy=False)
                                    top_raw = float(np.max(raw_scores)) if raw_scores.size else 0.0
                                    n_raw = int(np.sum(raw_scores >= self.conf))
                                    dets = postprocess_yolo_end2end(
                                        raw, info, conf_thres=self.conf, max_det=30
                                    )
                                else:
                                    yolo_branch = "classic"
                                    pred = raw[0] if raw.ndim == 3 else raw
                                    if pred.shape[0] < pred.shape[1]:
                                        pred = pred.T
                                    confs = pred[:, 4:].max(axis=1)
                                    top_raw = float(np.max(confs)) if confs.size else 0.0
                                    n_raw = int(np.sum(confs >= self.conf))
                                    dets = postprocess_yolo(
                                        raw,
                                        info,
                                        conf_thres=self.conf,
                                        iou_thres=self.iou,
                                        max_det=30,
                                    )
                                depth_in = preprocess_dav2(frame, size=depth_h)
                                depth_out = depth_engine.infer(depth_in)
                                depth_map = resize_depth_to_frame(
                                    next(iter(depth_out.values())), frame.shape[:2]
                                )
                                attach_metric_distances(dets, depth_map)
                                closest_m, closest_xy = closest_scene_metric(depth_map)
                                num_dets = len(dets)
                                logger.debug(
                                    "YOLO %s shape=%s top=%.3f above_conf=%s kept=%s conf_thres=%s",
                                    yolo_branch,
                                    tuple(raw.shape),
                                    top_raw,
                                    n_raw,
                                    num_dets,
                                    self.conf,
                                )
                                if num_dets:
                                    logger.debug(
                                        "YOLO kept: %s",
                                        ", ".join(
                                            f"{d.label}:{d.conf:.2f}" for d in dets[:8]
                                        ),
                                    )

                                if view == "traverse":
                                    if scnn is None and self.scnn_path.is_file():
                                        scnn = TrtEngine(self.scnn_path)
                                        scnn_h = int(scnn.input_shape[2])
                                        scnn_w = int(scnn.input_shape[3])
                                    if scnn is None:
                                        raise RuntimeError("Fast-SCNN engine not loaded")
                                    scnn_in = preprocess_fast_scnn(frame, scnn_h, scnn_w)
                                    scnn_out = next(iter(scnn.infer(scnn_in).values()))
                                    labels = logits_to_labels(scnn_out)
                                    floor, _wall, _other = remap_apartment(labels)
                                    floor_mask = resize_mask_to_frame(floor, frame.shape[:2])
                                    floor_mask = self._floor.update(floor_mask)
                                    scene = build_scene(
                                        depth_m=depth_map,
                                        floor_mask=floor_mask,
                                        detections=dets,
                                        frame_hw=frame.shape[:2],
                                        closest_m=closest_m,
                                    )
                                    hint = scene.payload.get("hint", "")
                                    scene_payload = dict(scene.payload)
                                    # Always re-attach YOLO objects from the same dets we draw
                                    # (avoids empty MQTT objects while boxes are visible).
                                    scene_payload["objects"] = _detections_to_objects(
                                        dets, frame.shape[1]
                                    )
                                    with self._lock:
                                        pending_id = self._pending_req_id
                                    if pending_id:
                                        scene_payload["req_id"] = pending_id
                                    logger.info(
                                        "scene publish objects=%s hint=%r",
                                        len(scene_payload["objects"]),
                                        hint,
                                    )
                                    vis = overlay_traversability(
                                        frame, scene.free_mask, scene.bev, hint
                                    )
                                    _draw_detections(vis, dets)
                                    if self.scene_pub is not None:
                                        self.scene_pub.publish(scene_payload)
                                else:
                                    vis = frame.copy()
                                    _draw_detections(vis, dets)
                                    if closest_xy is not None and np.isfinite(closest_m):
                                        cv2.circle(
                                            vis, closest_xy, 8, (40, 40, 255), 2, cv2.LINE_AA
                                        )
                                        cv2.circle(
                                            vis, closest_xy, 3, (40, 40, 255), -1, cv2.LINE_AA
                                        )
                                    label = (
                                        f"{num_dets} dets  closest={closest_m:.2f}m"
                                        if np.isfinite(closest_m)
                                        else f"{num_dets} dets"
                                    )
                                    cv2.putText(
                                        vis,
                                        label,
                                        (10, 48),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (40, 220, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )
                                infer_ms = (time.perf_counter() - t0) * 1000.0
                                hold_vis = vis.copy()
                            except Exception as exc:  # noqa: BLE001
                                capture_err = f"capture failed: {exc}"
                                with self._lock:
                                    self._error = capture_err
                        elif capture_err is None:
                            capture_err = "engines not ready"

                    with self._lock:
                        self._capture_error = capture_err
                        self._pending_req_id = None
                        if hold_vis is not None and capture_err is None:
                            self._hold_vis = hold_vis
                        if scene_payload:
                            self._last_scene = scene_payload
                        if capture_err is None:
                            self._infer_ms = float(infer_ms)
                            self._num_dets = int(num_dets)
                            self._closest_m = float(closest_m)
                            self._hint = hint
                    self._capture_done.set()

                # Idle display: freeze last capture when in boxes/traverse; else live
                if view == "camera":
                    if engines_loaded:
                        for eng in (yolo, depth_engine, scnn):
                            if eng is not None:
                                eng.close()
                        yolo = depth_engine = scnn = None
                        engines_loaded = False
                    vis = frame
                    hold_vis = None
                elif hold_vis is not None and not do_capture:
                    vis = hold_vis.copy()
                else:
                    vis = frame if not do_capture or capture_err else vis

                fps_n += 1
                now = time.perf_counter()
                with self._lock:
                    fps = self._fps
                    last_infer_ms = self._infer_ms
                if now - fps_t0 >= 1.0:
                    fps = fps_n / (now - fps_t0)
                    fps_n = 0
                    fps_t0 = now

                show_ms = infer_ms if do_capture and capture_err is None else last_infer_ms
                mode = "CAPTURE" if do_capture and capture_err is None else (
                    "HOLD" if hold_vis is not None and view != "camera" else "LIVE"
                )
                cv2.putText(
                    vis,
                    f"{fps:.1f} FPS  {view}  {mode}  {show_ms:.0f}ms",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

                jpeg = self._encode(vis)
                with self._lock:
                    if jpeg:
                        self._jpeg = jpeg
                    self._fps = float(fps)
                    if do_capture and capture_err is None:
                        # already updated above
                        pass
                    elif view == "camera":
                        self._hold_vis = None
                    self._frame_i += 1
        finally:
            cap.release()
            for eng in (yolo, depth_engine, scnn):
                if eng is not None:
                    eng.close()

streamer: Optional[EyesStreamer] = None
scene_publisher: Optional[ScenePublisher] = None
_cfg: Dict[str, Any] = {}


class DriveBridge:
    """MQTT drive pad — same contract as drive/debug_web."""

    def __init__(self) -> None:
        self.prefix = "robot/drive"
        self.broker = "127.0.0.1"
        self.broker_port = 1883
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.client: Optional[mqtt.Client] = None
        self.last_status: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._holding = False
        self._hold_dir: Optional[str] = None
        self._hold_ttl = 300
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[str] = None
        self._logs: Dict[str, deque] = {
            "brain": deque(maxlen=LOG_MAX_BRAIN),
            "drive": deque(maxlen=LOG_MAX_DRIVE),
        }
        self._log_seq = 0
        self._log_lock = threading.Lock()

    def configure(
        self,
        broker: str,
        port: int,
        prefix: str,
        username: Optional[str],
        password: Optional[str],
    ) -> None:
        self.broker = broker
        self.broker_port = port
        self.prefix = prefix.rstrip("/")
        self.username = username
        self.password = password

    def start(self) -> None:
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"eye_{os.getpid()}",
        )
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(self.broker, self.broker_port, keepalive=30)
            self.client.loop_start()
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._error = f"mqtt connect failed: {exc}"
            logger.warning("%s", self._error)
            self.client = None
            return
        self._running = True
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thread.start()

    def stop(self) -> None:
        self._running = False
        self._holding = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None

    def _topic(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        rc = getattr(reason_code, "value", reason_code)
        logger.info("MQTT connected rc=%s; drive pad %s", rc, self.prefix)
        client.subscribe(self._topic("status"))
        client.subscribe(self._topic("cmd"))
        client.subscribe(LOG_TOPIC)

    def _append_log(self, source: str, *, level: str, logger_name: str, msg: str, ts: float) -> None:
        source = str(source or "drive").lower()
        if source not in self._logs:
            source = "drive"
        with self._log_lock:
            self._log_seq += 1
            self._logs[source].append(
                {
                    "seq": self._log_seq,
                    "source": source,
                    "ts": ts,
                    "level": level,
                    "logger": logger_name,
                    "msg": msg[:800],
                }
            )

    def logs_snapshot(self, after: int = 0) -> Dict[str, Any]:
        with self._log_lock:
            brain = [row for row in self._logs["brain"] if row["seq"] > after]
            drive = [row for row in self._logs["drive"] if row["seq"] > after]
            seq = self._log_seq
        return {"ok": True, "seq": seq, "brain": brain, "drive": drive}

    def clear_logs(self) -> Dict[str, Any]:
        with self._log_lock:
            self._logs["brain"].clear()
            self._logs["drive"].clear()
            self._log_seq = 0
        return {"ok": True, "seq": 0}

    def _on_message(self, client, userdata, msg):
        topic = msg.topic or ""
        try:
            payload = msg.payload.decode("utf-8")
        except Exception:
            return
        if topic.endswith("/status"):
            try:
                self.last_status = json.loads(payload)
            except Exception:
                logger.debug("MQTT status decode failed topic=%s", topic)
            return
        if topic.endswith("/cmd"):
            try:
                data = json.loads(payload) if payload.strip() else {}
            except json.JSONDecodeError:
                data = {}
            cmd = str((data or {}).get("cmd") or "").lower()
            if cmd in ("hb", "heartbeat"):
                return
            self._append_log(
                "drive",
                level="INFO",
                logger_name="mqtt",
                msg=f"cmd {payload.strip() or '{}'}",
                ts=time.time(),
            )
            return
        if topic.startswith("robot/log/"):
            source = topic.rsplit("/", 1)[-1]
            try:
                body = json.loads(payload)
            except json.JSONDecodeError:
                self._append_log(
                    source if source in self._logs else "drive",
                    level="INFO",
                    logger_name=topic,
                    msg=payload[:800],
                    ts=time.time(),
                )
                return
            self._append_log(
                str(body.get("source") or source),
                level=str(body.get("level") or "INFO"),
                logger_name=str(body.get("logger") or topic),
                msg=str(body.get("msg") or payload),
                ts=float(body.get("ts") or time.time()),
            )

    def _publish_cmd(self, payload: Dict[str, Any]) -> None:
        if not self.client:
            raise RuntimeError(self._error or "mqtt not connected")
        cmd = str(payload.get("cmd") or "").lower()
        line = f"MQTT send {self._topic('cmd')} {json.dumps(payload, separators=(',', ':'))}"
        if cmd in ("hb", "heartbeat"):
            logger.debug(line)
        else:
            logger.info(line)
        self.client.publish(self._topic("cmd"), json.dumps(payload), qos=1)

    def _publish_stop(self) -> None:
        if not self.client:
            raise RuntimeError(self._error or "mqtt not connected")
        logger.info("MQTT send %s {}", self._topic("stop"))
        self.client.publish(self._topic("stop"), "{}", qos=1)

    def publish_play_speeds(self, speeds: Dict[str, int]) -> bool:
        """Tell brain to apply follow/seek/forward speeds live (retained)."""
        if not self.client:
            return False
        topic = "robot/play/speeds"
        payload = json.dumps(speeds)
        logger.info("MQTT send %s %s", topic, payload)
        self.client.publish(topic, payload, qos=1, retain=True)
        return True

    def _hb_loop(self) -> None:
        while self._running:
            with self._lock:
                holding = self._holding
                ttl = self._hold_ttl
            if holding:
                try:
                    self._publish_cmd({"cmd": "heartbeat", "ttl": ttl})
                except Exception:
                    pass
            time.sleep(HB_PERIOD_S)

    def hold(self, direction: str, speed: int, ttl_ms: int) -> Dict[str, Any]:
        cmd_map = {
            "forward": "forward",
            "backward": "backward",
            "left": "turn_left",
            "right": "turn_right",
        }
        cmd = cmd_map[direction]
        with self._lock:
            self._holding = True
            self._hold_dir = direction
            self._hold_ttl = ttl_ms
        body: Dict[str, Any] = {"cmd": cmd, "speed": speed, "ttl": ttl_ms, "dur": 0}
        if direction in ("left", "right"):
            body["counts"] = 0
        self._publish_cmd(body)
        self._publish_cmd({"cmd": "heartbeat", "ttl": ttl_ms})
        return {"ok": True, "holding": direction}

    def release(self) -> Dict[str, Any]:
        with self._lock:
            self._holding = False
            self._hold_dir = None
        self._publish_cmd({"cmd": "idle"})
        return {"ok": True, "holding": None}

    def emergency_stop(self) -> Dict[str, Any]:
        with self._lock:
            self._holding = False
            self._hold_dir = None
        self._publish_stop()
        return {"ok": True, "estop": True}

    def clear(self) -> Dict[str, Any]:
        self._publish_cmd({"cmd": "clear"})
        return {"ok": True}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            holding = self._hold_dir if self._holding else None
        return {
            "holding": holding,
            "status": self.last_status,
            "prefix": self.prefix,
            "mqtt_error": self._error,
            "mqtt_ok": self.client is not None,
        }


bridge = DriveBridge()

app = FastAPI(title="Eye", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class OverlayRequest(BaseModel):
    enabled: bool = Field(..., description="True = boxes + metric distances")


class ViewRequest(BaseModel):
    view: str = Field(..., pattern="^(camera|boxes|traverse)$")


class HoldRequest(BaseModel):
    direction: str = Field(..., pattern="^(forward|backward|left|right)$")
    speed: int = Field(100, ge=1, le=200)
    ttl_ms: int = Field(300, ge=50, le=1000)


class LanguageRequest(BaseModel):
    language: str = Field(..., pattern=r"^(en|fr|de)(_[1-9]\d*)?$")


class MicRequest(BaseModel):
    percent: int = Field(..., ge=0, le=150)


class SpeedsRequest(BaseModel):
    follow_turn: int = Field(..., ge=20, le=200)
    seek_turn: int = Field(..., ge=20, le=200)
    forward: int = Field(..., ge=20, le=200)


@app.on_event("startup")
def _startup() -> None:
    global streamer, scene_publisher
    logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())
    broker = str(_cfg.get("broker", os.environ.get("MQTT_BROKER", "127.0.0.1")))
    broker_port = int(_cfg.get("broker_port", os.environ.get("MQTT_PORT", "1883")))
    if bridge.client is None and bridge._error is None:
        bridge.configure(
            broker=broker,
            port=broker_port,
            prefix=str(_cfg.get("prefix", os.environ.get("ROBOT_MQTT_PREFIX", "robot/drive"))),
            username=_cfg.get("username") or os.environ.get("MQTT_USERNAME"),
            password=_cfg.get("password") or os.environ.get("MQTT_PASSWORD"),
        )
        bridge.start()

    def _handle_mqtt_capture(payload: Dict[str, Any]) -> None:
        if streamer is None:
            logger.warning("capture request ignored: streamer not ready")
            return
        view = str(payload.get("view") or "traverse")
        if view not in ("boxes", "traverse"):
            view = "traverse"
        req_id = payload.get("req_id")
        req_id_s = str(req_id) if req_id else None
        timeout = float(payload.get("timeout_s") or 60.0)
        result = streamer.request_capture(timeout=timeout, view=view, req_id=req_id_s)
        if not result.get("ok"):
            logger.warning("MQTT capture failed: %s", result.get("error"))

    if scene_publisher is None:
        scene_publisher = ScenePublisher(
            broker=broker,
            port=broker_port,
            topic=str(_cfg.get("scene_topic", "robot/nav/scene")),
            capture_topic=str(_cfg.get("capture_topic", "robot/nav/capture")),
            username=_cfg.get("username") or os.environ.get("MQTT_USERNAME"),
            password=_cfg.get("password") or os.environ.get("MQTT_PASSWORD"),
            on_capture=_handle_mqtt_capture,
        )
        scene_publisher.start()
    else:
        scene_publisher.set_capture_handler(_handle_mqtt_capture)

    if streamer is not None:
        return
    device = _cfg.get("device")
    if not device:
        device = find_video_device(
            _cfg.get("vendor", DEFAULT_VENDOR_ID),
            _cfg.get("product", DEFAULT_PRODUCT_ID),
        )
    streamer = EyesStreamer(
        device=device,
        width=int(_cfg.get("width", 640)),
        height=int(_cfg.get("height", 480)),
        fps=int(_cfg.get("fps", 30)),
        yolo_path=Path(_cfg.get("yolo", DEFAULT_YOLO)),
        depth_path=Path(_cfg.get("depth", DEFAULT_DEPTH)),
        scnn_path=Path(_cfg.get("scnn", DEFAULT_SCNN)),
        conf=float(_cfg.get("conf", 0.50)),
        iou=float(_cfg.get("iou", 0.45)),
        jpeg_quality=int(_cfg.get("jpeg_quality", 80)),
        overlay=bool(_cfg.get("overlay", False)),
        view=str(_cfg.get("view", "camera")),
        scene_pub=scene_publisher,
        seg_every=int(_cfg.get("seg_every", 2)),
    )
    streamer.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    global streamer, scene_publisher
    try:
        bridge.emergency_stop()
    except Exception:
        pass
    bridge.stop()
    if streamer is not None:
        streamer.stop()
        streamer = None
    if scene_publisher is not None:
        scene_publisher.stop()
        scene_publisher = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def api_state() -> Dict[str, Any]:
    eyes = streamer.snapshot() if streamer is not None else {"error": "streamer not ready"}
    drive = bridge.snapshot()
    return {**eyes, "drive": drive}


@app.get("/api/logs")
def api_logs(after: int = 0) -> Dict[str, Any]:
    """Tail brain/drive MQTT log lines for the debug UI."""
    return bridge.logs_snapshot(after=max(0, int(after)))


@app.post("/api/logs/clear")
def api_logs_clear() -> Dict[str, Any]:
    return bridge.clear_logs()


@app.get("/api/language")
def api_language() -> Dict[str, Any]:
    config_dir = _brain_config_dir()
    language, exists = _read_saved_language(config_dir)
    profiles = list_language_profiles(config_dir)
    known = {item["id"] for item in profiles}
    if known and language not in known:
      language = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in known else (profiles[0]["id"] if profiles else language)
    return {
        "ok": True,
        "language": language,
        "exists": exists,
        "profiles": profiles,
        "file": str(config_dir / "language.active"),
        "applies": "brain_start",
        "note": "Takes effect the next time brain starts",
    }


@app.get("/api/mic")
def api_mic() -> Dict[str, Any]:
    try:
        return get_default_mic()
    except RuntimeError as exc:
        raise HTTPException(503, f"could not read default mic: {exc}") from exc


@app.post("/api/mic")
def api_set_mic(body: MicRequest) -> Dict[str, Any]:
    try:
        result = set_default_mic(body.percent)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, f"could not set default mic: {exc}") from exc
    logger.info(
        "Default mic volume %s%% source=%s",
        result.get("percent"),
        result.get("source"),
    )
    return result


@app.get("/api/speeds")
def api_speeds() -> Dict[str, Any]:
    config_dir = _brain_config_dir()
    speeds, exists = read_play_speeds(config_dir)
    return {
        "ok": True,
        **speeds,
        "exists": exists,
        "file": str(config_dir / "play.speeds"),
        "applies": "live",
        "note": "Saved to play.speeds — kept across restarts",
    }


@app.post("/api/speeds")
def api_set_speeds(body: SpeedsRequest) -> Dict[str, Any]:
    try:
        path, speeds = write_play_speeds(
            _brain_config_dir(),
            {
                "follow_turn": body.follow_turn,
                "seek_turn": body.seek_turn,
                "forward": body.forward,
            },
        )
    except OSError as exc:
        raise HTTPException(500, f"could not write play speeds: {exc}") from exc
    mqtt_ok = False
    try:
        mqtt_ok = bridge.publish_play_speeds(speeds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Play speeds MQTT failed: %s", exc)
    logger.info(
        "Play speeds follow_turn=%s seek_turn=%s forward=%s → %s mqtt=%s",
        speeds["follow_turn"],
        speeds["seek_turn"],
        speeds["forward"],
        path,
        mqtt_ok,
    )
    return {
        "ok": True,
        **speeds,
        "exists": True,
        "file": str(path),
        "mqtt": mqtt_ok,
        "applies": "live" if mqtt_ok else "brain_start",
        "note": "Saved to play.speeds — kept across restarts",
    }


@app.post("/api/language")
def api_set_language(body: LanguageRequest) -> Dict[str, Any]:
    try:
        path = _write_saved_language(_brain_config_dir(), body.language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write language file: {exc}") from exc
    logger.info("Saved brain language %s → %s (applies on brain start)", body.language, path)
    code = parse_language_id(body.language) or body.language
    return {
        "ok": True,
        "language": code,
        "exists": True,
        "profiles": list_language_profiles(_brain_config_dir()),
        "file": str(path),
        "applies": "brain_start",
        "note": "Takes effect the next time brain starts",
    }


@app.post("/api/overlay")
def api_overlay(body: OverlayRequest) -> Dict[str, Any]:
    if streamer is None:
        raise HTTPException(503, "streamer not ready")
    streamer.set_overlay(body.enabled)
    return {"ok": True, "overlay": body.enabled, "view": "boxes" if body.enabled else "camera"}


@app.post("/api/view")
def api_view(body: ViewRequest) -> Dict[str, Any]:
    if streamer is None:
        raise HTTPException(503, "streamer not ready")
    streamer.set_view(body.view)
    return {"ok": True, "view": body.view}


@app.post("/api/capture")
def api_capture() -> Dict[str, Any]:
    """Run one YOLO/depth/(traverse) inference and publish scene if applicable."""
    if streamer is None:
        raise HTTPException(503, "streamer not ready")
    result = streamer.request_capture()
    if not result.get("ok"):
        raise HTTPException(400, result.get("error") or "capture failed")
    return result


@app.post("/api/hold")
def api_hold(body: HoldRequest) -> Dict[str, Any]:
    try:
        return bridge.hold(body.direction, body.speed, body.ttl_ms)
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/release")
def api_release() -> Dict[str, Any]:
    try:
        return bridge.release()
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    try:
        return bridge.emergency_stop()
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/clear")
def api_clear() -> Dict[str, Any]:
    try:
        return bridge.clear()
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.get("/stream.mjpg")
def stream_mjpg() -> StreamingResponse:
    if streamer is None:
        raise HTTPException(503, "streamer not ready")

    boundary = "frame"

    def gen() -> Generator[bytes, None, None]:
        last = b""
        while True:
            jpeg = streamer.jpeg_bytes() if streamer else b""
            if jpeg and jpeg is not last:
                last = jpeg
                yield (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii") + jpeg + b"\r\n"
            time.sleep(0.02)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "close",
        },
    )


class _QuietAccessFilter(logging.Filter):
    """Drop high-frequency poll/stream access lines from uvicorn."""

    _SKIP = ("/api/state", "/api/logs", "/api/mic", "/api/speeds", "/stream.mjpg")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SKIP)


def main() -> None:
    p = argparse.ArgumentParser(description="Eye (camera, drive pad, play speeds)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--device", default=None)
    p.add_argument("--vendor", default=DEFAULT_VENDOR_ID)
    p.add_argument("--product", default=DEFAULT_PRODUCT_ID)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    p.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    p.add_argument("--scnn", type=Path, default=DEFAULT_SCNN)
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument(
        "--overlay",
        action="store_true",
        help="Start with boxes+distance overlay enabled",
    )
    p.add_argument(
        "--view",
        choices=("camera", "boxes", "traverse"),
        default="camera",
        help="Initial view mode (traverse = Fast-SCNN floor + costmap)",
    )
    p.add_argument("--seg-every", type=int, default=2, help="Run Fast-SCNN every N frames")
    p.add_argument("--scene-topic", default="robot/nav/scene")
    p.add_argument("--capture-topic", default="robot/nav/capture")
    p.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "127.0.0.1"))
    p.add_argument(
        "--broker-port",
        type=int,
        default=int(os.environ.get("MQTT_PORT", "1883")),
    )
    p.add_argument(
        "--prefix",
        default=os.environ.get("ROBOT_MQTT_PREFIX", "robot/drive"),
    )
    p.add_argument("--username", default=os.environ.get("MQTT_USERNAME"))
    p.add_argument("--password", default=os.environ.get("MQTT_PASSWORD"))
    p.add_argument(
        "--brain-config",
        default=os.environ.get("PUPPET_CONFIG_DIR", str(DEFAULT_BRAIN_CONFIG)),
        help="Brain config dir for language.active and play.speeds (default: sibling brain/config)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger("paho").setLevel(logging.WARNING)
    logging.getLogger("paho.mqtt").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    for path in (args.yolo, args.depth, args.scnn):
        if not path.is_file():
            raise SystemExit(f"missing engine: {path}")

    device = args.device
    if device is None:
        device = find_video_device(args.vendor, args.product)

    view = args.view
    if args.overlay and view == "camera":
        view = "boxes"

    _cfg.update(
        {
            "device": device,
            "vendor": args.vendor,
            "product": args.product,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "yolo": args.yolo,
            "depth": args.depth,
            "scnn": args.scnn,
            "conf": args.conf,
            "iou": args.iou,
            "jpeg_quality": args.jpeg_quality,
            "overlay": args.overlay,
            "view": view,
            "seg_every": args.seg_every,
            "scene_topic": args.scene_topic,
            "capture_topic": args.capture_topic,
            "broker": args.broker,
            "broker_port": args.broker_port,
            "prefix": args.prefix,
            "username": args.username,
            "password": args.password,
            "brain_config": args.brain_config,
        }
    )

    bridge.configure(
        broker=args.broker,
        port=args.broker_port,
        prefix=args.prefix,
        username=args.username,
        password=args.password,
    )

    import uvicorn

    logger.info("Camera %s", device)
    logger.info("YOLO   %s", args.yolo)
    logger.info("Depth  %s", args.depth)
    logger.info("SCNN   %s", args.scnn)
    logger.info("View   %s", view)
    logger.info("MQTT   %s:%s  drive=%s", args.broker, args.broker_port, args.prefix)
    logger.info("Scene  %s  capture=%s", args.scene_topic, args.capture_topic)
    logger.info("Lang   %s (language.active + play.speeds)", args.brain_config)
    _print_access_urls(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    if "UVICORN_PORT" in os.environ and len(sys.argv) == 1:
        _cfg.setdefault("device", os.environ.get("EYES_DEVICE"))
        _cfg.setdefault("overlay", os.environ.get("EYES_OVERLAY", "0") == "1")
    main()
