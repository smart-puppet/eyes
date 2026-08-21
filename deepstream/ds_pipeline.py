#!/usr/bin/env python3
"""DeepStream nvinfer pipeline: USB cam → YOLO26n + depth → boxes + metres.

Default detector: YOLO26n FP16 (end2end). YOLOv8n via --yolo-config config_infer_yolo.txt.
Default depth: DA-V2 Metric Indoor Base INT8 (metres, nearer=smaller).
Fallback: MiDaS-small relative via --relative-depth --depth-config config_infer_midas.txt

Graph:
  v4l2src (MJPG) ! jpegdec ! nvvidconv ! nvstreammux
    ! nvinfer(yolo) ! nvinfer(depth, tensor-meta)
    ! nvdsosd ! nv3dsink/nveglglessink
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from bus_call import bus_call  # noqa: E402

ROOT = Path(__file__).resolve().parent
YOLO_CFG = str(ROOT / "config_infer_yolo26.txt")
YOLO_V8_CFG = str(ROOT / "config_infer_yolo.txt")  # fallback
DEPTH_CFG = str(ROOT / "config_infer_dav2_metric_base.txt")
MIDAS_CFG = str(ROOT / "config_infer_midas.txt")
YOLO_UID = 1
DEPTH_UID = 2
MIDAS_UID = 2  # legacy alias


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--yolo-config", default=YOLO_CFG)
    p.add_argument(
        "--depth-config",
        default=DEPTH_CFG,
        help="nvinfer config for depth (default: DA-V2 Metric Indoor Base INT8)",
    )
    p.add_argument(
        "--midas-config",
        default=None,
        help="Deprecated alias for --depth-config (MiDaS relative)",
    )
    p.add_argument(
        "--relative-depth",
        action="store_true",
        help="Treat depth tensor as MiDaS-style relative (larger=nearer). "
        "Default is metric metres (DA-V2 Metric, smaller=nearer).",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Only for --relative-depth: meters ≈ depth_scale / relative",
    )
    p.add_argument(
        "--calib-closest-m",
        type=float,
        default=None,
        help="Only for --relative-depth: one-shot scale from known closest metres",
    )
    p.add_argument(
        "--closest-percentile",
        type=float,
        default=None,
        help="Depth percentile for closest sample. "
        "Metric default=1 (near), relative default=99 (near).",
    )
    p.add_argument(
        "--closest-border",
        type=float,
        default=0.08,
        help="Ignore this fraction of depth-map borders when finding closest obstacle",
    )
    p.add_argument("--no-depth", action="store_true", help="YOLO-only (debug)")
    p.add_argument("--no-midas", action="store_true", help=argparse.SUPPRESS)  # legacy
    p.add_argument("--raw", action="store_true", help="Use raw YUYV instead of MJPG")
    p.add_argument(
        "--fakesink",
        action="store_true",
        help="Use fakesink instead of display (CI / headless smoke test)",
    )
    return p.parse_args()


def _layer_to_numpy(layer) -> np.ndarray:
    dims = layer.inferDims
    shape = tuple(int(dims.d[i]) for i in range(dims.numDims))
    ptr = pyds.get_ptr(layer.buffer)
    if layer.dataType == pyds.NvDsInferDataType.FLOAT:
        ctype = ctypes.c_float
    elif layer.dataType == pyds.NvDsInferDataType.HALF:
        ctype = ctypes.c_uint16  # view as float16 below
    elif layer.dataType == pyds.NvDsInferDataType.INT32:
        ctype = ctypes.c_int32
    else:
        ctype = ctypes.c_int8
    buf = ctypes.cast(ptr, ctypes.POINTER(ctype))
    arr = np.ctypeslib.as_array(buf, shape=shape)
    if layer.dataType == pyds.NvDsInferDataType.HALF:
        return arr.view(np.float16).astype(np.float32)
    return np.array(arr, copy=True, dtype=np.float32)


def _depth_map_from_frame_user_meta(frame_meta) -> np.ndarray | None:
    l_user = frame_meta.frame_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tensor_meta.unique_id == DEPTH_UID:
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                depth = _layer_to_numpy(layer)
                # Expected [1,H,W] or [H,W]
                if depth.ndim == 3:
                    depth = depth[0]
                return depth
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None


def _sample_box_depth(
    depth: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
    frame_w: int,
    frame_h: int,
    *,
    metric: bool,
) -> float:
    """Sample depth inside a detection box.

    metric=True  → metres; low percentile (nearer surfaces).
    metric=False → MiDaS relative; larger=nearer; high percentile.
    """
    dh, dw = depth.shape[:2]
    cx = left + 0.5 * width
    cy = top + 0.5 * height
    rw = max(width * 0.175, 1.0)
    rh = max(height * 0.175, 1.0)
    x1 = int(np.clip((cx - rw) / frame_w * dw, 0, dw - 1))
    x2 = int(np.clip((cx + rw) / frame_w * dw, 0, dw - 1))
    y1 = int(np.clip((cy - rh) / frame_h * dh, 0, dh - 1))
    y2 = int(np.clip((cy + rh) / frame_h * dh, 0, dh - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    patch = depth[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return float("nan")
    finite = patch[np.isfinite(patch) & (patch > 1e-6)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 25.0 if metric else 75.0))


def _relative_to_meters(relative_depth: float, depth_scale: float | None) -> float:
    """meters ≈ depth_scale / relative_depth (MiDaS-style only)."""
    if depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0:
        return float("nan")
    if not np.isfinite(relative_depth) or relative_depth <= 1e-6:
        return float("nan")
    return float(depth_scale / relative_depth)


def _format_meters(meters: float, calibrated: bool) -> str:
    if not np.isfinite(meters):
        return "n/a"
    return f"{meters:.2f}m" if calibrated else f"~{meters:.2f}m"


def _closest_scene_sample(
    depth: np.ndarray,
    frame_w: int,
    frame_h: int,
    border_frac: float = 0.08,
    percentile: float = 1.0,
    *,
    metric: bool,
) -> tuple[float, tuple[int, int] | None]:
    """Closest full-frame depth sample → (raw_value, frame_xy)."""
    dh, dw = depth.shape[:2]
    if dh < 8 or dw < 8:
        return float("nan"), None

    border_frac = float(np.clip(border_frac, 0.0, 0.4))
    my = int(dh * border_frac)
    mx = int(dw * border_frac)
    y1, y2 = my, max(my + 1, dh - my)
    x1, x2 = mx, max(mx + 1, dw - mx)
    roi = depth[y1:y2, x1:x2]
    finite = np.isfinite(roi) & (roi > 1e-6)
    if not finite.any():
        return float("nan"), None

    vals = roi[finite]
    pct = float(np.clip(percentile, 0.0, 100.0))
    target = float(np.percentile(vals, pct))
    if metric:
        mask = finite & (roi <= target)
        if not mask.any():
            mask = finite
        masked = np.where(mask, roi, np.inf)
        flat = int(np.argmin(masked))
    else:
        mask = finite & (roi >= target)
        if not mask.any():
            mask = finite
        masked = np.where(mask, roi, -np.inf)
        flat = int(np.argmax(masked))

    ly, lx = np.unravel_index(flat, roi.shape)
    val = float(roi[ly, lx])
    dx = x1 + int(lx)
    dy = y1 + int(ly)
    fx = int(np.clip(dx / max(dw - 1, 1) * (frame_w - 1), 0, frame_w - 1))
    fy = int(np.clip(dy / max(dh - 1, 1) * (frame_h - 1), 0, frame_h - 1))
    return val, (fx, fy)


class ProbeState:
    def __init__(
        self,
        depth_scale: float | None,
        frame_w: int,
        frame_h: int,
        closest_percentile: float,
        closest_border: float = 0.08,
        calib_closest_m: float | None = None,
        *,
        metric: bool = True,
    ):
        self.metric = metric
        self.depth_scale = 800.0 if depth_scale is None else depth_scale
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.closest_percentile = closest_percentile
        self.closest_border = closest_border
        self.calib_closest_m = calib_closest_m
        # Metric DA-V2 outputs metres already.
        self.calibrated = True if metric else bool(
            depth_scale is not None and calib_closest_m is None
        )
        self._calib_rels: list[float] = []
        self.t0 = time.perf_counter()
        self.n = 0
        self.fps = 0.0
        self.closest_m = float("nan")

    def value_to_meters(self, value: float) -> float:
        if self.metric:
            return float(value) if np.isfinite(value) and value > 0 else float("nan")
        return _relative_to_meters(value, self.depth_scale)

    def maybe_calibrate(self, closest_rel: float) -> None:
        if self.metric or self.calib_closest_m is None or self.calibrated:
            return
        if not np.isfinite(closest_rel) or closest_rel <= 1e-6:
            return
        self._calib_rels.append(closest_rel)
        if len(self._calib_rels) < 15:
            return
        rel = float(np.median(self._calib_rels))
        self.depth_scale = float(self.calib_closest_m * rel)
        self.calibrated = True
        print(
            f"depth calibrated: closest={self.calib_closest_m:.3f}m  "
            f"rel={rel:.3f}  => depth_scale={self.depth_scale:.3f}",
            flush=True,
        )


def osd_sink_pad_buffer_probe(pad, info, u_data: ProbeState):
    try:
        return _osd_probe_impl(pad, info, u_data)
    except Exception as exc:  # noqa: BLE001 — never crash the streaming thread
        print(f"probe error: {exc}", file=sys.stderr, flush=True)
        return Gst.PadProbeReturn.OK


def _osd_probe_impl(pad, info, u_data: ProbeState):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        depth = _depth_map_from_frame_user_meta(frame_meta)
        closest_m = float("nan")
        closest_xy: tuple[int, int] | None = None
        if depth is not None:
            closest_val, closest_xy = _closest_scene_sample(
                depth,
                u_data.frame_w,
                u_data.frame_h,
                border_frac=u_data.closest_border,
                percentile=u_data.closest_percentile,
                metric=u_data.metric,
            )
            u_data.maybe_calibrate(closest_val)
            closest_m = u_data.value_to_meters(closest_val)
            u_data.closest_m = closest_m

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            label = obj_meta.obj_label or str(obj_meta.class_id)
            conf = float(obj_meta.confidence)
            rect = obj_meta.rect_params
            dist_txt = ""
            if depth is not None:
                raw = _sample_box_depth(
                    depth,
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                    u_data.frame_w,
                    u_data.frame_h,
                    metric=u_data.metric,
                )
                dist = u_data.value_to_meters(raw)
                if np.isfinite(dist):
                    dist_txt = f" {_format_meters(dist, u_data.calibrated)}"

            text = f"{label} {conf:.2f}{dist_txt}"
            obj_meta.text_params.display_text = text
            obj_meta.text_params.font_params.font_size = 12
            obj_meta.text_params.set_bg_clr = 1
            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
            obj_meta.rect_params.border_width = 2
            obj_meta.rect_params.border_color.set(0.0, 0.85, 0.3, 1.0)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # FPS overlay
        u_data.n += 1
        now = time.perf_counter()
        if now - u_data.t0 >= 1.0:
            u_data.fps = u_data.n / (now - u_data.t0)
            u_data.n = 0
            u_data.t0 = now
            closest_txt = _format_meters(closest_m, u_data.calibrated)
            calib = "cal" if u_data.calibrated else "uncal"
            print(
                f"fps={u_data.fps:.1f} objs={frame_meta.num_obj_meta} "
                f"closest={closest_txt} ({calib}) depth={'yes' if depth is not None else 'no'}",
                flush=True,
            )

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 2
        tp = display_meta.text_params[0]
        mode = "metric" if u_data.metric else "relative"
        tp.display_text = (
            f"YOLO+depth ({mode})  {u_data.fps:.1f} FPS  objs={frame_meta.num_obj_meta}"
        )
        tp.x_offset = 10
        tp.y_offset = 12
        tp.font_params.font_name = "Serif"
        tp.font_params.font_size = 14
        tp.font_params.font_color.set(1.0, 1.0, 0.2, 1.0)
        tp.set_bg_clr = 1
        tp.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)

        tp2 = display_meta.text_params[1]
        closest_label = _format_meters(closest_m, u_data.calibrated)
        if u_data.calibrated:
            tp2.display_text = f"closest obstacle  {closest_label}"
        elif u_data.calib_closest_m is not None:
            tp2.display_text = f"closest obstacle  {closest_label}  (calibrating...)"
        else:
            tp2.display_text = f"closest obstacle  {closest_label}  (uncalibrated)"
        tp2.x_offset = 10
        tp2.y_offset = 36
        tp2.font_params.font_name = "Serif"
        tp2.font_params.font_size = 14
        tp2.font_params.font_color.set(0.2, 1.0, 1.0, 1.0)
        tp2.set_bg_clr = 1
        tp2.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)

        # Mark nearest depth sample on screen (wall / floor / anything).
        if closest_xy is not None and np.isfinite(closest_m):
            display_meta.num_circles = 1
            circle = display_meta.circle_params[0]
            circle.xc = int(closest_xy[0])
            circle.yc = int(closest_xy[1])
            circle.radius = 8
            circle.circle_color.set(1.0, 0.2, 0.2, 1.0)
            circle.has_bg_color = 1
            circle.bg_color.set(1.0, 0.2, 0.2, 0.35)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def make_element(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Failed to create Gst element '{factory}'")
    return el


def build_pipeline(args: argparse.Namespace) -> tuple[Gst.Pipeline, ProbeState]:
    pipeline = Gst.Pipeline.new("eyes-ds-pipeline")

    source = make_element("v4l2src", "camera-source")
    source.set_property("device", args.device)
    # Drop late buffers for lower latency on live USB.
    if source.find_property("do-timestamp") is not None:
        source.set_property("do-timestamp", True)

    caps_src = make_element("capsfilter", "caps-src")
    if args.raw:
        caps_src.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, width={args.width}, height={args.height}, framerate={args.fps}/1"
            ),
        )
        decode = make_element("videoconvert", "raw-convert")
    else:
        caps_src.set_property(
            "caps",
            Gst.Caps.from_string(
                f"image/jpeg, width={args.width}, height={args.height}, framerate={args.fps}/1"
            ),
        )
        decode = make_element("jpegdec", "jpeg-decoder")

    # Prefer L4T nvvidconv on JP 6.2 (nvvideoconvert has known issues).
    conv = make_element("nvvidconv", "nvvidconv-premux")
    caps_nvmm = make_element("capsfilter", "caps-nvmm")
    caps_nvmm.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"),
    )

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batch-size", 1)
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", 16666)
    streammux.set_property("nvbuf-memory-type", 0)

    queue1 = make_element("queue", "queue-yolo")
    queue1.set_property("leaky", 2)  # downstream
    queue1.set_property("max-size-buffers", 1)
    queue1.set_property("max-size-bytes", 0)
    queue1.set_property("max-size-time", 0)

    pgie = make_element("nvinfer", "yolo-nvinfer")
    pgie.set_property("config-file-path", args.yolo_config)

    depth_infer = None
    queue2 = None
    skip_depth = bool(args.no_depth or args.no_midas)
    if not skip_depth:
        queue2 = make_element("queue", "queue-depth")
        queue2.set_property("leaky", 2)
        queue2.set_property("max-size-buffers", 1)
        queue2.set_property("max-size-bytes", 0)
        queue2.set_property("max-size-time", 0)
        depth_infer = make_element("nvinfer", "depth-nvinfer")
        depth_infer.set_property("config-file-path", args.depth_config)

    nvvidconv = make_element("nvvidconv", "nvvidconv-osd")
    nvosd = make_element("nvdsosd", "onscreendisplay")
    nvosd.set_property("process-mode", 0)  # CPU mode works broadly; GPU=1 if available
    nvosd.set_property("display-text", 1)

    if args.fakesink:
        sink = make_element("fakesink", "fake-sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
    else:
        # Jetson display sink preference.
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sink = make_element("nveglglessink", "nvegl-sink")
        sink.set_property("sync", False)
        sink.set_property("qos", False)

    for el in (source, caps_src, decode, conv, caps_nvmm, streammux, queue1, pgie):
        pipeline.add(el)
    if depth_infer is not None:
        pipeline.add(queue2)
        pipeline.add(depth_infer)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    source.link(caps_src)
    caps_src.link(decode)
    decode.link(conv)
    conv.link(caps_nvmm)

    sinkpad = streammux.request_pad_simple("sink_0")
    if sinkpad is None:
        sinkpad = streammux.get_request_pad("sink_0")
    if sinkpad is None:
        raise RuntimeError("Failed to get nvstreammux sink_0 pad")
    srcpad = caps_nvmm.get_static_pad("src")
    link_ret = srcpad.link(sinkpad)
    if link_ret != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"Failed to link camera to nvstreammux: {link_ret}")

    streammux.link(queue1)
    queue1.link(pgie)
    if depth_infer is not None:
        pgie.link(queue2)
        queue2.link(depth_infer)
        depth_infer.link(nvvidconv)
    else:
        pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    metric = not args.relative_depth
    pct = args.closest_percentile
    if pct is None:
        pct = 1.0 if metric else 99.0
    state = ProbeState(
        args.depth_scale,
        args.width,
        args.height,
        closest_percentile=pct,
        closest_border=args.closest_border,
        calib_closest_m=args.calib_closest_m,
        metric=metric,
    )
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Unable to get nvosd sink pad")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, state)
    return pipeline, state


def main() -> int:
    args = parse_args()
    if args.midas_config:
        args.depth_config = args.midas_config
        if "midas" in Path(args.midas_config).name.lower():
            args.relative_depth = True

    for cfg in (args.yolo_config,):
        if not Path(cfg).is_file():
            print(f"error: missing config {cfg}", file=sys.stderr)
            return 1
    skip_depth = bool(args.no_depth or args.no_midas)
    if not skip_depth and not Path(args.depth_config).is_file():
        print(f"error: missing config {args.depth_config}", file=sys.stderr)
        return 1

    parser_libs = (
        ROOT / "libnvdsinfer_yolov8_parser.so",
        ROOT / "libnvdsinfer_custom_impl_Yolo.so",
    )
    if not any(p.is_file() for p in parser_libs):
        print(
            "error: missing YOLO bbox parser .so "
            "(libnvdsinfer_yolov8_parser.so or libnvdsinfer_custom_impl_Yolo.so)",
            file=sys.stderr,
        )
        return 1

    Gst.init(None)
    pipeline, _state = build_pipeline(args)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print(f"Playing {args.device} @ {args.width}x{args.height}", flush=True)
    print(f"YOLO  config: {args.yolo_config}", flush=True)
    if not skip_depth:
        mode = "relative" if args.relative_depth else "metric"
        print(f"Depth config: {args.depth_config} ({mode})", flush=True)
    print("Press Ctrl+C to quit", flush=True)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("error: failed to set pipeline to PLAYING", file=sys.stderr, flush=True)
        return 1
    print(f"Pipeline state change: {ret.value_nick}", flush=True)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
