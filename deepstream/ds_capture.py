"""DeepStream capture backend: GPU-resident YOLO + Depth with on-demand readout.

Graph (matches the proven ds_pipeline.py layout):
  v4l2src (MJPG) → jpegdec → nvvidconv → nvstreammux
    → nvinfer(YOLO) → nvinfer(depth, tensor-meta)
    → nvvidconv → nvdsosd → fakesink

Probe on the nvosd sink pad (same location as ds_pipeline.py):
  • Every frame: extract the camera image via pyds.get_nvds_buf_surface
    (safe here because nvosd maps the NvBufSurface for drawing).
  • On capture request: also extract YOLO detections + depth tensor.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from trt.yolo import Detection  # noqa: E402

logger = logging.getLogger("eyes.ds")

YOLO_UID = 1
DEPTH_UID = 2


@dataclass
class CaptureResult:
    """Outcome of a single capture."""

    frame: np.ndarray
    detections: List[Detection]
    depth_map: np.ndarray  # HxW float32 metres
    infer_ms: float = 0.0


def _layer_to_numpy(layer) -> np.ndarray:
    dims = layer.inferDims
    shape = tuple(int(dims.d[i]) for i in range(dims.numDims))
    ptr = pyds.get_ptr(layer.buffer)
    if layer.dataType == pyds.NvDsInferDataType.FLOAT:
        ctype = ctypes.c_float
    elif layer.dataType == pyds.NvDsInferDataType.HALF:
        ctype = ctypes.c_uint16
    elif layer.dataType == pyds.NvDsInferDataType.INT32:
        ctype = ctypes.c_int32
    else:
        ctype = ctypes.c_int8
    buf = ctypes.cast(ptr, ctypes.POINTER(ctype))
    arr = np.ctypeslib.as_array(buf, shape=shape)
    if layer.dataType == pyds.NvDsInferDataType.HALF:
        return arr.view(np.float16).astype(np.float32)
    return np.array(arr, copy=True, dtype=np.float32)


def _depth_from_frame_meta(frame_meta) -> Optional[np.ndarray]:
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
                if depth.ndim == 3:
                    depth = depth[0]
                return depth
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None


def _detections_from_frame_meta(
    frame_meta, frame_w: int, frame_h: int, conf_thres: float
) -> List[Detection]:
    dets: List[Detection] = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break
        conf = float(obj_meta.confidence)
        if conf < conf_thres:
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
            continue
        rect = obj_meta.rect_params
        x1 = float(rect.left)
        y1 = float(rect.top)
        x2 = x1 + float(rect.width)
        y2 = y1 + float(rect.height)
        cls_id = int(obj_meta.class_id)
        dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls_id=cls_id))
        try:
            l_obj = l_obj.next
        except StopIteration:
            break
    dets.sort(key=lambda d: d.conf, reverse=True)
    return dets[:50]


class DSCaptureBackend:
    """DeepStream pipeline with on-demand capture readout.

    The pipeline runs continuously.  ``capture()`` signals the probe to
    extract the *next* frame's YOLO + depth results.  Between captures the
    probe still extracts the camera frame (cheap on the nvosd pad) for the
    live MJPEG preview.
    """

    def __init__(
        self,
        *,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        yolo_config: Optional[str] = None,
        depth_config: Optional[str] = None,
        conf: float = 0.50,
        raw_camera: bool = False,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.conf = conf
        self.raw_camera = raw_camera

        ds_dir = Path(__file__).resolve().parent
        self.yolo_config = yolo_config or str(ds_dir / "config_infer_yolo26.txt")
        self.depth_config = depth_config or str(ds_dir / "config_infer_dav2_metric_base.txt")

        self._lock = threading.Lock()
        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._capture_requested = threading.Event()
        self._capture_done = threading.Event()
        self._capture_result: Optional[CaptureResult] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_skip = 0  # extract every Nth frame for preview
        self._paused = False

        self._fps = 0.0
        self._fps_n = 0
        self._fps_t0 = 0.0

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            f = self._latest_frame
            return f.copy() if f is not None else None

    @property
    def current_fps(self) -> float:
        return self._fps

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_pipeline, name="ds-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._capture_requested.set()
        self._capture_done.set()
        if self._loop:
            self._loop.quit()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    def pause(self) -> None:
        """Pause the pipeline to free GPU for other tasks (e.g. LLM)."""
        if self._pipeline and not self._paused:
            self._pipeline.set_state(Gst.State.PAUSED)
            self._paused = True
            logger.info("DeepStream pipeline PAUSED")

    def resume(self) -> None:
        """Resume the pipeline and wait for a fresh frame."""
        if self._pipeline and self._paused:
            self._pipeline.set_state(Gst.State.PLAYING)
            self._paused = False
            logger.info("DeepStream pipeline PLAYING (resumed)")
            # Wait for the first fresh frame so the MJPEG preview updates
            # immediately instead of staying frozen.
            with self._frame_lock:
                self._latest_frame = None
            for _ in range(100):
                if self._latest_frame is not None:
                    break
                time.sleep(0.05)

    def capture(self, timeout: float = 10.0) -> Optional[CaptureResult]:
        if self._paused:
            self.resume()
        with self._lock:
            self._capture_result = None
        self._capture_done.clear()
        self._capture_requested.set()
        if not self._capture_done.wait(timeout):
            return None
        with self._lock:
            return self._capture_result

    # ------------------------------------------------------------------

    def _make_element(self, factory: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Failed to create GStreamer element '{factory}'")
        return el

    def _build_pipeline(self) -> Gst.Pipeline:
        pipeline = Gst.Pipeline.new("ds-capture-pipeline")

        source = self._make_element("v4l2src", "camera-source")
        source.set_property("device", self.device)

        caps_src = self._make_element("capsfilter", "caps-src")
        if self.raw_camera:
            caps_src.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, width={self.width}, height={self.height}, "
                    f"framerate={self.fps}/1"
                ),
            )
            decode = self._make_element("videoconvert", "raw-convert")
        else:
            caps_src.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"image/jpeg, width={self.width}, height={self.height}, "
                    f"framerate={self.fps}/1"
                ),
            )
            decode = self._make_element("jpegdec", "jpeg-decoder")

        conv = self._make_element("nvvidconv", "nvvidconv-premux")
        caps_nvmm = self._make_element("capsfilter", "caps-nvmm")
        caps_nvmm.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"),
        )

        streammux = self._make_element("nvstreammux", "stream-muxer")
        streammux.set_property("width", self.width)
        streammux.set_property("height", self.height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("live-source", 1)
        streammux.set_property("batched-push-timeout", 33000)
        streammux.set_property("nvbuf-memory-type", 0)

        queue1 = self._make_element("queue", "queue-yolo")
        queue1.set_property("leaky", 2)
        queue1.set_property("max-size-buffers", 1)
        queue1.set_property("max-size-bytes", 0)
        queue1.set_property("max-size-time", 0)

        pgie = self._make_element("nvinfer", "yolo-nvinfer")
        pgie.set_property("config-file-path", self.yolo_config)

        queue2 = self._make_element("queue", "queue-depth")
        queue2.set_property("leaky", 2)
        queue2.set_property("max-size-buffers", 1)
        queue2.set_property("max-size-bytes", 0)
        queue2.set_property("max-size-time", 0)

        depth_infer = self._make_element("nvinfer", "depth-nvinfer")
        depth_infer.set_property("config-file-path", self.depth_config)

        # nvvidconv + nvosd — same layout as the proven ds_pipeline.py.
        # The nvosd element maps NvBufSurface so pyds.get_nvds_buf_surface
        # works reliably on its sink pad.
        nvvidconv = self._make_element("nvvidconv", "nvvidconv-osd")
        nvosd = self._make_element("nvdsosd", "onscreendisplay")
        nvosd.set_property("process-mode", 0)
        nvosd.set_property("display-text", 0)

        sink = self._make_element("fakesink", "fakesink")
        sink.set_property("sync", False)
        sink.set_property("async", False)

        for el in (
            source, caps_src, decode,
            conv, caps_nvmm, streammux,
            queue1, pgie, queue2, depth_infer,
            nvvidconv, nvosd, sink,
        ):
            pipeline.add(el)

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
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link camera to nvstreammux")

        streammux.link(queue1)
        queue1.link(pgie)
        pgie.link(queue2)
        queue2.link(depth_infer)
        depth_infer.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        # Probe on the nvosd sink pad — exactly where ds_pipeline.py probes.
        osd_sink_pad = nvosd.get_static_pad("sink")
        if not osd_sink_pad:
            raise RuntimeError("Cannot get nvosd sink pad")
        osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._probe_callback, None
        )

        return pipeline

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    def _probe_callback(self, pad, info, user_data):
        try:
            return self._probe_impl(pad, info)
        except Exception as exc:
            logger.error("probe error: %s", exc, exc_info=True)
            return Gst.PadProbeReturn.OK

    def _probe_impl(self, pad, info) -> int:
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        if l_frame is None:
            return Gst.PadProbeReturn.OK

        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return Gst.PadProbeReturn.OK

        # FPS counter
        self._fps_n += 1
        now = time.perf_counter()
        if self._fps_t0 == 0.0:
            self._fps_t0 = now
        elif now - self._fps_t0 >= 1.0:
            self._fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now

        # Extract the camera frame for the MJPEG preview every 3rd probe
        # call to avoid saturating the GPU with RGBA→BGR conversions.
        do_capture = self._capture_requested.is_set()
        frame = None
        self._frame_skip += 1
        if do_capture or self._frame_skip >= 3:
            self._frame_skip = 0
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame = np.array(n_frame, copy=True)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                with self._frame_lock:
                    self._latest_frame = frame
            except Exception as exc:
                logger.debug("frame extract failed: %s", exc)
                frame = None

        if not do_capture:
            return Gst.PadProbeReturn.OK
        self._capture_requested.clear()

        t0 = time.perf_counter()

        depth = _depth_from_frame_meta(frame_meta)
        if depth is None:
            # Depth nvinfer may not have run on this frame; re-arm and retry
            # on the next one.
            logger.debug("capture: no depth tensor, retrying next frame")
            self._capture_requested.set()
            return Gst.PadProbeReturn.OK

        if depth.ndim == 3:
            depth = depth[0]
        # Keep DA-V2 native resolution (518×518). Floor/BEV run here; the
        # overlay nearest-neighbor upscales the mask onto the camera frame.

        dets = _detections_from_frame_meta(
            frame_meta, self.width, self.height, self.conf
        )

        infer_ms = (time.perf_counter() - t0) * 1000.0

        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        result = CaptureResult(
            frame=frame,
            detections=dets,
            depth_map=depth,
            infer_ms=infer_ms,
        )

        with self._lock:
            self._capture_result = result
        self._capture_done.set()

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Pipeline thread
    # ------------------------------------------------------------------

    def _bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("DeepStream pipeline EOS")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("DeepStream error: %s  debug=%s", err, debug)
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("DeepStream warning: %s  debug=%s", err, debug)
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                logger.debug(
                    "Pipeline state: %s → %s (pending %s)",
                    old.value_nick, new.value_nick, pending.value_nick,
                )
        return True

    def _run_pipeline(self) -> None:
        Gst.init(None)
        try:
            self._pipeline = self._build_pipeline()
        except Exception:
            logger.exception("Failed to build DeepStream pipeline")
            return

        self._loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self._loop)

        logger.info(
            "DeepStream pipeline starting: %s @ %dx%d depth=%s",
            self.device,
            self.width,
            self.height,
            Path(self.depth_config).name,
        )
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start DeepStream pipeline")
            return
        logger.info("DeepStream pipeline PLAYING (state=%s)", ret.value_nick)

        try:
            self._loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            logger.info("DeepStream pipeline stopped")
