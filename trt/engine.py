"""TensorRT engine runner with pinned host I/O buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt

from . import cudart


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _dtype_of(trt_dtype: trt.DataType):
    mapping = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32: np.int32,
        trt.int8: np.int8,
        trt.bool: np.bool_,
    }
    # Some argmax heads emit INT64 class indices.
    if hasattr(trt, "int64"):
        mapping[trt.int64] = np.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = np.float32
    try:
        return mapping[trt_dtype]
    except KeyError as exc:
        raise KeyError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc


class TrtEngine:
    """Load a TensorRT engine and run inference with reusable buffers."""

    def __init__(self, engine_path: str | Path, stream: Optional[cudart.Stream] = None):
        self.engine_path = Path(engine_path)
        self.stream = stream or cudart.Stream()
        self._owns_stream = stream is None

        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.bindings: Dict[str, dict] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = _dtype_of(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape unsupported for tensor {name}: {shape}")
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            host = np.empty(shape, dtype=dtype)
            device = cudart.malloc(nbytes)
            self.bindings[name] = {
                "host": host,
                "device": device,
                "nbytes": nbytes,
                "dtype": dtype,
                "shape": shape,
                "mode": mode,
            }
            self.context.set_tensor_address(name, device)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected 1 input, got {self.input_names}")

    @property
    def input_name(self) -> str:
        return self.input_names[0]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.bindings[self.input_name]["shape"]

    def infer(self, inp: np.ndarray, *, copy_outputs: bool = False) -> Dict[str, np.ndarray]:
        """Run inference.

        By default returns views into reusable host buffers (lowest latency).
        Pass ``copy_outputs=True`` if you need to keep tensors across frames.
        """
        self.submit(inp)
        return self.wait(copy_outputs=copy_outputs)

    def submit(self, inp: np.ndarray) -> None:
        """Enqueue H2D + execute + D2H on this engine's stream (no sync)."""
        binding = self.bindings[self.input_name]
        host = binding["host"]
        if inp.shape != host.shape:
            raise ValueError(f"Input shape {inp.shape} != engine {host.shape}")
        if inp.dtype != host.dtype:
            inp = inp.astype(host.dtype, copy=False)
        np.copyto(host, inp)

        cudart.memcpy_htod_async(binding["device"], host, binding["nbytes"], self.stream)
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for name in self.output_names:
            out = self.bindings[name]
            cudart.memcpy_dtoh_async(out["host"], out["device"], out["nbytes"], self.stream)

    def wait(self, *, copy_outputs: bool = False) -> Dict[str, np.ndarray]:
        self.stream.synchronize()
        outputs = {name: self.bindings[name]["host"] for name in self.output_names}
        if copy_outputs:
            return {k: v.copy() for k, v in outputs.items()}
        return outputs

    def close(self) -> None:
        for binding in self.bindings.values():
            cudart.free(binding["device"])
        self.bindings.clear()
        if self._owns_stream:
            self.stream.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
