"""Minimal CUDA Runtime bindings via ctypes (no pycuda required)."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional


class CudaError(RuntimeError):
    pass


def _load_cudart() -> ctypes.CDLL:
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.12",
    ]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    path = ctypes.util.find_library("cudart")
    if path:
        return ctypes.CDLL(path)
    raise CudaError("Could not load libcudart")


_cudart = _load_cudart()
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int
_cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamDestroy.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int
_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int
_cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
_cudart.cudaGetErrorString.restype = ctypes.c_char_p

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3


def _check(err: int) -> None:
    if err != 0:
        msg = _cudart.cudaGetErrorString(err)
        raise CudaError(msg.decode() if msg else f"CUDA error {err}")


def malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _check(_cudart.cudaMalloc(ctypes.byref(ptr), nbytes))
    return ptr.value or 0


def free(ptr: int) -> None:
    if ptr:
        _check(_cudart.cudaFree(ctypes.c_void_p(ptr)))


def memcpy(dst: int, src: int, nbytes: int, kind: int) -> None:
    _check(_cudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind))


def memcpy_htod(dst: int, src_host, nbytes: int) -> None:
    if hasattr(src_host, "ctypes"):
        src_ptr = src_host.ctypes.data
    else:
        src_ptr = src_host
    memcpy(dst, src_ptr, nbytes, cudaMemcpyHostToDevice)


def memcpy_dtoh(dst_host, src: int, nbytes: int) -> None:
    if hasattr(dst_host, "ctypes"):
        dst_ptr = dst_host.ctypes.data
    else:
        dst_ptr = dst_host
    memcpy(dst_ptr, src, nbytes, cudaMemcpyDeviceToHost)


class Stream:
    def __init__(self) -> None:
        handle = ctypes.c_void_p()
        _check(_cudart.cudaStreamCreate(ctypes.byref(handle)))
        self.handle = handle.value or 0

    def synchronize(self) -> None:
        _check(_cudart.cudaStreamSynchronize(ctypes.c_void_p(self.handle)))

    def destroy(self) -> None:
        if self.handle:
            _check(_cudart.cudaStreamDestroy(ctypes.c_void_p(self.handle)))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


def memcpy_htod_async(dst: int, src_host, nbytes: int, stream: Optional[Stream]) -> None:
    src_ptr = src_host.ctypes.data if hasattr(src_host, "ctypes") else src_host
    stream_h = stream.handle if stream else 0
    _check(
        _cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src_ptr),
            nbytes,
            cudaMemcpyHostToDevice,
            ctypes.c_void_p(stream_h),
        )
    )


def memcpy_dtoh_async(dst_host, src: int, nbytes: int, stream: Optional[Stream]) -> None:
    dst_ptr = dst_host.ctypes.data if hasattr(dst_host, "ctypes") else dst_host
    stream_h = stream.handle if stream else 0
    _check(
        _cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src),
            nbytes,
            cudaMemcpyDeviceToHost,
            ctypes.c_void_p(stream_h),
        )
    )
