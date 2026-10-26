"""
Adopted from vllm/distributed/device_communicators/cuda_wrapper.py.

A pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aegaeon.logger import init_logger

logger = init_logger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int
cudaEvent_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class cudaIpcEventHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 64)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t 	cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char* 	cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t 	cudaMalloc ( void** devPtr, size_t size )
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​cudaError_t 	cudaHostRegister ( void* ptr, size_t size, unsigned int flags )
        Function(
            "cudaHostRegister",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32],
        ),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        # ​cudaError_t cudaIpcCloseMemHandle ( void* devPtr ) # noqa
        Function("cudaIpcCloseMemHandle", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle, cudaEvent_t event ) # noqa
        Function(
            "cudaIpcGetEventHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcEventHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenEventHandle ( cudaIpcEventHandle_t* event, cudaIpcEventHandle_t handle ) # noqa
        Function(
            "cudaIpcOpenEventHandle",
            cudaError_t,
            [
                ctypes.POINTER(ctypes.c_void_p),
                cudaIpcEventHandle_t,
            ],
        ),
        # ​cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, int flags ) # noqa
        Function(
            "cudaStreamWaitEvent",
            cudaError_t,
            [
                cudaStream_t,
                cudaEvent_t,
                ctypes.c_uint,
            ],
        ),
        # ​cudaError_t cudaEventSynchronize ( cudaEvent_t event ) # noqa
        Function("cudaEventSynchronize", cudaError_t, [cudaEvent_t]),
        # ​cudaError_t cudaEventQuery ( cudaEvent_t event ) # noqa
        Function("cudaEventQuery", cudaError_t, [cudaEvent_t]),
        # ​cudaError_t cudaEventDestroy ( cudaEvent_t event ) # noqa
        Function("cudaEventDestroy", cudaError_t, [cudaEvent_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            assert torch.version.cuda is not None
            major_version = torch.version.cuda.split(".")[0]
            so_file = f"libcudart.so.{major_version}"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaHostRegister(self, ptr: ctypes.c_void_p, size: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaHostRegister"](ptr, size, 0))

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr

    def cudaIpcCloseMemHandle(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaIpcCloseMemHandle"](devPtr))

    def cudaIpcGetEventHandle(self, event: cudaEvent_t) -> cudaIpcEventHandle_t:
        handle = cudaIpcEventHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetEventHandle"](ctypes.byref(handle), event)
        )
        return handle

    def cudaIpcOpenEventHandle(self, handle: cudaIpcEventHandle_t) -> cudaEvent_t:
        event = cudaEvent_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenEventHandle"](ctypes.byref(event), handle)
        )
        return event

    def cudaStreamWaitEvent(
        self, stream: cudaStream_t, event: cudaEvent_t, flags: int = 0
    ) -> None:
        self.CUDART_CHECK(self.funcs["cudaStreamWaitEvent"](stream, event, flags))

    def cudaEventQuery(self, event: cudaEvent_t) -> bool:
        return self.funcs["cudaEventQuery"](event) == 0

    def cudaEventDestroy(self, event: cudaEvent_t):
        self.CUDART_CHECK(self.funcs["cudaEventDestroy"](event))
