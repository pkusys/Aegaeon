import numpy as np
import os
import random
import torch
import socket
from typing import Tuple, List, Optional, Union, Dict, TYPE_CHECKING
import warnings

import vllm.envs as envs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.tokenizers import get_tokenizer as _get_tokenizer

from aegaeon.logger import init_logger
from aegaeon.cuda_wrapper import (
    CudaRTLibrary,
    cudaEvent_t,
    cudaStream_t,
    cudaIpcMemHandle_t,
    cudaIpcEventHandle_t,
)

if TYPE_CHECKING:
    from aegaeon.model_registry import ModelSpec
    from aegaeon.device_registry import DeviceSpec
    from aegaeon.config import ParallelConfig
    from aegaeon.stage_engine import StepOutput
    from aegaeon.request import Request

logger = init_logger(__name__)

TTFT_SLO = float(os.environ.get("AEGAEON_TTFT_SLO", "10"))
TPOT_SLO = float(os.environ.get("AEGAEON_TPOT_SLO", "0.1"))

MB = 1024**2
GB = 1024**3
VRAM_ALIGN = 16

DTYPE_SIZE = {
    torch.uint8: 1,
    torch.int8: 1,
    torch.uint16: 2,
    torch.int16: 2,
    torch.uint32: 4,
    torch.int32: 4,
    torch.uint64: 8,
    torch.int64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.multiprocessing.reductions import (
    reduce_tensor,
    rebuild_tensor,
    reduce_storage,
    rebuild_storage_filename,
)


class _DeviceNamespace:
    def __getattr__(self, name: str) -> "DeviceSpec":
        from aegaeon.device_registry import get_device_registry
        spec = get_device_registry().get_by_name(name)
        if spec is not None:
            return spec
        raise AttributeError(
            f"Device '{name}' not found in registry. "
            f"Add it to the device config YAML, or ensure the default registry includes it."
        )

    def from_str(self, s: str) -> Optional["DeviceSpec"]:
        from aegaeon.device_registry import get_device_registry
        return get_device_registry().from_str(s)

    def __iter__(self):
        from aegaeon.device_registry import get_device_registry
        return iter(get_device_registry())

    def __len__(self):
        from aegaeon.device_registry import get_device_registry
        return len(get_device_registry())

    def __contains__(self, item):
        from aegaeon.device_registry import get_device_registry
        return item in get_device_registry()


DeviceType = _DeviceNamespace()


def set_device_registry(path_or_registry):
    from aegaeon.device_registry import DeviceRegistry, set_registry
    if isinstance(path_or_registry, str):
        registry = DeviceRegistry.from_yaml(path_or_registry)
    elif isinstance(path_or_registry, DeviceRegistry):
        registry = path_or_registry
    else:
        raise TypeError(
            f"Expected str (path to YAML) or DeviceRegistry, "
            f"got {type(path_or_registry).__name__}"
        )
    set_registry(registry)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prod(t: Tuple[int, ...]) -> int:
    res = 1
    for v in t:
        res *= v
    return res


def get_distributed_init_method(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def get_open_port() -> int:
    port = envs.VLLM_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = np.zeros([len(x), max_len], dtype=np.int32) + pad
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, : len(blocktb)] = blocktb
    return torch.tensor(padded_x, dtype=dtype, device=device)


def estimate_switch_time(
    model: "ModelSpec",
    device_type: "DeviceSpec",
    parallel_config: "ParallelConfig",
) -> float:
    from aegaeon.config import get_model_config

    PCIE_INEFFICIENCY = 0.15
    model_config = get_model_config(model)
    model_size_in_bytes = model_config.get_model_size_in_bytes(parallel_config)
    bandwidth_in_bytes = device_type.pcie_bandwidth_in_bytes()

    return model_size_in_bytes / (bandwidth_in_bytes * PCIE_INEFFICIENCY)


def reduce_shared_cpu_tensor(tensor: torch.Tensor):
    """
    Transform a shared CPU tensor so that it could be shared in ray's
    object store and rebuilt in other processes, all while being zero-copy
    and using the same storage.

    XXX: Of course, sharing CPU tensors across physical nodes is not possible
    and will result in error if done so. This is arguably why ray doesn't
    provide this functionality in the first space as its object store is
    meant to be cross-node.
    """
    assert (
        tensor.is_cpu and tensor.is_shared()
    ), f"reduce_cpu_tensor() can only reduce CPU tensors in shared memory"

    _, (cls, storage, metadata) = reduce_tensor(tensor)
    _, (*storage,) = reduce_storage(storage)

    # These can now be cloudpickle-ed
    return (cls, storage, metadata)


def rebuild_shared_cpu_tensor(handle) -> torch.Tensor:
    """
    Rebuild a shared CPU tensor from handles obtained with `reduce_shared_cpu_tensor`.
    """
    cls, storage, metadata = handle
    storage = rebuild_storage_filename(*storage)
    return rebuild_tensor(cls, storage, metadata)


def reduce_cuda_event(cudart: CudaRTLibrary, event: torch.cuda.Event):
    """
    Reduce a CUDA event for interprocess sharing.
    """
    handle = cudart.cudaIpcGetEventHandle(cudaEvent_t(event.cuda_event))

    device = torch.cuda.current_device()
    handle = device, bytes(handle)
    # logger.info(f"reduce_cuda_event: {handle}")
    return handle


def rebuild_cuda_event(
    cudart: CudaRTLibrary, handle, is_worker: bool = True
) -> Optional[cudaEvent_t]:
    """
    Rebuild a CUDA event from handle obtained with `reduce_cuda_event`.
    """
    # logger.info(f"rebuild_cuda_event: {handle}")
    device, bytes = handle
    if is_worker and device == torch.cuda.current_device():
        # XXX: a process cannot open an event handle exported by itself
        return None

    handle = cudaIpcEventHandle_t.from_buffer_copy(bytes)
    return cudart.cudaIpcOpenEventHandle(handle)


def get_logits_processor(model: torch.nn.Module) -> LogitsProcessor:
    return model.logits_processor


def get_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "lm_head"):
        return model.lm_head
    elif hasattr(model, "output"):
        # For InternLM
        return model.output
    else:
        raise ValueError(f"lm_head not found for model {type(model)}")


def compute_request_metrics(
    request: "Request", outputs: List["StepOutput"]
) -> Tuple[float, float, List[float]]:
    if len(outputs) == 0 or len(outputs) != request.decode_tokens:
        return (TTFT_SLO, 0, [])

    # TTFT
    ttft = outputs[0].step_time - request.arrival_time

    # QOS
    qos = 0
    each_token = [output.step_time for output in outputs]
    start = request.arrival_time
    accum = 0
    target = TTFT_SLO
    for token in each_token:
        if token - start <= target:
            accum = target
            target += TPOT_SLO
        else:
            qos += accum
            start = token
            accum = 0
            target += TPOT_SLO
    qos += min(target, each_token[-1] - start)
    qos = qos / (each_token[-1] - request.arrival_time)

    per_token = [ttft]
    for t1, t2 in zip(each_token[:-1], each_token[1:]):
        per_token.append(t2 - t1)
    return (ttft, qos, per_token)


def compute_request_latencies(
    request: "Request", outputs: List["StepOutput"]
) -> Dict[str, float]:
    decode_wait_time = 0
    decode_time = 0
    control_overhead = 0
    block_overhead = 0

    prefill_wait_time = outputs[0].dispatch_time - request.arrival_time
    prefill_time = outputs[0].step_time - outputs[0].issue_time
    control_overhead += outputs[0].issue_time - outputs[0].dispatch_time
    for cur, prev in zip(outputs[1:], outputs[:-1]):
        cur_block_overhead = 0 if cur.block_overhead is None else cur.block_overhead
        decode_wait_time += cur.dispatch_time - prev.step_time
        control_overhead += cur.issue_time - cur.dispatch_time
        decode_time += cur.step_time - cur.issue_time - cur_block_overhead
        block_overhead += cur_block_overhead

    return {
        "prefill_wait_time": prefill_wait_time,
        "prefill_time": prefill_time,
        "decode_wait_time": decode_wait_time,
        "decode_time": decode_time,
        "control_overhead": control_overhead,
        "block_overhead": block_overhead,
    }


def ensure_infile(file: str):
    try:
        with open(file, "r") as test:
            pass
    except:
        raise RuntimeError(f"Error opening input file {file}")


def ensure_outfile(file: str):
    try:
        with open(file, "a") as test:
            pass
    except:
        raise RuntimeError(f"Error opening output file {file}")


def get_tokenizer(name: str):
    return _get_tokenizer(name, trust_remote_code=True)
