from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    _reshape_kv_cache,
    init_attn_backend,
    get_shared_kv_cache_layers,
)
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class CacheSegment:
    """One physical backing tensor's bytes inside a swappable cache block.

    ``shared_by`` may contain layers from different cache groups. Those layers
    have independent block tables, so a physical block id belongs to only one
    group at a time even though the runtime views alias the same storage.
    """

    layer_name: str
    shared_by: tuple[str, ...]
    cpu_offset: int
    page_size_bytes: int


@dataclass
class CompositeCache:
    config: KVCacheConfig
    attn_groups: list[list[AttentionGroup]]
    kernel_block_sizes: list[int]
    raw_tensors: dict[str, torch.Tensor]
    runtime_tensors: dict[str, Any]
    segments: list[CacheSegment]
    page_size_bytes: int


def make_composite_config(
    vllm_config: VllmConfig,
    layer_specs: dict[str, KVCacheSpec],
    available_memory_bytes: int,
) -> tuple[KVCacheConfig, list[CacheSegment], int]:
    
    # Worker has already called vLLM's platform block-size alignment hook. For
    # non-integral hybrid page ratios that hook enlarges the attention block
    # and pads only the smaller remainder of the Mamba/GDN page. This grouping
    # pass handles any remaining spec unification. Worker later expands each
    # physical attention block id back into kernel-sized ids.
    groups = get_kv_cache_groups(vllm_config, layer_specs)
    config = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory_bytes
    )
    if config.num_blocks <= 0:
        raise MemoryError(
            "Not enough cache memory for one group-aware cache block: "
            f"available={available_memory_bytes} bytes"
        )

    segments: list[CacheSegment] = []
    offset = 0
    for tensor_spec in config.kv_cache_tensors:
        if tensor_spec.block_stride > 0:
            raise NotImplementedError(
                "Aegaeon swap does not yet support vLLM's packed KV-cache "
                "backing layout"
            )
        if tensor_spec.size % config.num_blocks:
            raise ValueError(
                "KV-cache tensor size must contain an integral page per block: "
                f"size={tensor_spec.size}, blocks={config.num_blocks}"
            )
        page_size = tensor_spec.size // config.num_blocks
        shared_by = tuple(tensor_spec.shared_by)
        if not shared_by:
            raise ValueError("KV-cache tensor is not bound to any layer")
        segments.append(
            CacheSegment(
                layer_name=shared_by[0],
                shared_by=shared_by,
                cpu_offset=offset,
                page_size_bytes=page_size,
            )
        )
        offset += page_size
    return config, segments, offset


def initialize_composite_cache(
    vllm_config: VllmConfig,
    layer_specs: dict[str, KVCacheSpec],
    available_memory_bytes: int,
    device: torch.device,
    allocate,
) -> CompositeCache:
    config, segments, composite_page_size = make_composite_config(
        vllm_config, layer_specs, available_memory_bytes
    )
    # Use vLLM's complete backend initialization.  Besides grouping layers it
    # negotiates kernel block sizes, shares backend workspace, and initializes
    # builders in the order expected by hybrid GDN/full-attention models.
    attn_groups, _, kernel_block_sizes = init_attn_backend(config, vllm_config, device)
    initialize_mamba_ssu_backend(vllm_config.mamba_config, config)
    raw_tensors: dict[str, torch.Tensor] = {}
    for tensor_spec in config.kv_cache_tensors:
        tensor = allocate((tensor_spec.size,), torch.int8)
        # Match vLLM 0.26's _allocate_kv_cache semantics.  Aegaeon's arena
        # allocator deliberately returns uninitialized storage, which can still
        # contain model-staging bytes (including NaN bit patterns).  Hybrid
        # Qwen3.5 prefills do not necessarily overwrite every vectorized cache
        # lane before the attention/GDN kernels read a page, so an ``empty``
        # cache can poison the first forward pass.
        tensor.zero_()
        for layer_name in tensor_spec.shared_by:
            raw_tensors[layer_name] = tensor
    flat_groups = [group for groups in attn_groups for group in groups]
    runtime_tensors = _reshape_kv_cache(
        flat_groups,
        raw_tensors,
        vllm_config.cache_config.cache_dtype,
        kernel_block_sizes,
        shared_kv_cache_layers=get_shared_kv_cache_layers(vllm_config),
        kv_cache_config=config,
    )
    # Cache initialization normally runs on the CUDA default stream, whereas
    # Worker.step executes on Aegaeon's dedicated compute stream.  There was no
    # dependency between those streams, so the first request could race the
    # zero-fill above.  Finish initialization before publishing/binding views.
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return CompositeCache(
        config=config,
        attn_groups=attn_groups,
        kernel_block_sizes=kernel_block_sizes,
        raw_tensors=raw_tensors,
        runtime_tensors=runtime_tensors,
        segments=segments,
        page_size_bytes=composite_page_size,
    )
