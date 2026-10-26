import ctypes
import gc
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import ray
import ray.actor
import torch
import vllm.ir
from vllm.config import (
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.models import (
    VllmModelForTextGeneration,
    is_text_generation_model,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions, group_and_batch_mm_kwargs
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.attn_utils import (
    get_kv_cache_spec as get_vllm_kv_cache_spec,
)
from vllm.v1.worker.utils import bind_kv_cache

from aegaeon.allocator import (
    get_alloc,
    has_alloc,
    initialize_alloc,
    release_alloc,
)
from aegaeon.block_manager import CacheGroupRuntimeConfig
from aegaeon.config import ModelConfig, ParallelConfig, QuickLoaderConfig
from aegaeon.cuda_graph import GraphKey, select_decode_graph_key
from aegaeon.device_registry import DeviceSpec, get_device_registry
from aegaeon.foundry_runtime import FoundryRuntime
from aegaeon.graph_bindings import relocate_cuda_buffers, single_split_attention
from aegaeon.loader import get_model
from aegaeon.logger import init_logger
from aegaeon.request import RequestMeta
from aegaeon.utils import (
    GB,
    CudaRTLibrary,
    cudaEvent_t,
    cudaIpcMemHandle_t,
    cudaStream_t,
    make_tensor_with_pad,
    rebuild_cuda_event,
    reduce_cuda_event,
    set_random_seed,
)
from aegaeon.vllm_cache import initialize_composite_cache

if TYPE_CHECKING:
    from aegaeon.stage_engine import Stage

logger = init_logger(__name__)

# Sleep this many seconds when synchronizing for model prefetch.
SLEEP_WHEN_WAITING_PREFETCH = 0.002
_PREFETCH_SKIPPED = object()

# Cap max allocation of finiteness check
_FINITE_CHECK_CHUNK = 64 * (2**20)


def _guard_vllm_c_rms_norm() -> None:
    """Restrict the vLLM-C RMSNorm kernel to the shapes it can actually handle.

    vLLM 0.26's CUDA ``vllm_c`` RMSNorm kernel only supports contiguous 2D
    activations with a 1D weight. Some architectures (e.g. PLaMo-2/3) call
    ``RMSNorm.forward_native`` with reshaped >2D activations or 2D weights,
    which crashes inside the kernel once IR op priorities are installed.
    Narrow the implementation's ``supports_args`` predicate so dispatch falls
    back to the native implementation for those cases.
    """
    try:
        from vllm.ir.op import IrOp
    except Exception:  # pragma: no cover - vLLM version without IR ops
        return
    op = IrOp.registry.get("rms_norm")
    if op is None:
        return
    impl = op.impls.get("vllm_c")
    if impl is None or getattr(impl, "_aegaeon_shape_guarded", False):
        return

    original = impl._supports_args

    def _supports(x, weight, epsilon, variance_size=None):
        if x.dim() != 2 or not x.is_contiguous():
            return False
        if weight is not None and (weight.dim() != 1 or not weight.is_contiguous()):
            return False
        if original is not None and not original(x, weight, epsilon, variance_size):
            return False
        return True

    impl._supports_args = _supports
    impl._aegaeon_shape_guarded = True

"""
XXX: DO NOT rely on ray to assign GPUs to Workers and sort out placement (i.e. by
setting `num_gpus=1` below). We need to share `torch.cuda.Event`s between Workers
which need `torch.device` information to rebuild, requiring a Worker to have
access to all devices.
Instead, set torch.device manually at Worker initialization. 
"""


@ray.remote(num_cpus=16)
class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    initiating KV cache movement, and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    Workers support lightweight model swapping. Each worker is initialized
    lazily without actually loading a model or allocating kvcache yet, which is
    delayed to the switching time (`switch`).

    XXX: inter-GPU migration is currently not implemented due to the extra
    complexity in managing CUDA IPC resource lifetimes. For the use case of
    Aegaeon workers, the impact should be minimal.
    """

    def __init__(
        self,
        stage: "Stage",
        node_id: int,
        engine_id: int,
        worker_id: int,
        device_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        # CPU cache related
        cpu_cache_filename: str,
        cpu_cache_size: int,
        config_path: str = "",
    ) -> None:
        start = time.time()

        if config_path:
            content = os.environ.get("AEGAEON_CONFIG_CONTENT")
            if content:
                from aegaeon.models import load_config_from_content

                load_config_from_content(content)
            else:
                from aegaeon.models import set_model_registry

                set_model_registry(config_path)
        else:
            from aegaeon.config import ServerConfig, set_server_config
            from aegaeon.device_registry import (
                DeviceRegistry,
            )
            from aegaeon.device_registry import (
                set_registry as set_device_reg,
            )
            from aegaeon.model_registry import (
                ModelSpecRegistry,
            )
            from aegaeon.model_registry import (
                set_registry as set_model_reg,
            )

            set_model_reg(ModelSpecRegistry.default())
            set_server_config(ServerConfig())
            set_device_reg(DeviceRegistry.default())

        # Common lazy initialzation; executed only once
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # get rid of whatever ray sets
        if "AEGAEON_CUDA_VISIBLE_DEVICES" in os.environ:
            # respect custom settings
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[
                "AEGAEON_CUDA_VISIBLE_DEVICES"
            ]
        assert torch.cuda.is_available(), "Aegaeon is only available on CUDA devices"
        torch.cuda.set_device(
            device_id
        )  # equivalent to CUDA_VISIBLE_DEVICES set by ray; however other devices are still visible

        if "AEGAEON_CONFIG_PATH" in os.environ:
            content = os.environ.get("AEGAEON_CONFIG_CONTENT")
            if content:
                from aegaeon.models import load_config_from_content

                load_config_from_content(content)
            else:
                from aegaeon.models import set_model_registry

                set_model_registry(os.environ["AEGAEON_CONFIG_PATH"])

        self.stage = stage
        self.node_id = node_id
        self.engine_id = engine_id
        self.worker_id = worker_id
        self.device_id = device_id
        self.model = None
        self.model_config: ModelConfig | None = None
        self.parallel_config: ParallelConfig = parallel_config
        self.device = torch.device("cuda", device_id)
        name = torch.cuda.get_device_name(self.device)
        self.device_type = get_device_registry().from_str(name)
        if self.device_type is None:
            self.device_type = get_device_registry().detect_device(
                self.device.index or 0
            )
        assert (
            self.device_type is not None
        ), f"unsupported device type {torch.cuda.get_device_name(self.device)}"

        self.vllm_config: VllmConfig | None = None

        # Per-layer raw cache
        self.kv_cache: dict[str, torch.Tensor] | None = None
        self.kv_cache_run: dict[str, object] | None = None
        self.composite_cache = None
        self.kv_cache_config = None
        self.attn_groups = None
        self.cache_group_descriptors = None
        self.kv_swap_view = None
        self.kv_block_shape = None
        self._current_positions = None
        self._current_slot_mappings_by_layer = None

        # K/V swap on CPU
        assert os.path.isfile(
            cpu_cache_filename
        ), f"no shared memory object found at {cpu_cache_filename}"
        swap_dtype = torch.float16
        swap_element_size = torch.empty((), dtype=swap_dtype).element_size()
        if cpu_cache_size % swap_element_size:
            raise ValueError(
                f"CPU cache size {cpu_cache_size} is not aligned to "
                f"{swap_element_size}-byte {swap_dtype} elements"
            )
        self.kv_swap: torch.Tensor = torch.from_file(
            cpu_cache_filename,
            shared=True,
            size=cpu_cache_size // swap_element_size,
            dtype=swap_dtype,
            device="cpu",
        )
        self.cudart = CudaRTLibrary()
        # print(self.kv_swap.data_ptr(), self.kv_swap.nbytes)
        self.cudart.cudaHostRegister(
            ctypes.c_void_p(self.kv_swap.data_ptr()), self.kv_swap.nbytes
        )

        # CUDA streams
        # Events from these streams may be shared to other workers for synchronization
        # Refer to block_manager.py for more details.
        self.swap_in_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        self.swap_out_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        self.compute_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)

        # Events
        # Mapping: request_id -> all events (created locally for the request)
        self.move_event_table: dict[int, list[torch.cuda.Event]] = defaultdict(list)
        self.mrope_deltas: dict[int, int] = {}

        # Statistics
        self.execution_time = 0.0

        # Model
        self.model: torch.nn.Module = None
        self.generation_model: VllmModelForTextGeneration[torch.Tensor] | None = None

        # Fast switching
        self.quick_loader = None
        self._model_uses_arena = False
        if os.getenv("FAST_SWITCH") == "1":
            from aegaeon.loader import DEF_LOAD_CONFIG, QuickLoader

            capacity = self.device_type.mem_capacity_in_bytes()
            # Fraction of device memory reserved for the QuickLoader arena.
            # Default 70/80; larger values leave less non-arena headroom for
            # vLLM workspaces/RoPE but allow two large models plus shard
            # staging to coexist during prefetch.  Comes from the loader config
            # because Ray actors on an existing cluster do not inherit the
            # server process environment.
            arena_fraction = getattr(loader_config, "arena_fraction", 0.875)
            self.allocator_size = min(int(capacity * arena_fraction), capacity - 8 * GB)
            # The arena is lazy so DefaultModelLoader models can own ordinary
            # CUDA storage without a large reserved allocation beside them.
            self.quick_loader = QuickLoader(self.cudart, loader_config, DEF_LOAD_CONFIG)
        else:
            raise RuntimeError()

        # Prefetch related
        self.prefetch_task: tuple[ModelConfig, bool, VllmConfig] | None = None
        self.prefetch_model: torch.nn.Module | object | None = None
        self._prefetch_failed_models: set[str] = set()
        self.prefetch_thread = threading.Thread(target=self._prefetch, daemon=True)
        self.prefetch_thread.start()

        self.foundry_runtime: FoundryRuntime | None = None
        self._foundry_decode_warmed_up = False
        self._foundry_decode_graph_saved = False
        self._foundry_decode_graph_state: dict[str, object] | None = None
        self._foundry_decode_graph_states: dict[GraphKey, dict[str, object]] = {}
        self._foundry_pending_decode_graph_loads = None
        self._foundry_model_replay_count = 0

        # Set random seed
        set_random_seed(0)

        end = time.time()
        logger.info(
            f"launched on {self.node_id} and device #{self.device_id} in {end - start:.2f}s"
        )

    def __repr__(self):
        return f"{self.stage} engine #{self.engine_id}, worker #{self.worker_id}|"

    def get_device_type(self) -> DeviceSpec:
        return self.device_type

    def initialize(
        self,
        distributed_init_method: str,
        rank: int,
        local_rank: int,
    ) -> None:
        """Finish worker initialization by setting up the distributed environment."""
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        # TORCH_NCCL_AVOID_RECORD_STREAMS is the default since PyTorch 2.x;
        # setting it explicitly triggers a deprecation warning, so it is removed.

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        self.distributed_init_method = distributed_init_method
        self.rank = rank
        self.local_rank = local_rank

        if self.stage.value == "decode":
            from aegaeon.config import get_server_config
            from aegaeon.model_registry import get_registry

            cuda_graph_config = get_server_config().cuda_graph
            if cuda_graph_config.mode in ("save", "load"):
                self.foundry_runtime = FoundryRuntime(cuda_graph_config)
                model_names = [spec.name for spec in get_registry()]
                self.foundry_runtime.set_up_before_distributed(
                    self.rank,
                    model_names,
                )

        _init_worker_distributed_environment(
            self.parallel_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )
        if self.foundry_runtime is not None:
            if self.foundry_runtime.graph_config.mode == "load":
                self._preload_foundry_model_architectures()
            self.foundry_runtime.set_up_after_distributed()

        # Preload flashinfer, triton_attn, flex_attention and mla_attention
        # to speed up the first time model loading.
        import vllm.model_executor.layers.attention.mla_attention
        import vllm.v1.attention.backends.flash_attn
        import vllm.v1.attention.backends.flashinfer
        import vllm.v1.attention.backends.flex_attention
        import vllm.v1.attention.backends.mla.cutlass_mla
        import vllm.v1.attention.backends.mla.flashattn_mla
        import vllm.v1.attention.backends.mla.flashinfer_mla
        import vllm.v1.attention.backends.mla.flashmla
        import vllm.v1.attention.backends.mla.triton_mla
        import vllm.v1.attention.backends.triton_attn  # noqa: F401

        logger.info(f"initialized on {self.node_id} and device #{self.device_id}")

    def _preload_foundry_model_architectures(self) -> None:
        """Import every configured vLLM model class before the first switch.

        vLLM normally resolves a model class lazily from inside model loading's
        CUDA device context. Some architecture modules create module-level
        tensors while being imported. After a Foundry graph has been loaded,
        creating those import-time tensors on CUDA can interact with the
        active deterministic layout and fail before the next model is built.

        Resolve all classes during Worker initialization instead. This also
        keeps any incidental CUDA initialization inside Foundry's scratch area,
        before set_up_after_distributed advances to the model layout boundary.
        """
        from aegaeon.config import get_model_config
        from aegaeon.model_registry import get_registry
        from vllm.model_executor.model_loader.utils import get_model_architecture

        for spec in get_registry():
            try:
                model_config = get_model_config(spec)
                if model_config is None:
                    raise RuntimeError("model config could not be created")
                model_class, architecture = get_model_architecture(
                    model_config.to_vllm_config()
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to preload vLLM architecture for Foundry model "
                    f"{spec.name!r}"
                ) from exc
            logger.info(
                "preloaded Foundry model architecture %s for %s (%s)",
                architecture,
                spec.name,
                model_class.__name__,
            )

    def _activate_foundry_model_layout(self) -> None:
        """Select the new model before preparing its deterministic layout."""
        if self.foundry_runtime is None:
            return
        if self.model_config is None:
            raise RuntimeError(
                "Cannot activate a Foundry model before model_config is set"
            )
        self.foundry_runtime.activate_model(self.model_config.model.name)
        self.foundry_runtime.prepare_model_layout(self.rank)

    @torch.inference_mode()
    def switch(
        self,
        cache_memory_bytes: int,
        model_config: ModelConfig,
        vllm_config: VllmConfig,
        enable_quick_loader: bool = True,
        prefetch_model_config: ModelConfig | None = None,
        prefetch_enable_quick_loader: bool = True,
        prefetch_vllm_config: VllmConfig | None = None,
    ) -> tuple[int, tuple[int, ...], list[tuple[str, int, int, int | None]]]:
        """Switch the worker to use the given model config.

        Reset the worker state, and (re)loads model weights and initializes kvcache.
        Allocator memory layout:
        +-------+----------+----------------+
        | Model | KV Cache | Prefetch Model |
        +-------+----------+----------------+
        """

        self._wait_model_prefetch()
        self._reset()

        # vLLM 0.26's fused MoE / MLA kernels require a per-worker WorkspaceManager
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
            reset_workspace_manager,
        )
        reset_workspace_manager()
        if not is_workspace_manager_initialized():
            init_workspace_manager(self.device, 1)

        self.model_config = model_config
        vllm_config.parallel_config.rank = self.rank
        self.vllm_config = vllm_config

        self._validate_vllm_parallel_config(vllm_config)
        # Aegaeon does not inherit vLLM's WorkerBase, so install the process-
        # local runtime settings that WorkerBase normally installs. Without
        # this, vLLM 0.26 silently falls back to native IR ops and keeps the
        # torch.library wrapper enabled even in fully eager mode.
        vllm_config.kernel_config.ir_op_priority.set_default()
        _guard_vllm_c_rms_norm()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

        self._activate_foundry_model_layout()

        with set_current_vllm_config(vllm_config):
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
                pipeline_model_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
            )
            self._init_model(model_config, enable_quick_loader)
            if self.foundry_runtime is not None and self._model_uses_arena:
                buffer_bytes = relocate_cuda_buffers(self.model, get_alloc())
                # The copies run on the default stream; graph warmup/capture
                # uses a nonblocking compute stream and must see completed data.
                torch.cuda.default_stream(self.device).synchronize()
                logger.info(
                    "relocated Foundry CUDA model buffers: %d bytes", buffer_bytes
                )
            if (
                self.foundry_runtime is not None
                and self.foundry_runtime.graph_config.mode == "load"
            ):
                self._start_foundry_decode_graph_loads()
            cache_registration = self._init_cache(cache_memory_bytes)

        if self.foundry_runtime is not None:
            self.foundry_runtime.finish_model_layout(
                self.rank,
                {
                    "parameters": dict(self.model.named_parameters()),
                    "buffers": dict(self.model.named_buffers()),
                    "kv_cache": self.kv_cache,
                },
            )
            if self.foundry_runtime.graph_config.mode == "save":
                self._capture_foundry_decode_graphs_during_model_load()
            elif self.foundry_runtime.graph_config.mode == "load":
                self._load_foundry_decode_graphs_during_model_load()

        # Submit prefetch
        if (
            prefetch_model_config is not None
            and str(prefetch_model_config.model) not in self._prefetch_failed_models
        ):
            assert prefetch_vllm_config is not None
            prefetch_vllm_config.parallel_config.rank = self.rank
            self.prefetch_task = (
                prefetch_model_config,
                prefetch_enable_quick_loader,
                prefetch_vllm_config,
            )
        return cache_registration

    def _make_foundry_dummy_decode_batch(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> tuple[list[RequestMeta], list[list[list[int]]]]:
        """Build a cheap pure-decode batch with tables sized for max_seq_len."""
        # seq_len=2/context_len=1 selects the optimized single-token decode
        # path while keeping the capture itself cheap. The attention metadata
        # receives max_seq_len separately, so the captured kernel is valid up
        # to that upper bound.
        requests: list[RequestMeta] = [
            (-(request_index + 1), [0], 2, 1, None)
            for request_index in range(batch_size)
        ]
        if self.vllm_config.model_config.uses_mrope:
            # Dummy decode requests have no preceding prefill from which to
            # derive an M-RoPE delta. A zero delta represents an ordinary
            # text-only sequence and lets _prepare_model_inputs build a valid
            # static positions tensor for capture. Real positions are copied
            # into that tensor before every replay.
            for request_id, *_ in requests:
                self.mrope_deltas[request_id] = 0

        has_recurrent_group = any(
            descriptor.kind == "mamba_none"
            for descriptor in self.cache_group_descriptors
        )
        if has_recurrent_group and batch_size > self.kv_cache_config.num_blocks:
            raise RuntimeError(
                "Foundry dummy decode batch requires one recurrent state "
                f"page per row: batch_size={batch_size}, "
                f"num_blocks={self.kv_cache_config.num_blocks}"
            )

        request_tables: list[list[list[int]]] = []
        for request_index in range(batch_size):
            group_tables: list[list[int]] = []
            for descriptor in self.cache_group_descriptors:
                if descriptor.kind == "attention":
                    table_width = (
                        max_seq_len + descriptor.kv_block_size - 1
                    ) // descriptor.kv_block_size
                    group_tables.append([0] * table_width)
                else:
                    # Each row must update a distinct dummy recurrent page.
                    # These are ordinary schedulable pages, but capture runs
                    # before serving and a real prefill clears a reused page.
                    group_tables.append([request_index])
            request_tables.append(group_tables)
        return requests, request_tables

    def _foundry_decode_graph_paths(self) -> list[tuple[GraphKey, Path]]:
        """Return decode graph keys and archive paths in SAVE capture order."""
        assert self.foundry_runtime is not None
        config = self.foundry_runtime.graph_config
        if len(config.seq_len_buckets) != 1:
            raise ValueError(
                "Model-load Foundry graphs expect one maximum sequence "
                f"length, got {config.seq_len_buckets}"
            )

        max_seq_len = config.seq_len_buckets[0]
        rank_archive_dir = self.foundry_runtime.rank_archive_dir(self.rank)
        paths: list[tuple[GraphKey, Path]] = []
        for graph_index, batch_size in enumerate(
            sorted(config.batch_sizes, reverse=True)
        ):
            key = GraphKey(batch_size=batch_size, seq_len_bucket=max_seq_len)
            path = rank_archive_dir / (
                f"graph_{graph_index}_decode_bs{batch_size}_" f"seq{max_seq_len}.json"
            )
            paths.append((key, path))
        return paths

    def _start_foundry_decode_graph_loads(self) -> None:
        """Start Foundry template construction while Aegaeon initializes KV."""
        import foundry

        graph_paths = self._foundry_decode_graph_paths()
        missing = [str(path) for _, path in graph_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Foundry decode graph archive is incomplete; first missing "
                f"graph: {missing[0]}"
            )

        logger.info(
            "starting Foundry builds for %d decode graphs",
            len(graph_paths),
        )
        self._foundry_pending_decode_graph_loads = foundry.CUDAGraph.start_graph_builds(
            [str(path) for _, path in graph_paths],
            num_threads=4,
        )

    @torch.inference_mode()
    def _load_foundry_decode_graphs_during_model_load(self) -> None:
        """Finish template-aware graph loading in the original SAVE order."""
        assert self.foundry_runtime is not None
        pending = self._foundry_pending_decode_graph_loads
        if pending is None:
            raise RuntimeError("Foundry decode graph builds were not started")

        import foundry

        graph_paths = self._foundry_decode_graph_paths()
        max_seq_len = self.foundry_runtime.graph_config.seq_len_buckets[0]

        self.foundry_runtime.prepare_graph_workspaces(self.compute_stream)

        # Match SAVE's out-of-region warmup. Besides compiling lazy kernels,
        # this keeps library/workspace initialization out of allocator replay.
        with torch.cuda.stream(self.compute_stream):
            for graph_key, _ in graph_paths:
                requests, block_tables = self._make_foundry_dummy_decode_batch(
                    graph_key.batch_size,
                    max_seq_len,
                )
                (
                    input_tokens,
                    input_positions,
                    inputs_embeds,
                    attn_metadata,
                    total_num_scheduled_tokens,
                    _,
                ) = self._prepare_model_inputs(
                    requests,
                    block_tables,
                    max_seq_len_override=max_seq_len,
                )
                self._run_model_forward(
                    input_tokens,
                    input_positions,
                    inputs_embeds,
                    attn_metadata,
                    total_num_scheduled_tokens,
                )
            self.compute_stream.synchronize()

        # Warmup leaves small tensors in PyTorch's ordinary caching pool.
        # Drop every live reference and empty that pool before recreating the
        # static graph inputs inside Foundry's deterministic VMM region.
        del requests, block_tables
        del input_tokens, input_positions, inputs_embeds, attn_metadata, _
        self._current_positions = None
        self._current_slot_mappings_by_layer = None
        gc.collect()
        torch.cuda.empty_cache()

        graph_states: dict[GraphKey, dict[str, object]] = {}
        foundry.resume_allocation_region()
        try:
            with torch.cuda.stream(self.compute_stream):
                for graph_index, (graph_key, graph_path) in enumerate(graph_paths):
                    input_start = self.foundry_runtime.begin_graph_inputs(
                        self.rank, graph_path.name
                    )
                    requests, block_tables = self._make_foundry_dummy_decode_batch(
                        graph_key.batch_size,
                        max_seq_len,
                    )
                    (
                        input_tokens,
                        input_positions,
                        inputs_embeds,
                        attn_metadata,
                        total_num_scheduled_tokens,
                        logits_indices,
                    ) = self._prepare_model_inputs(
                        requests,
                        block_tables,
                        max_seq_len_override=max_seq_len,
                    )

                    self.foundry_runtime.finish_graph_inputs(
                        self.rank,
                        graph_path.name,
                        input_start,
                        {
                            "input_tokens": input_tokens,
                            "input_positions": input_positions,
                            "inputs_embeds": inputs_embeds,
                            "attn_metadata": attn_metadata,
                            "slot_mappings_by_layer": self._current_slot_mappings_by_layer,
                            "logits_indices": logits_indices,
                        },
                    )

                    for slot_mapping in {
                        tensor.data_ptr(): tensor
                        for tensor in self._current_slot_mappings_by_layer.values()
                    }.values():
                        slot_mapping.fill_(-1)

                    decode_graph, loaded_output = (
                        foundry.CUDAGraph.finish_one_graph_load(
                            pending,
                            graph_index,
                        )
                    )
                    if not isinstance(loaded_output, torch.Tensor):
                        raise TypeError(
                            "Expected a Tensor output from Foundry graph "
                            f"{graph_index}, got {type(loaded_output).__name__}"
                        )

                    graph_states[graph_key] = {
                        "graph": decode_graph,
                        "key": graph_key,
                        "input_tokens": input_tokens,
                        "input_positions": input_positions,
                        "inputs_embeds": inputs_embeds,
                        "attn_metadata": attn_metadata,
                        "slot_mappings_by_layer": (
                            self._current_slot_mappings_by_layer
                        ),
                        "logits_indices": logits_indices,
                        "output": loaded_output,
                    }
                    logger.info(
                        "loaded Foundry decode graph %d/%d: "
                        "batch=%d, max_seq_len=%d",
                        graph_index + 1,
                        len(graph_paths),
                        graph_key.batch_size,
                        graph_key.seq_len_bucket,
                    )

            self.compute_stream.synchronize()
            final_offset = foundry.get_current_alloc_offset()
            self.foundry_runtime.validate_final_offset(self.rank)
            foundry.stop_allocation_region()
        except Exception:
            foundry.stop_allocation_region()
            raise

        self._foundry_decode_graph_states = graph_states
        self._foundry_pending_decode_graph_loads = None
        logger.info(
            "loaded %d Foundry decode graph shapes; " "final_offset=%d",
            len(graph_states),
            final_offset,
        )

    @torch.inference_mode()
    def _capture_foundry_decode_graphs_during_model_load(self) -> None:
        """Capture every configured decode batch size before serving requests."""
        assert self.foundry_runtime is not None
        config = self.foundry_runtime.graph_config
        if config.mode != "save":
            return
        if len(config.seq_len_buckets) != 1:
            raise ValueError(
                "Model-load Foundry capture expects one maximum sequence "
                f"length, got {config.seq_len_buckets}"
            )

        import foundry

        max_seq_len = config.seq_len_buckets[0]
        capture_sizes = sorted(config.batch_sizes, reverse=True)
        rank_archive_dir = self.foundry_runtime.rank_archive_dir(self.rank)
        rank_archive_dir.mkdir(parents=True, exist_ok=True)
        # Foundry's manifest scanner also matches graph_manifest.json. Remove
        # the previous run's manifest before overwriting an existing archive.
        (rank_archive_dir / "graph_manifest.json").unlink(missing_ok=True)

        logger.info(
            "warming up Foundry decode capture for %d batch sizes, "
            "batch=%d..%d, max_seq_len=%d",
            len(capture_sizes),
            capture_sizes[-1],
            capture_sizes[0],
            max_seq_len,
        )

        self.foundry_runtime.prepare_graph_workspaces(self.compute_stream)

        # Warm up kernels before entering Foundry's deterministic allocation
        # region. Capture inputs below are allocated again inside it.
        with torch.cuda.stream(self.compute_stream):
            for batch_size in capture_sizes:
                requests, block_tables = self._make_foundry_dummy_decode_batch(
                    batch_size,
                    max_seq_len,
                )
                (
                    input_tokens,
                    input_positions,
                    inputs_embeds,
                    attn_metadata,
                    total_num_scheduled_tokens,
                    _,
                ) = self._prepare_model_inputs(
                    requests,
                    block_tables,
                    max_seq_len_override=max_seq_len,
                )
                self._run_model_forward(
                    input_tokens,
                    input_positions,
                    inputs_embeds,
                    attn_metadata,
                    total_num_scheduled_tokens,
                )
            self.compute_stream.synchronize()

        # Match LOAD exactly: graph inputs must not reuse addresses from the
        # ordinary warmup pool, because those addresses are not reproducible.
        del requests, block_tables
        del input_tokens, input_positions, inputs_embeds, attn_metadata, _
        self._current_positions = None
        self._current_slot_mappings_by_layer = None
        gc.collect()
        torch.cuda.empty_cache()

        graph_states: dict[GraphKey, dict[str, object]] = {}
        foundry.resume_allocation_region()
        try:
            with torch.cuda.stream(self.compute_stream):
                for graph_index, batch_size in enumerate(capture_sizes):
                    graph_name = f"graph_{graph_index}_decode_bs{batch_size}_seq{max_seq_len}.json"
                    input_start = self.foundry_runtime.begin_graph_inputs(
                        self.rank, graph_name
                    )
                    requests, block_tables = self._make_foundry_dummy_decode_batch(
                        batch_size,
                        max_seq_len,
                    )
                    (
                        input_tokens,
                        input_positions,
                        inputs_embeds,
                        attn_metadata,
                        total_num_scheduled_tokens,
                        logits_indices,
                    ) = self._prepare_model_inputs(
                        requests,
                        block_tables,
                        max_seq_len_override=max_seq_len,
                    )

                    self.foundry_runtime.finish_graph_inputs(
                        self.rank,
                        graph_name,
                        input_start,
                        {
                            "input_tokens": input_tokens,
                            "input_positions": input_positions,
                            "inputs_embeds": inputs_embeds,
                            "attn_metadata": attn_metadata,
                            "slot_mappings_by_layer": self._current_slot_mappings_by_layer,
                            "logits_indices": logits_indices,
                        },
                    )

                    # Dummy captures must not write useful request state into
                    # the shared KV cache. These tensors are retained and will
                    # later receive real runtime slot mappings before replay.
                    for slot_mapping in {
                        tensor.data_ptr(): tensor
                        for tensor in self._current_slot_mappings_by_layer.values()
                    }.values():
                        slot_mapping.fill_(-1)

                    decode_graph = foundry.CUDAGraph()
                    with foundry.graph(
                        decode_graph,
                        stream=self.compute_stream,
                    ):
                        captured_hidden_states = self._run_model_forward(
                            input_tokens,
                            input_positions,
                            inputs_embeds,
                            attn_metadata,
                            total_num_scheduled_tokens,
                        )

                    decode_graph.replay()
                    self.compute_stream.synchronize()

                    graph_key = GraphKey(
                        batch_size=batch_size,
                        seq_len_bucket=max_seq_len,
                    )
                    graph_path = rank_archive_dir / (
                        f"graph_{graph_index}_decode_bs{batch_size}_"
                        f"seq{max_seq_len}.json"
                    )
                    decode_graph.save(
                        str(graph_path),
                        output_tensors=captured_hidden_states,
                    )
                    graph_states[graph_key] = {
                        "graph": decode_graph,
                        "key": graph_key,
                        "input_tokens": input_tokens,
                        "input_positions": input_positions,
                        "inputs_embeds": inputs_embeds,
                        "attn_metadata": attn_metadata,
                        "slot_mappings_by_layer": (
                            self._current_slot_mappings_by_layer
                        ),
                        "logits_indices": logits_indices,
                        "output": captured_hidden_states,
                    }
                    logger.info(
                        "captured Foundry decode graph %d/%d: "
                        "batch=%d, max_seq_len=%d",
                        graph_index + 1,
                        len(capture_sizes),
                        batch_size,
                        max_seq_len,
                    )

            foundry.save_graph_manifest(str(rank_archive_dir))
            foundry.pack_fatbins_to_folder(str(rank_archive_dir))
            foundry.set_pack_fatbins_on_exit(False)
            final_offset = self.foundry_runtime.finish_graph_archive(self.rank)
        except Exception:
            foundry.stop_allocation_region()
            raise

        self._foundry_decode_graph_states = graph_states
        self._foundry_decode_graph_saved = True
        logger.info(
            "captured %d Foundry decode graphs during model load; "
            "max_seq_len=%d, final_offset=%d",
            len(graph_states),
            max_seq_len,
            final_offset,
        )

    def _prefetch(self):
        """Prefetch thread."""
        while True:
            if self.prefetch_task is None or self.prefetch_model is not None:
                # No prefetch, or already prefetched.
                time.sleep(SLEEP_WHEN_WAITING_PREFETCH)
                continue

            assert self.prefetch_model is None
            model_config, enable_quick_loader, prefetch_vllm_config = self.prefetch_task
            if not enable_quick_loader:
                # A caller that explicitly disables QuickLoader cannot benefit
                # from arena-backed prefetching.
                logger.info(
                    "prefetch disabled for %s (%s): QuickLoader was disabled",
                    model_config.model,
                    self,
                )
                self.prefetch_model = _PREFETCH_SKIPPED
                continue

            if self.quick_loader is None:
                raise ValueError("QuickLoader not initialized")
            logger.info(f"prefetching {model_config.model} ({self})")
            try:
                self.prefetch_model = get_model(
                    vllm_config=prefetch_vllm_config,
                    quick_loader=self.quick_loader,
                )
            except Exception:
                logger.exception(
                    "prefetch failed for %s (%s); falling back to direct load",
                    model_config.model,
                    self,
                )
                self._prefetch_failed_models.add(str(model_config.model))
                self.prefetch_task = None
                self.prefetch_model = None

    def _reset(self):
        """Reset the worker state, clearing model weights and kvcache.

        This method will wait for all ongoing block movements to complete before
        release GPU memory. It does not, however, ensure that all requests logically
        on the worker have received move-out operations, which is the control plane's
        responsibility.
        """
        foundry_load = (
            self.foundry_runtime is not None
            and self.foundry_runtime.graph_config.mode == "load"
        )
        if self.foundry_runtime is not None:
            self.foundry_runtime.deactivate_model()

        if (
            self.model is None
            and self.composite_cache is None
            and self.vllm_config is None
            and not has_alloc()
            and self._foundry_decode_graph_state is None
            and not self._foundry_decode_graph_states
            and self._foundry_pending_decode_graph_loads is None
        ):
            return

        # Switching models is a lifetime boundary for all old model/cache
        # tensors. Synchronize every stream before releasing their storage.
        torch.cuda.synchronize(self.device)

        # Loaded CUDA graphs retain their input/output tensors and native graph
        # objects. Drop every reference while the old Foundry VMM mapping is
        # still valid; prepare_model_layout() may unmap that region for the next
        # model.
        self._foundry_decode_warmed_up = False
        self._foundry_decode_graph_saved = False
        self._foundry_decode_graph_state = None
        self._foundry_decode_graph_states.clear()
        self._foundry_pending_decode_graph_loads = None
        self._foundry_model_replay_count = 0

        # bind_kv_cache() stores cache tensors on Attention modules. Break
        # those bindings and the config -> layer -> cache ownership chain.
        old_vllm_config = self.vllm_config
        if old_vllm_config is not None:
            compilation_config = getattr(old_vllm_config, "compilation_config", None)
            forward_context = getattr(
                compilation_config, "static_forward_context", None
            )
            if isinstance(forward_context, dict):
                for layer in forward_context.values():
                    if hasattr(layer, "kv_cache"):
                        layer.kv_cache = None
                forward_context.clear()
                layer = None
            forward_context = None
            compilation_config = None
            old_vllm_config = None

        # Clear states
        self.model_config = None
        self.vllm_config = None
        self.kv_cache = None
        self.kv_cache_run = None
        self.kv_swap_view = None
        self.composite_cache = None
        self.kv_cache_config = None
        self.attn_groups = None
        self.cache_group_descriptors = None
        self.cache_segment_group_ids = None
        self._current_positions = None
        self._current_slot_mappings_by_layer = None
        self.mrope_deltas.clear()
        self.model = None
        self.generation_model = None
        self._model_needs_text_embeds = False

        # These process-global objects own model-lifetime CUDA allocations.
        # Clear them while the old mapping is still valid, before empty_cache
        # and prepare_model_layout() can replace its physical backing.
        import vllm.model_executor.layers.rotary_embedding as rotary_embedding

        rotary_embedding._ROPE_DICT.clear()
        clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_workspaces is None and foundry_load:
            raise RuntimeError(
                "Foundry switching requires PyTorch's CUDA BLAS workspace cleanup API"
            )
        if clear_workspaces is not None:
            clear_workspaces()

        if self._model_uses_arena and has_alloc():
            from aegaeon.loader.quick_loader import _DUMMY_BUFFER

            _DUMMY_BUFFER.clear()
            if foundry_load:
                # The arena storage belongs to the active model's preallocated
                # VMM mapping. Rewinding its cursor would keep a stale Storage
                # alive after Foundry replaces that mapping on the next switch.
                release_alloc()
            else:
                get_alloc().clear()
        gc.collect()
        torch.cuda.empty_cache()
        self._model_uses_arena = False

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        logger.info(
            "released previous model/cache on %s: %.2f / %.2f GiB free",
            self.device,
            free_bytes / GB,
            total_bytes / GB,
        )

    def _wait_model_prefetch(self):
        if self.prefetch_task is not None:
            while self.prefetch_task is not None and self.prefetch_model is None:
                time.sleep(SLEEP_WHEN_WAITING_PREFETCH)

    def _prefetch_is_stale(self, model_config: ModelConfig) -> bool:
        prefetch_mc = self.prefetch_task[0] if self.prefetch_task is not None else None
        return bool(
            prefetch_mc is not None
            and prefetch_mc.model == model_config.model  # only checks name
            and prefetch_mc.version != model_config.version
        )

    def _init_model(
        self,
        model_config: ModelConfig,
        enable_quick_loader: bool,
    ) -> None:
        """Initialize the model.
        Loads the model weights from CPU, or wait for the prefetch to complete.
        """
        start = time.time()
        if self.prefetch_model is _PREFETCH_SKIPPED or self._prefetch_is_stale(
            model_config
        ):
            self.prefetch_task = None
            self.prefetch_model = None
        if enable_quick_loader and self.quick_loader is None:
            raise RuntimeError(
                "QuickLoader was requested but is not initialized; refusing "
                "to fall back silently to DefaultModelLoader"
            )
        logger.info(
            "loading %s with %s",
            model_config.model,
            "QuickLoader" if enable_quick_loader else "DefaultModelLoader",
        )
        # QuickLoader and DefaultModelLoader require mutually exclusive CUDA
        # allocation modes.  In particular, merely passing ``quick_loader=None``
        # is insufficient after QuickLoader globally patched nn.Parameter.
        if enable_quick_loader:
            if not has_alloc():
                initialize_alloc(self.allocator_size, self.device)
        elif has_alloc():
            release_alloc()
            gc.collect()
            torch.cuda.empty_cache()
        if self.prefetch_task is None or os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            # No prefetch submitted
            if enable_quick_loader:
                quick_loader = self.quick_loader
            else:
                quick_loader = None
            self.model = get_model(
                vllm_config=self.vllm_config,
                quick_loader=quick_loader,
            )
            how = f"directly ({self})"
        else:
            if self.prefetch_task[0].model != model_config.model:
                logger.error(
                    f"Mismatch between prefetching model {self.prefetch_task[0].model} "
                    f"and scheduled model {model_config.model}"
                )
                raise RuntimeError()

            # Compact the prefetched model into the beginning of the arena.
            # Merely copying bytes is not enough: Parameter objects keep their
            # original Storage, which is later reused by the new KV cache.  If
            # they are not rebound here, cache initialization overwrites model
            # weights and greedy decoding can degenerate to repeated token 0.
            alloc = get_alloc()
            assert isinstance(self.prefetch_model, torch.nn.Module)
            params = [param for param in self.prefetch_model.parameters()]
            relocated_bytes = alloc.relocate_tensors(params)
            logger.info(
                "relocated %d prefetched parameters (%.2f GiB) into compact arena",
                len(params),
                relocated_bytes / GB,
            )
            pf_vllm_config = self.prefetch_task[2]
            assert pf_vllm_config is not None
            self.model = self.prefetch_model
            self.vllm_config = pf_vllm_config
            # XXX: this order guarantees that _prefetch won't interleave
            self.prefetch_task = None
            self.prefetch_model = None
            how = f"with prefetch ({self})"

        # ``torch.nn.Module`` exposes unknown attributes through __getattr__,
        # which is neither a sound type nor runtime guarantee for
        # ``compute_logits``. Validate vLLM's public generation protocol once
        # when the model is loaded and retain the narrowed model separately.
        if not is_text_generation_model(self.model):
            raise TypeError(
                f"Loaded model {type(self.model).__name__} does not implement "
                "vLLM's text-generation interface"
            )
        self.generation_model = self.model
        # For multimodel
        self._model_needs_text_embeds = hasattr(self.model, "per_layer_embeddings")
        self._model_uses_arena = enable_quick_loader

        # vLLM base_loader does not sync after loading,
        # Let's synchronize here to make sure the model is ready to use
        torch.cuda.default_stream(self.device).synchronize()
        end = time.time()
        logger.info(
            f"model {self.model_config.model} loaded {how}; elapsed: {end - start:.2f}s"
        )

    def _init_cache(
        self,
        available_gpu_memory: int,
    ) -> tuple[int, tuple[int, ...], list[tuple[str, int, int, int | None]]]:
        """Allocate the GPU KV cache (the CPU cache is allocated beforehand and shared
        by all engines and workers on a node).

        The KV cache shape is (num_gpu_blocks, *kv_block_shape), where kv_block_shape
        is decided by the model and parallel setting.
        """
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.pipeline_parallel_size == 1
        ), "Current worker impl requires PP=1."
        # vLLM normally runs this hook in its Worker after the model is loaded,
        # because the selected attention backend and the model's Mamba/GDN
        # state shapes are only available then. Aegaeon owns that lifecycle, so
        # it must invoke the hook explicitly before generating KVCacheSpecs.
        #
        # For a non-integral page ratio this is the crucial step: vLLM rounds
        # the attention block size up to a kernel-aligned value whose real KV
        # page covers the recurrent page, then pads only the recurrent page.
        # Example: 4096 B/token, 16-token kernel alignment, 1,097,728 B GDN
        # state -> 272-token attention block and a 1,114,112 B shared page.
        previous_block_size = self.vllm_config.cache_config.block_size
        current_platform.update_block_size_for_backend(self.vllm_config)
        aligned_block_size = self.vllm_config.cache_config.block_size
        logger.info(
            "vLLM hybrid cache alignment: attention block_size %s -> %s, "
            "mamba_page_size_padded=%s",
            previous_block_size,
            aligned_block_size,
            self.vllm_config.cache_config.mamba_page_size_padded,
        )
        kv_cache_spec = self.get_kv_cache_spec()
        recurrent_specs = {
            layer_name: spec
            for layer_name, spec in kv_cache_spec.items()
            if isinstance(spec, MambaSpec)
        }
        if recurrent_specs and self.vllm_config.cache_config.mamba_cache_mode != "none":
            raise ValueError(
                "Aegaeon requires mamba_cache_mode='none' because it owns "
                "request block allocation and migration; got "
                f"{self.vllm_config.cache_config.mamba_cache_mode!r}"
            )

        def allocate(shape, dtype):
            if self._model_uses_arena:
                return get_alloc().allocate(shape, dtype)
            return torch.empty(shape, dtype=dtype, device=self.device)

        # available_gpu_memory is a scheduling upper bound, arena space
        # remaining gives the hard physical limit. This is introduced for
        # dummy weight mode.
        if self._model_uses_arena and has_alloc():
            available_gpu_memory = min(available_gpu_memory, get_alloc().free_size())

        self.composite_cache = initialize_composite_cache(
            self.vllm_config,
            kv_cache_spec,
            available_gpu_memory,
            self.device,
            allocate,
        )
        self.kv_cache_config = self.composite_cache.config
        self.attn_groups = self.composite_cache.attn_groups
        self.kv_cache = self.composite_cache.raw_tensors
        self.kv_cache_run = self.composite_cache.runtime_tensors
        if len(self.composite_cache.kernel_block_sizes) != len(
            self.kv_cache_config.kv_cache_groups
        ):
            raise NotImplementedError(
                "Aegaeon requires one runtime kernel block size per decoder "
                "cache group"
            )
        self.cache_group_descriptors: list[CacheGroupRuntimeConfig] = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            specs = (
                list(group_spec.kv_cache_specs.values())
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else [group_spec]
            )
            if all(isinstance(spec, AttentionSpec) for spec in specs):
                kind = "attention"
            elif all(isinstance(spec, MambaSpec) for spec in specs):
                kind = "mamba_none"
            else:
                raise NotImplementedError(
                    "Aegaeon cannot schedule a cache group containing mixed "
                    f"spec classes: {[type(spec).__name__ for spec in specs]}"
                )
            block_sizes = {spec.block_size for spec in specs}
            if len(block_sizes) != 1:
                raise ValueError(f"Cache group has multiple block sizes: {block_sizes}")
            kv_block_size = block_sizes.pop()
            kernel_block_size = self.composite_cache.kernel_block_sizes[group_id]
            sliding_window = None
            if kind == "attention":
                storage_block_sizes = {spec.storage_block_size for spec in specs}
                if storage_block_sizes != {kv_block_size}:
                    raise NotImplementedError(
                        "Aegaeon swap does not yet support compressed attention "
                        "blocks whose storage_block_size differs from block_size: "
                        f"group={group_id}, block_size={kv_block_size}, "
                        f"storage_block_sizes={storage_block_sizes}"
                    )
                sliding_windows = {
                    getattr(spec, "sliding_window", None) for spec in specs
                }
                # If a block table is shared by full-attention and local-
                # attention layers, the full-attention consumer wins and the
                # whole history remains live. Uniform local-attention groups
                # can safely discard history before their common window.
                non_full_windows = sliding_windows - {None}
                if None in sliding_windows:
                    sliding_window = None
                elif len(non_full_windows) == 1:
                    sliding_window = non_full_windows.pop()
                else:
                    raise NotImplementedError(
                        "Aegaeon requires one sliding-window size per cache "
                        f"group, got {sliding_windows} for group {group_id}"
                    )
            descriptor = CacheGroupRuntimeConfig(
                kind=kind,
                kv_block_size=kv_block_size,
                kernel_block_size=kernel_block_size,
                sliding_window=sliding_window,
            )
            self.cache_group_descriptors.append(descriptor)
            logger.info(
                "cache group %d (%s): %d-token physical block -> "
                "%d x %d-token kernel block%s",
                group_id,
                descriptor.kind,
                descriptor.kv_block_size,
                descriptor.blocks_per_kv_block,
                descriptor.kernel_block_size,
                (
                    ""
                    if descriptor.sliding_window is None
                    else f", sliding window {descriptor.sliding_window}"
                ),
            )

        # A vLLM backing tensor can be shared by one layer from several cache
        # groups. Conversely, a tensor need not participate in every group.
        # Keep this ownership map so swap can avoid copying bytes which are not
        # addressable by the physical block's owning group.
        group_layer_names = [
            set(group.layer_names) for group in self.kv_cache_config.kv_cache_groups
        ]
        self.cache_segment_group_ids: list[frozenset[int]] = []
        for segment in self.composite_cache.segments:
            segment_layers = set(segment.shared_by)
            owners = frozenset(
                group_id
                for group_id, layer_names in enumerate(group_layer_names)
                if segment_layers & layer_names
            )
            if not owners:
                raise ValueError(
                    f"Cache segment {segment.layer_name} is not owned by any "
                    "KV-cache group"
                )
            self.cache_segment_group_ids.append(owners)

        tp_size = self.parallel_config.tensor_parallel_size
        total_cpu_page_bytes = self.composite_cache.page_size_bytes * tp_size
        if total_cpu_page_bytes % self.kv_swap.element_size() != 0:
            raise ValueError(
                f"Composite cache page {total_cpu_page_bytes} is not aligned to "
                f"CPU cache dtype size {self.kv_swap.element_size()}"
            )
        self.kv_block_shape = (total_cpu_page_bytes // self.kv_swap.element_size(),)

        cpu_bytes = self.kv_swap.view(torch.uint8)
        num_cpu_blocks = cpu_bytes.numel() // total_cpu_page_bytes
        self.kv_swap_view = cpu_bytes.narrow(
            0, 0, num_cpu_blocks * total_cpu_page_bytes
        ).view(num_cpu_blocks, total_cpu_page_bytes)
        naive_page_size = sum(spec.page_size_bytes for spec in kv_cache_spec.values())
        logger.info(
            "group-aware cache initialized: %d physical blocks, %d groups, "
            "%d backing tensors, %d bytes/physical-block (old composite: %d)",
            self.kv_cache_config.num_blocks,
            len(self.kv_cache_config.kv_cache_groups),
            len(self.composite_cache.segments),
            self.composite_cache.page_size_bytes,
            naive_page_size,
        )
        bind_kv_cache(
            kv_caches=self.kv_cache_run,
            forward_context=self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches=[],
        )
        return (
            self.kv_cache_config.num_blocks,
            self.kv_block_shape,
            [
                (
                    descriptor.kind,
                    descriptor.kv_block_size,
                    descriptor.kernel_block_size,
                    descriptor.sliding_window,
                )
                for descriptor in self.cache_group_descriptors
            ],
        )

    def get_kv_cache_spec(self):
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # vLLM 0.26 annotates AttentionSpec with the selected backend's physical
        # block-stride convention.  Reimplementing this locally omitted that
        # annotation and can bind a hybrid model to an incompatible cache view.
        return get_vllm_kv_cache_spec(self.vllm_config)
    
    def _count_non_finite_chunked(self, parameter: torch.Tensor):
        """Return (nan, +inf, -inf) counts if ``parameter`` has any non-finite
        value, else None.  Chunked to bound the transient GPU allocation."""
        flat = torch.flatten(parameter)   # view when contiguous (our params are)
        counts = [0, 0, 0]
        found = False
        for start in range(0, flat.numel(), _FINITE_CHECK_CHUNK):
            chunk = flat[start : start + _FINITE_CHECK_CHUNK]
            if not bool(torch.isfinite(chunk).all().item()):
                found = True
                counts[0] += int(torch.isnan(chunk).sum().item())
                counts[1] += int(torch.isposinf(chunk).sum().item())
                counts[2] += int(torch.isneginf(chunk).sum().item())
        if not found:
            return None
        return counts[0], counts[1], counts[2]

    def _validate_loaded_model_parameters(self) -> None:
        """Reject corrupt weights before a request reaches a model forward."""
        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            return
        checked = 0
        for name, parameter in self.model.named_parameters():
            if parameter.numel() == 0 or not parameter.is_floating_point():
                continue
            non_finite = self._count_non_finite_chunked(parameter)
            if non_finite is not None:
                nan_count, posinf_count, neginf_count = non_finite
                raise FloatingPointError(
                    f"Loaded non-finite model parameter {name!r} for "
                    f"{self.model_config.model}: nan={nan_count}, "
                    f"+inf={posinf_count}, -inf={neginf_count}, "
                    f"shape={tuple(parameter.shape)}, dtype={parameter.dtype}"
                )
            checked += 1
        logger.info("validated %d finite model parameters", checked)

    @torch.inference_mode()
    def step(
        self,
        requests: list[RequestMeta],
        block_tables: list[list[list[int]]],
        time_blocked: bool = False,
    ) -> tuple[list[int | None], list[float] | None]:
        """Run one step of inference on the batch of requests.

        This method will wait for involved requests to finish moving in.
        It does not, however, ensure that all requests logically on the worker have
        received move-in operations, which is the stage engine's responsibility.
        """
        block_times = [] if time_blocked else None

        if len(requests) == 0:
            return ([], block_times)

        # Check whether synchronization is necessary;
        # block only when no request in the batch is ready.
        ready_mask: list[bool] = []
        blocked = False
        for i, request in enumerate(requests):
            request_id = request[0]
            if request_id in self.move_event_table:
                # Query if the event is finished already
                event = self.move_event_table[request_id][-1]
                logger.debug(f"querying for {event}")
                if not event.query():
                    ready_mask.append(False)
                    continue
            # No need to pend for this request
            ready_mask.append(True)

        if not any(ready_mask):
            # Sadly no request is ready; must wait for at least one event.
            # For simplicity, we wait for all events.
            blocked = True
            logger.info(f"blocking for {[request[0] for request in requests]}")
            for request in requests:
                request_id = request[0]
                # Block to wait for the event (requirement #1)
                events = self.move_event_table.get(request_id, None)
                if events is not None:
                    events[-1].wait(self.compute_stream)

            # All ready now
            ready_mask = [True for _ in range(len(requests))]

        # Run forward
        with torch.cuda.stream(self.compute_stream):
            # Prepare model inputs
            active_indices = [i for i, ready in enumerate(ready_mask) if ready]
            # vLLM 0.26 metadata builders split a batch at the decode/prefill
            # boundary. Keep decodes first, then restore caller order below.
            execution_indices = sorted(
                active_indices, key=lambda i: requests[i][3] == 0
            )
            execution_requests = [requests[i] for i in execution_indices]
            execution_block_tables = [block_tables[i] for i in execution_indices]

            if time_blocked and blocked:
                start = time.time()
                self.compute_stream.synchronize()
                end = time.time()

            (
                input_tokens,
                input_positions,
                inputs_embeds,
                attn_metadata,
                total_num_scheduled_token,
                logits_indices,
            ) = self._prepare_model_inputs(
                execution_requests,
                execution_block_tables,
            )
            is_pure_decode = all(
                request[3] > 0 and request[2] - request[3] == 1
                for request in execution_requests
            )

            graph_state = None
            if is_pure_decode and self.foundry_runtime is not None:
                graph_key = select_decode_graph_key(
                    self.foundry_runtime.graph_config,
                    batch_size=len(execution_requests),
                    max_seq_len=max(request[2] for request in execution_requests),
                )
                if graph_key is not None:
                    graph_state = self._foundry_decode_graph_states.get(graph_key)

            if graph_state is not None:
                self._copy_foundry_decode_inputs(
                    graph_state,
                    input_tokens,
                    input_positions,
                    attn_metadata,
                    self._current_slot_mappings_by_layer,
                )
                graph_state["graph"].replay()
                if self._foundry_model_replay_count == 0:
                    logger.info(
                        "Foundry decode replay active: model=%s, batch=%d, seq_bucket=%d",
                        self.foundry_runtime.active_model_name,
                        graph_key.batch_size,
                        graph_key.seq_len_bucket,
                    )
                self._foundry_model_replay_count += 1
                hidden_states = graph_state["output"]
            else:
                hidden_states = self._run_model_forward(
                    input_tokens,
                    input_positions,
                    inputs_embeds,
                    attn_metadata,
                    total_num_scheduled_token,
                )
                self.compute_stream.synchronize()
                hidden_states = hidden_states.index_select(0, logits_indices)

            # Greedy sampling (without SamplingMetadata)
            # Use the vLLM generation protocol validated during model loading.
            generation_model = self.generation_model
            if generation_model is None:
                raise RuntimeError("No text-generation model is loaded")
            logits = generation_model.compute_logits(hidden_states)
            if logits is None:
                raise RuntimeError(
                    "Model returned no logits on this worker; greedy sampling "
                    "must run on a tensor-parallel rank that gathers logits"
                )

            generated_token_ids_tensor = torch.argmax(logits, dim=-1).cpu().tolist()

        generated_by_request = [None] * len(requests)
        for request_index, token_id in zip(
            execution_indices, generated_token_ids_tensor
        ):
            generated_by_request[request_index] = token_id

        # Return tokens
        if time_blocked:
            for ready in ready_mask:
                block_time = end - start if blocked and ready else 0
                block_times.append(block_time)

        return (generated_by_request, block_times)

    @staticmethod
    def _copy_foundry_decode_inputs(
        graph_state,
        input_tokens,
        input_positions,
        attn_metadata,
        slot_mappings_by_layer,
    ) -> None:
        """Copy one real decode batch into a Foundry graph's static buffers."""
        graph_state["input_tokens"].copy_(
            input_tokens,
            non_blocking=True,
        )
        graph_state["input_positions"].copy_(
            input_positions,
            non_blocking=True,
        )

        static_slot_mappings = graph_state["slot_mappings_by_layer"]
        copied_slot_mappings = set()
        for layer_name, source in slot_mappings_by_layer.items():
            target = static_slot_mappings[layer_name]
            copy_key = (source.data_ptr(), target.data_ptr())
            if copy_key in copied_slot_mappings:
                continue
            target.copy_(source, non_blocking=True)
            copied_slot_mappings.add(copy_key)

        static_attn_metadata = graph_state["attn_metadata"]
        copied_metadata = set()
        for layer_name, source in attn_metadata.items():
            target = static_attn_metadata[layer_name]
            copy_key = (id(source), id(target))
            if copy_key in copied_metadata:
                continue

            if isinstance(source, GDNAttentionMetadata) or isinstance(
                target, GDNAttentionMetadata
            ):
                if not isinstance(source, GDNAttentionMetadata) or not isinstance(
                    target, GDNAttentionMetadata
                ):
                    raise TypeError(
                        "Foundry GDN metadata type changed between capture and replay: "
                        f"captured={type(target).__name__}, "
                        f"runtime={type(source).__name__}"
                    )

                scalar_fields = (
                    "num_prefills",
                    "num_prefill_tokens",
                    "num_decodes",
                    "num_decode_tokens",
                    "num_spec_decodes",
                    "num_spec_decode_tokens",
                    "num_actual_tokens",
                )
                changed_fields = {
                    field_name: (
                        getattr(target, field_name),
                        getattr(source, field_name),
                    )
                    for field_name in scalar_fields
                    if getattr(target, field_name) != getattr(source, field_name)
                }
                if changed_fields:
                    raise RuntimeError(
                        "Foundry GDN metadata scalar ABI changed between capture "
                        f"and replay: {changed_fields}"
                    )

                for field_name in (
                    "non_spec_query_start_loc",
                    "non_spec_state_indices_tensor",
                ):
                    source_tensor = getattr(source, field_name)
                    target_tensor = getattr(target, field_name)
                    if not isinstance(source_tensor, torch.Tensor) or not isinstance(
                        target_tensor, torch.Tensor
                    ):
                        raise RuntimeError(
                            "Foundry only supports non-speculative GDN decode; "
                            f"missing tensor field {field_name}"
                        )
                    if (
                        source_tensor.shape != target_tensor.shape
                        or source_tensor.dtype != target_tensor.dtype
                        or source_tensor.device != target_tensor.device
                    ):
                        raise RuntimeError(
                            f"Foundry GDN tensor ABI changed for {field_name}: "
                            f"captured=(shape={tuple(target_tensor.shape)}, "
                            f"dtype={target_tensor.dtype}, device={target_tensor.device}), "
                            f"runtime=(shape={tuple(source_tensor.shape)}, "
                            f"dtype={source_tensor.dtype}, device={source_tensor.device})"
                        )
                    target_tensor.copy_(source_tensor, non_blocking=True)

                copied_metadata.add(copy_key)
                continue

            target.query_start_loc.copy_(
                source.query_start_loc,
                non_blocking=True,
            )
            target.seq_lens.copy_(
                source.seq_lens,
                non_blocking=True,
            )
            target.block_table.zero_()
            runtime_width = source.block_table.shape[1]
            target.block_table[:, :runtime_width].copy_(
                source.block_table,
                non_blocking=True,
            )
            copied_metadata.add(copy_key)

    def _run_model_forward(
        self,
        input_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        attn_metadata,
        total_num_scheduled_tokens: int,
    ) -> torch.Tensor:
        """Execute one eager model forward with the metadata just prepared."""
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=total_num_scheduled_tokens,
            slot_mapping=self._current_slot_mappings_by_layer,
        ):
            model_kwargs = {
                "input_ids": input_tokens,
                "positions": input_positions,
                "intermediate_tensors": None,
            }
            if inputs_embeds is not None:
                model_kwargs["inputs_embeds"] = inputs_embeds
                if not getattr(self.model, "requires_raw_input_tokens", False):
                    model_kwargs["input_ids"] = None
            hidden_states = self.model(**model_kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"Expected model forward to return a Tensor, got "
                f"{type(hidden_states).__name__}"
            )
        return hidden_states

    def _prepare_model_inputs(
        self,
        requests: list[RequestMeta],
        _block_tables: list[list[list[int]]],
        max_seq_len_override: int | None = None,
    ):
        """Helper method to prepare the model inputs based on a given batch.
        Prepares metadata needed for the base model forward pass.
        XXX: we assume only the flash-attn backend is used.
        XXX: sliding window is not supported.

        The API assumes requests is sorted by decode -> prefill, matching
        vLLM 0.26's metadata builders.

        The result tensors and data structures use the same decode -> prefill
        order established by ``step``. For example,

        - the leading requests/tokens belong to decode rows;
        - the remaining requests/tokens belong to prefill rows.

        NOTE: Definition of context_len, query_len, and seq_len:
        |---------- N-1 iteration --------|
        |---------------- N iteration ---------------------|
        |- tokenA -|......................|-- newTokens ---|
        |---------- context_len ----------|
        |-------------------- seq_len ---------------------|
                                          |-- query_len ---|
        """
        input_tokens: list[int] = []
        input_positions: list[int] = []
        position_chunks: list[torch.Tensor] = []
        # vLLM keeps model positions and cache/attention positions separate.
        # M-RoPE models pass a (3, num_tokens) tensor to the model, while
        # CommonAttentionMetadata.positions remains ordinary 1-D positions.
        cache_positions: list[int] = []
        num_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        slot_mappings: list[list[int]] = [[] for _ in range(num_cache_groups)]

        seq_lens: list[int] = []
        prefill_seq_lens: list[int] = []
        decode_seq_lens: list[int] = []
        context_lens: list[int] = []
        query_lens: list[int] = []
        block_tables: list[list[list[int]]] = [[] for _ in range(num_cache_groups)]
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        assert len(requests) > 0

        for request, request_block_tables in zip(requests, _block_tables):
            request_id, tokens, seq_len, context_len, multimodal_input = request
            is_prompt = context_len == 0
            if len(request_block_tables) != num_cache_groups:
                raise ValueError(
                    f"request {request_id} has {len(request_block_tables)} "
                    f"cache-group tables; expected {num_cache_groups}"
                )
            if is_prompt:
                self._zero_recurrent_state(request_block_tables)

            for group_id, group_table in enumerate(request_block_tables):
                if not group_table:
                    raise RuntimeError(
                        f"request {request_id} has an empty block table for "
                        f"cache group {group_id}"
                    )
                descriptor = self.cache_group_descriptors[group_id]
                block_tables[group_id].append(
                    descriptor.to_kernel_block_ids(group_table)
                )

            seq_lens.append(seq_len)
            context_lens.append(context_len)
            query_len = seq_len - context_len
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            cache_positions.extend(range(context_len, seq_len))
            if self.vllm_config.model_config.uses_mrope:
                if context_len == 0:
                    features = self._multimodal_features(multimodal_input)
                    mrope_positions, mrope_delta = self.model.get_mrope_input_positions(
                        tokens, features
                    )
                    self.mrope_deltas[request_id] = int(mrope_delta)
                    position_chunks.append(torch.as_tensor(mrope_positions))
                else:
                    mrope_delta = self._mrope_delta_for_decode(
                        request_id,
                        multimodal_input,
                    )
                    position_chunks.append(
                        torch.tensor(
                            MRotaryEmbedding.get_next_input_positions(
                                mrope_delta, context_len, seq_len
                            ),
                            dtype=torch.long,
                        )
                    )
            else:
                input_positions.extend(list(range(context_len, seq_len)))

            if is_prompt:
                num_prefills += 1
                num_prefill_tokens += len(tokens)
                prefill_seq_lens.append(seq_len)
            else:
                assert (
                    query_len == 1
                ), f"seq_len: {seq_len}, context_len: {context_len}, query_len: {query_len}"
                num_decode_tokens += query_len
                decode_seq_lens.append(seq_len)

            # Slot mappings are cache-group-specific. Attention groups map
            # physical KV-manager pages through their kernel-sized virtual
            # subblocks. Mamba state groups address recurrent pages through
            # their block table and do not use token slots.
            for group_id, descriptor in enumerate(self.cache_group_descriptors):
                group_table = request_block_tables[group_id]
                if descriptor.kind == "attention":
                    for position in range(context_len, seq_len):
                        slot_mappings[group_id].append(
                            descriptor.slot_for_position(group_table, position)
                        )
                else:
                    slot_mappings[group_id].extend([-1] * (seq_len - context_len))

        max_query_len = max(query_lens)
        total_query_len = sum(query_lens)
        block_table_tensors: list[torch.Tensor] = []
        for group_id, group_tables in enumerate(block_tables):
            max_group_table_len = max(len(table) for table in group_tables)
            tensor = make_tensor_with_pad(
                group_tables,
                max_len=max_group_table_len,
                pad=0,
                dtype=torch.int32,
                device=self.device,
            )
            if not tensor.is_contiguous():
                logger.warning(
                    "block table for cache group %d is non-contiguous: "
                    "shape=%s, stride=%s",
                    group_id,
                    tuple(tensor.shape),
                    tensor.stride(),
                )
            block_table_tensors.append(tensor)
        assert max_query_len > 0, f"query_lens: {query_lens}"

        context_lens_tensor_cpu = torch.tensor(
            context_lens, dtype=torch.int, device="cpu"
        )

        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=self.device)
        query_lens_tensor = torch.tensor(
            query_lens, dtype=torch.long, device=self.device
        )
        query_start_loc = torch.zeros(
            query_lens_tensor.shape[0] + 1, dtype=torch.int32, device=self.device
        )
        seq_start_loc = torch.zeros(
            seq_lens_tensor.shape[0] + 1, dtype=torch.int32, device=self.device
        )

        torch.cumsum(
            seq_lens_tensor, dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:]
        )
        torch.cumsum(
            query_lens_tensor,
            dim=0,
            dtype=query_start_loc.dtype,
            out=query_start_loc[1:],
        )

        input_tokens_tensor = torch.tensor(
            input_tokens, dtype=torch.long, device=self.device
        )
        if position_chunks:
            input_positions_tensor = torch.cat(position_chunks, dim=1).to(
                device=self.device, dtype=torch.long
            )
        else:
            input_positions_tensor = torch.tensor(
                input_positions, dtype=torch.long, device=self.device
            )
        self._current_positions = torch.tensor(
            cache_positions, dtype=torch.long, device=self.device
        )
        slot_mappings_tensor = torch.tensor(
            slot_mappings, dtype=torch.long, device=self.device
        )
        self._current_slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings_tensor, self.kv_cache_config
        )

        logits_indices = query_start_loc[1:] - 1
        inputs_embeds = self._prepare_multimodal_embeddings(
            requests, input_tokens_tensor, query_start_loc.cpu()
        )
        attn_metadata = self._build_attention_metadata_simplified(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens_tensor,
            seq_lens_cpu=seq_lens_tensor.cpu(),
            num_computed_tokens_cpu=context_lens_tensor_cpu,  # MARK
            block_tables=block_table_tensors,
            slot_mappings=slot_mappings_tensor,
            total_num_scheduled_tokens=total_query_len,  # MARK
            max_query_len=max_query_len,  # MARK
            num_reqs=len(requests),
            max_seq_len_override=max_seq_len_override,
        )
        return (
            input_tokens_tensor,
            input_positions_tensor,
            inputs_embeds,
            attn_metadata,
            total_query_len,
            logits_indices,
        )

    def _mrope_delta_for_decode(
        self,
        request_id: int,
        multimodal_input,
    ) -> int:
        """Resolve M-RoPE state after a request migrates to this worker."""
        cached_delta = self.mrope_deltas.get(request_id)
        if cached_delta is not None:
            return cached_delta

        if multimodal_input:
            _, mrope_delta = self.model.get_mrope_input_positions(
                multimodal_input["prompt_token_ids"],
                self._multimodal_features(multimodal_input),
            )
        else:
            # With no image/video features, Qwen's three M-RoPE axes are all
            # ordinary text positions and max(position) + 1 - prompt_len is 0.
            # A decode worker does not see the remote prefill that established
            # this value, so reconstruct the text-only delta locally.
            mrope_delta = 0

        resolved_delta = int(mrope_delta)
        self.mrope_deltas[request_id] = resolved_delta
        return resolved_delta

    def _zero_recurrent_state(self, block_tables: list[list[int]]) -> None:
        """Clear reused GDN/Mamba pages before a brand-new prefill."""
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            layer_specs = (
                group_spec.kv_cache_specs
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else {layer_name: group_spec for layer_name in group.layer_names}
            )
            if not all(isinstance(spec, MambaSpec) for spec in layer_specs.values()):
                continue
            # In mode "none", this group's table has exactly one state page
            # per request. Different Mamba groups own different physical ids.
            block_id = block_tables[group_id][0]
            for layer_name in group.layer_names:
                segment = next(
                    segment
                    for segment in self.composite_cache.segments
                    if layer_name in segment.shared_by
                )
                raw = self.composite_cache.raw_tensors[layer_name]
                num_blocks = raw.numel() // segment.page_size_bytes
                raw.view(torch.uint8).view(num_blocks, segment.page_size_bytes)[
                    block_id
                ].zero_()

    @staticmethod
    def _multimodal_features(multimodal_input):
        if not multimodal_input:
            return []
        positions = multimodal_input["mm_placeholders"]
        kwargs = multimodal_input["mm_kwargs"]
        hashes = multimodal_input.get("mm_hashes", {})
        features = []
        for modality, index in argsort_mm_positions(positions):
            mm_hash = hashes.get(modality, [None] * len(positions[modality]))[index]
            features.append(
                MultiModalFeatureSpec(
                    data=kwargs[modality][index],
                    modality=modality,
                    identifier=mm_hash or f"{modality}-{index}",
                    mm_position=positions[modality][index],
                    mm_hash=mm_hash,
                )
            )
        return features

    def _prepare_multimodal_embeddings(
        self,
        requests: list[RequestMeta],
        input_tokens: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
    ) -> torch.Tensor | None:
        features: list[MultiModalFeatureSpec] = []
        is_multimodal = torch.zeros(
            input_tokens.numel(), dtype=torch.bool, device=self.device
        )
        for request_index, request in enumerate(requests):
            if request[3] != 0:
                continue
            multimodal_input = request[4]
            request_features = self._multimodal_features(multimodal_input)
            request_start = int(query_start_loc_cpu[request_index])
            for feature in request_features:
                features.append(feature)
                start = request_start + feature.mm_position.offset
                end = start + feature.mm_position.length
                is_multimodal[start:end] = True

        if not features:
            if not getattr(self, "_model_needs_text_embeds", False):
                return None
            if not hasattr(self.model, "embed_input_ids"):
                return None
            return self.model.embed_input_ids(input_tokens)
        if not hasattr(self.model, "embed_multimodal") or not hasattr(
            self.model, "embed_input_ids"
        ):
            raise TypeError(
                f"Model {type(self.model).__name__} received multimodal input but "
                "does not implement vLLM's multimodal embedding interface"
            )

        mm_embeddings = []
        mm_kwargs = [
            (feature.modality, feature.data)
            for feature in features
            if feature.data is not None
        ]
        for _, _, kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=True
        ):
            outputs = self.model.embed_multimodal(**kwargs_batch)
            mm_embeddings.extend(outputs)
        return self.model.embed_input_ids(
            input_tokens,
            multimodal_embeddings=mm_embeddings,
            is_multimodal=is_multimodal,
        )

    def _build_attention_metadata_simplified(
        self,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        num_computed_tokens_cpu: torch.Tensor,
        block_tables: list[torch.Tensor],
        slot_mappings: torch.Tensor,
        total_num_scheduled_tokens: int,
        max_query_len: int,
        num_reqs: int,
        max_seq_len_override: int | None = None,
    ):
        actual_max_seq_len = int(seq_lens_cpu.max().item())
        max_seq_len = (
            actual_max_seq_len if max_seq_len_override is None else max_seq_len_override
        )
        if max_seq_len < actual_max_seq_len:
            raise ValueError(
                "max_seq_len_override cannot be smaller than the batch: "
                f"override={max_seq_len}, actual={actual_max_seq_len}"
            )
        is_prefilling = num_computed_tokens_cpu == 0
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=total_num_scheduled_tokens,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            positions=self._current_positions,
            is_prefilling=is_prefilling,
        )
        # Adaptive SplitKV uses the host max_seq_len to pick a reduction
        # topology. Capture fixes that value at the bucket; eager uses the
        # actual length. A single split keeps both paths numerically aligned.
        return single_split_attention(metadata)

    @torch.inference_mode()
    def swap_blocks(
        self,
        request_ids: list[int],
        source_block_ids: list[int],
        target_block_ids: list[int],
        group_ids: list[int],
        first_kernel_blocks: list[int],
        valid_kernel_blocks: list[int],
        source_event_handles: list[
            list[ray.ObjectRef] | None
        ],  # elem: None or events for all workers
        is_swap_in: bool,
    ):
        """
        Swap the valid portions of some blocks between CPU and GPU.

        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False.

        This method may block the corresponding stream if the source blocks
        aren't ready. The availability of the target blocks is ensured at
        the block manager level.

        Return a CUDA event for testing completion of this operation.
        """
        metadata_lengths = {
            len(source_block_ids),
            len(target_block_ids),
            len(group_ids),
            len(first_kernel_blocks),
            len(valid_kernel_blocks),
        }
        if len(metadata_lengths) != 1:
            raise ValueError(
                "Swap block ids and metadata must have equal lengths: "
                f"source={len(source_block_ids)}, target={len(target_block_ids)}, "
                f"groups={len(group_ids)}, first={len(first_kernel_blocks)}, "
                f"valid={len(valid_kernel_blocks)}"
            )
        for group_id in group_ids:
            if not 0 <= group_id < len(self.cache_group_descriptors):
                raise ValueError(f"Invalid swap cache-group id {group_id}")

        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream
        new_event = torch.cuda.Event(interprocess=True, enable_timing=False)

        # Wait for source blocks if needed
        self._wait_source_events(
            cudaStream_t(stream.cuda_stream),
            new_event,
            request_ids,
            source_event_handles,
        )

        # Swap one continuous extent per backing tensor and live physical page.
        # Full attention uses a prefix on its tail page; sliding-window
        # attention can also use a suffix on its oldest live page. It
        # deliberately does not issue one DMA per kernel page: the A100 PCIe
        # measurements show that split transfers lose throughput to per-copy
        # overhead.
        copied_bytes = 0
        copy_operations = 0
        old_full_page_bytes = (
            len(source_block_ids) * self.composite_cache.page_size_bytes
        )
        with torch.cuda.stream(stream):
            assert self.kv_swap_view.device.type == "cpu"
            for segment_index, segment in enumerate(self.composite_cache.segments):
                raw = self.composite_cache.raw_tensors[segment.layer_name]
                num_gpu_blocks = raw.numel() // segment.page_size_bytes
                gpu_pages = raw.view(torch.uint8).view(
                    num_gpu_blocks, segment.page_size_bytes
                )
                owners = self.cache_segment_group_ids[segment_index]
                for (
                    source_id,
                    target_id,
                    group_id,
                    first_block,
                    valid_blocks,
                ) in zip(
                    source_block_ids,
                    target_block_ids,
                    group_ids,
                    first_kernel_blocks,
                    valid_kernel_blocks,
                ):
                    if group_id not in owners:
                        continue
                    descriptor = self.cache_group_descriptors[group_id]
                    byte_offset, copy_bytes = descriptor.byte_extent_for_kernel_blocks(
                        segment.page_size_bytes,
                        first_block,
                        valid_blocks,
                    )
                    if copy_bytes == 0:
                        continue
                    copied_bytes += copy_bytes
                    copy_operations += 1
                    rank_page_offset = (
                        self.parallel_config.tensor_parallel_rank
                        * self.composite_cache.page_size_bytes
                        + segment.cpu_offset
                        + byte_offset
                    )
                    if is_swap_in:
                        cpu_page = self.kv_swap_view[source_id].narrow(
                            0,
                            rank_page_offset,
                            copy_bytes,
                        )
                        gpu_pages[target_id].narrow(0, byte_offset, copy_bytes).copy_(
                            cpu_page, non_blocking=True
                        )
                    else:
                        cpu_page = self.kv_swap_view[target_id].narrow(
                            0,
                            rank_page_offset,
                            copy_bytes,
                        )
                        cpu_page.copy_(
                            gpu_pages[source_id].narrow(0, byte_offset, copy_bytes),
                            non_blocking=True,
                        )

        # Record event
        new_event.record(stream)

        saved_ratio = (
            0.0
            if old_full_page_bytes == 0
            else 1.0 - copied_bytes / old_full_page_bytes
        )
        logger.info(
            "swap requests %s (%s to %s): copied %d/%d bytes "
            "(saved %.2f%%), %d DMA extents",
            request_ids,
            "CPU" if is_swap_in else "GPU",
            "GPU" if is_swap_in else "CPU",
            copied_bytes,
            old_full_page_bytes,
            saved_ratio * 100,
            copy_operations,
        )
        return reduce_cuda_event(self.cudart, new_event)

    def _wait_source_events(
        self,
        stream: cudaStream_t,
        new_event: torch.cuda.Event,
        request_ids: list[int],
        source_event_handles: list[list[ray.ObjectRef] | None],
    ):
        # XXX: this is potentially a choke point
        futures = [
            handles[self.worker_id]
            for handles in source_event_handles
            if handles is not None
        ]
        try:
            event_handles = ray.get(futures)
        except Exception as e:
            logger.error(f"error getting source events; error: {e}")
            ray.actor.exit_actor()
        i = 0
        for request_id, fut in zip(request_ids, source_event_handles):
            if fut is not None:
                # The request had a move operation going and it may have not finished yet.
                # Wait for it before we do anything with it ourselves (the event may be from
                # a stream on another device).
                event = rebuild_cuda_event(
                    self.cudart, event_handles[i], is_worker=True
                )
                if event is None:
                    local_event = True
                    if request_id not in self.move_event_table:
                        logger.error(
                            f"request {request_id} should have a local event here at {self}"
                        )
                    event = cudaEvent_t(
                        self.move_event_table[request_id][-1].cuda_event
                    )
                else:
                    local_event = False
                self.cudart.cudaStreamWaitEvent(stream, event, 0)
                logger.debug(f"waited for {event}")
                if not local_event:
                    self.cudart.cudaEventDestroy(event)
                i += 1
            self.move_event_table[request_id].append(new_event)

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called when a request is finished or aborted"""
        self.move_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, request_ids: list[int]):
        """Clear the resources associated with the requests."""
        for request_id in request_ids:
            self.clear_request_resource(request_id)

    def wait_for_all_move_in(self):
        """Wait for all ongoing move-in operations to finish."""
        # start = time.time()
        self.swap_in_stream.synchronize()
        # end = time.time()
        # logger.info(f"synchronized move-in ({end-start:.2f}s)")

    def wait_for_all_move_out(self):
        """Wait for all ongoing move-out operations to finish."""
        # start = time.time()
        self.swap_out_stream.synchronize()
        # end = time.time()
        # logger.info(f"synchronized move-out ({end-start:.2f}s)")

    def migrate_blocks(
        self,
        request_ids: list[int],
        source_engine_id: int,
        source_block_ids: list[int],
        target_block_ids: list[int],
        source_events: list[
            list[ray.ObjectRef] | None
        ],  # elem: None or events for all workers
    ):
        raise NotImplementedError()

    def migrate_blocks_source(
        self,
        request_ids: list[int],
        migration_event: ray.ObjectRef,
    ):
        raise NotImplementedError()

    def register_kvcache_mem_handles(
        self,
        prefill_engine_id: int,
        prefill_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles: list[list[cudaIpcMemHandle_t]],
    ):
        raise NotImplementedError()

    def unregister_kvcache_mem_handles(
        self,
        prefill_engine_id: int,
    ):
        raise NotImplementedError()

    def add_model_content(self, model_path: str, handle: "ModelHandle") -> None:
        """Actually load model to memory."""
        if self.quick_loader is not None:
            self.quick_loader.model_contents[model_path] = handle.to_model(
                self.quick_loader.model_cache_start
            )

    def delete_model_content(self, model_path: str) -> bool:
        """Remove reference only."""
        if (
            self.quick_loader is not None
            and model_path in self.quick_loader.model_contents
        ):
            del self.quick_loader.model_contents[model_path]
            return True
        return False

    def release_requests(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self.mrope_deltas.pop(request_id, None)

    def _validate_vllm_parallel_config(self, vllm_config: VllmConfig) -> None:
        """Ensure a switched model can reuse this Worker's process groups."""
        vllm_parallel_config = vllm_config.parallel_config
        exp_tp = self.parallel_config.tensor_parallel_size
        exp_pp = self.parallel_config.pipeline_parallel_size
        cur_tp = vllm_parallel_config.tensor_parallel_size
        cur_pp = vllm_parallel_config.pipeline_parallel_size

        if exp_tp != cur_tp:
            raise ValueError(
                "Cannot switch to a model with a different tensor parallel size: "
                f"worker={exp_tp}, model={cur_tp}"
            )

        if exp_pp != cur_pp:
            raise ValueError(
                "Cannot switch to a model with a different pipeline parallel size: "
                f"worker={exp_pp}, model={cur_pp}"
            )


def _init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(False)

    init_distributed_environment(
        parallel_config.world_size, rank, distributed_init_method, local_rank
    )
