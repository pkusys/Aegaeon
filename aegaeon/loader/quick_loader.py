import contextlib
import ctypes
import os
import sys
import threading
import warnings
from collections.abc import Generator

import torch
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.models.registry import ModelRegistry

from aegaeon.allocator import Allocator, get_alloc
from aegaeon.config import QuickLoaderConfig
from aegaeon.logger import init_logger
from aegaeon.utils import CudaRTLibrary

from .allocator import (
    SliceInfo,
)
from .meta import (
    CheckPointConfig,
    ModelContent,
    TensorsContent,
    create_tensor_from_storage,
)

logger = init_logger(__name__)

# Dummy weights share one arena buffer per dtype. The buffer only needs to be
# large enough for the biggest parameter of that dtype because dummy mode does
# not load checkpoint values.
_DUMMY_BUFFER: dict[torch.dtype, tuple[torch.Tensor, int]] = {}
_managed_parameter_state = threading.local()

_OLMO_CONV_LOADER_PATCHED = "_aegaeon_shard_id_forwarded"


def _patch_olmo_fused_conv_shard_loader():
    """Forward the stacked shard id to OLMo's fused conv1d weight loader.

    vLLM maps the checkpoint's separate q_conv1d/k_conv1d/v_conv1d tensors onto
    a single fused ``conv1d.weight`` parameter, tagging each tensor with a
    ``shard_id`` attribute (0/1/2). The generic AutoWeightsLoader calls the
    param's weight_loader without that id, while OLMo's fused-conv loader
    requires it to pick the target section -> dims[None] crash. Wrap the
    factory so the id falls back to the tag on the loaded tensor.
    """
    try:
        from vllm.model_executor.layers.mamba.gdn import (
            olmo_gdn_linear_attn as _gdn,
        )
    except Exception:
        return
    factory = _gdn._make_fused_conv1d_weight_loader
    if getattr(factory, _OLMO_CONV_LOADER_PATCHED, False):
        return

    def _wrapped(dims, tp_size, tp_rank):
        loader = factory(dims, tp_size, tp_rank)

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            shard_id = loaded_shard_id
            if shard_id is None:
                shard_id = getattr(loaded_weight, "shard_id", None)
            if shard_id is None:
                raise ValueError(
                    "Cannot load OLMo fused conv1d weight: no shard id"
                )
            loader(param, loaded_weight, shard_id)

        return weight_loader

    setattr(_wrapped, _OLMO_CONV_LOADER_PATCHED, True)
    _gdn._make_fused_conv1d_weight_loader = _wrapped


@contextlib.contextmanager
def managed_parameter_storage():
    """Route parameters created by QuickLoader into Aegaeon's CUDA arena.

    ``nn.Parameter`` has to be patched before vLLM constructs a model, but the
    patch must not force models loaded by DefaultModelLoader into the arena.
    Keep the patched class storage-preserving by default and enable arena
    allocation only for the duration of ``QuickLoader.load_model``.
    """
    previous = getattr(_managed_parameter_state, "enabled", False)
    _managed_parameter_state.enabled = True
    try:
        yield
    finally:
        _managed_parameter_state.enabled = previous


class ManagedParameter(torch.nn.Parameter):
    """Parameter weights that are set later at copying."""

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)

        # DefaultModelLoader must retain normal torch.nn.Parameter semantics:
        # wrap the tensor it was given instead of silently replacing its
        # storage with an uninitialized slice of Aegaeon's arena.
        if not getattr(_managed_parameter_state, "enabled", False):
            return super().__new__(cls, data, requires_grad)

        alloc = get_alloc()
        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            dtype = data.dtype
            numel = data.numel()
            if dtype not in _DUMMY_BUFFER or _DUMMY_BUFFER[dtype][1] < numel:
                _DUMMY_BUFFER[dtype] = (alloc.allocate((numel,), dtype), numel)
            dummy = _DUMMY_BUFFER[dtype][0][:numel].reshape(data.shape)
            return super().__new__(cls, dummy, requires_grad)

        slice_data = alloc.allocate(data.shape, data.dtype)
        obj = super().__new__(cls, slice_data, requires_grad)
        # Some vLLM modules intentionally construct constants (ones/zeros)
        # that are not guaranteed to be present in every checkpoint.  Preserve
        # those values instead of turning them into uninitialized CUDA bytes.
        if data.device.type != "meta":
            with torch.no_grad():
                obj.copy_(data)
        return obj

    # def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    #     return self.set_(src.untyped_storage()).view(src.dtype).reshape(src.shape)


class QuickLoader(BaseModelLoader):
    """Model loader that uses snapshots from QuickCache."""

    def __init__(
        self,
        cudart: CudaRTLibrary,
        config: QuickLoaderConfig,
        load_config: LoadConfig,
    ):
        super().__init__(load_config)
        if not os.getenv("FAST_SWITCH") == "1":
            raise ValueError("QuickLoader should not be initialized.")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available!")

        # Monkey-patch `nn.Parameter` as `ManagedParameter`
        def patch_parameter_subclasses(old, new):
            import gc

            with warnings.catch_warnings():
                # Warnings here can be ignored, may have deprecated types in gc
                warnings.simplefilter("ignore")
                for obj in gc.get_objects():
                    if isinstance(obj, type):
                        bases = obj.__bases__
                        if old in bases and new is not obj and new not in bases:
                            obj.__bases__ = tuple(new if b is old else b for b in bases)

        patch_parameter_subclasses(torch.nn.Parameter, ManagedParameter)
        patch_parameter_subclasses(torch.nn.parameter.Parameter, ManagedParameter)

        torch.nn.Parameter = ManagedParameter
        torch.nn.parameter.Parameter = ManagedParameter

        whitelist = ["torch", "vllm"]
        for module in sys.modules.values():
            if module.__name__.split(".")[0] in whitelist and hasattr(
                module, "Parameter"
            ):
                module.Parameter = ManagedParameter

        model_cache_filename = config.model_cache_filename
        if not os.path.isfile(model_cache_filename):
            raise ValueError(f"File {config.model_cache_filename} not found")

        model_cache = torch.from_file(
            model_cache_filename,
            shared=True,
            size=config.model_cache_size,
            dtype=torch.uint8,
        )

        cudart.cudaHostRegister(
            ctypes.c_void_p(model_cache.data_ptr()), model_cache.nbytes
        )
        assert model_cache.is_shared() and model_cache.is_pinned()

        model_cache[:] = model_cache[:]

        self.storage = model_cache.untyped_storage()
        self.model_cache_start = model_cache.data_ptr()
        self.model_contents: dict[str, ModelContent] = {}
        for model_name, handle in config.model_cache_snapshot.items():
            self.model_contents[model_name] = handle.to_model(self.model_cache_start)

        # self.pinned_allocator = PinnedAllocator(config.pinned_buffer_size)
        torch.cuda.synchronize()

    def load_model(self, *args, **kwargs):
        """Construct QuickLoader models inside the managed CUDA arena only."""
        _patch_olmo_fused_conv_shard_loader()
        with managed_parameter_storage():
            return super().load_model(*args, **kwargs)

    def load_weights(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
    ) -> torch.nn.Module:
        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            return model

        params_dict = dict(model.named_parameters())
        for _, param in params_dict.items():
            param.detach_()

        tensors_generator = self.load_model_to_tensors(model_config.model)
        loaded_weights = model.load_weights(tensors_generator)

        # Match vLLM's DefaultModelLoader correctness check.  Discarding the
        # returned set allowed new architecture parameters to remain backed by
        # uninitialized GPU memory while startup still reported success.
        if model_config.quantization is None and loaded_weights is not None:
            expected_weights = {name for name, _ in model.named_parameters()}
            missing_weights = expected_weights - loaded_weights

            # Tied checkpoints only store one side of the shared embedding.
            # MiniCPM-S-1B-sft (MiniCPMForCausalLM) stores it on the lm_head
            # side, which vLLM skips, so feed that checkpoint tensor into the
            # missing embed parameter.
            tie_replicated = []
            for name in list(missing_weights):
                dst = params_dict[name]
                if not name.endswith("embed_tokens.weight"):
                    continue
                for key, tensor in self.load_model_to_tensors(
                    model_config.model
                ):
                    if (
                        "lm_head.weight" in key
                        and tensor.dim() == 2
                        and tensor.shape[1] == dst.shape[1]
                        and dst.shape[0] >= tensor.shape[0]
                    ):
                        with torch.no_grad():
                            dst[: tensor.shape[0]].copy_(
                                tensor.to(dtype=dst.dtype, device=dst.device)
                            )
                            if dst.shape[0] > tensor.shape[0]:
                                dst[tensor.shape[0]:].zero_()
                        loaded_weights.add(name)
                        missing_weights.discard(name)
                        tie_replicated.append(name)
                        break
            if tie_replicated:
                logger.info(
                    "tie-replicated %d weight(s) from the checkpoint's "
                    "lm_head side: %s",
                    len(tie_replicated),
                    tie_replicated[:5],
                )

            if missing_weights:
                preview = sorted(missing_weights)
                if len(preview) > 20:
                    preview = preview[:20] + [
                        f"... and {len(missing_weights) - 20} more"
                    ]
                raise ValueError(
                    "QuickLoader did not initialize every model parameter from "
                    f"the checkpoint: {preview}"
                )
            logger.info(
                "QuickLoader verified %d initialized model parameters",
                len(loaded_weights),
            )

        return model

        # TODO: the following is only possible with patched vLLM+QuickLoader,
        # which can save weights that are loader-friendly.

        # for name, weight in tensors_generator:
        #     param = params_dict[name]
        #     if param.dtype != weight.dtype:
        #         param.set_(source=weight.view(param.dtype))
        #     else:
        #         param.set_(source=weight)

    def download_model(self, model_config):
        raise NotImplementedError(
            "download_model is not implemented for QuickLoader. "
            "Please manually download the model and provide the local path."
        )

    def load_model_to_tensors(
        self, model_name: str, checkpoint_config: CheckPointConfig | None = None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if not model_name in self.model_contents:
            raise NotImplementedError(f"{model_name} should be cached")
        cpu_model_content = self.model_contents[model_name]
        if checkpoint_config is None:
            checkpoint_config = cpu_model_content.checkpoint_config

        alloc = get_alloc()
        for sharding_content in cpu_model_content.sharding_contents:
            for tensor_content in sharding_content.tensors_contents:
                alloc.push()
                loaded_tensor_content = self.load_tensors_content(
                    alloc, tensor_content, sharding_content.id
                )

                tensors_storage = loaded_tensor_content.tensors_storage
                tensor_info_map = (
                    loaded_tensor_content.tensors_meta.aligned_tensor_info_map
                )
                for name, tensor_info in tensor_info_map.items():
                    tensor: torch.Tensor = create_tensor_from_storage(
                        tensors_storage, tensor_info
                    )
                    yield name, tensor
                alloc.pop()

    def load_tensors_content(
        self,
        alloc: Allocator,
        tensors_content: TensorsContent,
        sharding_id: int,
    ) -> TensorsContent:
        tensors_meta = tensors_content.tensors_meta
        size = tensors_meta.get_storage_size()
        slice_storage = alloc.allocate((size,), torch.uint8, raw=True)
        slice_info = SliceInfo.with_addr_size(slice_storage.data_ptr(), size)

        loaded_tensors_content = TensorsContent(
            slice_storage, tensors_meta, slice_info, is_global_storage=False
        )

        slice_storage.copy_(tensors_content.tensors_storage, non_blocking=True)

        # pinned_size = int(self.pinned_allocator.get_size() / parallelism)
        # pinned_offset = sharding_id * pinned_size
        # pinned_buffer = self.pinned_allocator.allocate_pinned_buffer(
        #     pinned_size, pinned_offset
        # )

        # offset = 0
        # while offset < size:
        #     bound = min(offset + pinned_size, size)
        #     copy_size = bound - offset

        #     dst = (
        #         torch.tensor([], device=loaded_tensors_content.tensors_storage.device, dtype=torch.uint8)
        #             .set_(loaded_tensors_content.tensors_storage[offset:bound])
        #     )
        #     buf = (
        #         torch.tensor([], device=pinned_buffer.device, dtype=torch.uint8)
        #             .set_(pinned_buffer[:copy_size])
        #     )
        #     src = (
        #         torch.tensor([], device=tensors_content.tensors_storage.device, dtype=torch.uint8)
        #             .set_(tensors_content.tensors_storage[offset:bound])
        #     )

        #     cuts_num = copy_size // (64 * 1024 * 1024)
        #     pipeline_storage_transfer(
        #         dst, buf, src, create_cuts(copy_size, cuts_num))
        #     offset = bound

        return loaded_tensors_content


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(
    model_config: ModelConfig,
) -> tuple[type[torch.nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}"
    )


# def pipeline_storage_transfer(
#     stream: torch.cuda.Stream,
#     cuda_storage: torch.UntypedStorage,
#     pinned_buffer: torch.UntypedStorage,
#     cpu_storage: torch.UntypedStorage,
# ):
#     if not (
#         cuda_storage.nbytes() == pinned_buffer.nbytes()
#         and pinned_buffer.nbytes() == cpu_storage.nbytes()
#     ):
#         raise MemoryError(
#             f"Pipeline transferring requires all storages and buffer have the same size!"
#             f"(cuda: {cuda_storage.nbytes()}, pinned: {pinned_buffer.nbytes()}, cpu: {cpu_storage.nbytes()})"
#         )
#     size = cuda_storage.nbytes()
#     cuts_num = size // (64 * 1024 * 1024)
#     cuts = create_cuts(size, cuts_num)

#     for start, end in cuts:
#         # src = np.frombuffer(((ctypes.c_uint8) * (end-start)).from_address(cpu_storage.data_ptr()+start), dtype=np.uint8)
#         # dst = np.frombuffer(((ctypes.c_uint8) * (end-start)).from_address(pinned_buffer.data_ptr()+start), dtype=np.uint8)
#         # np.copyto(dst, src)
#         pinned_buffer[start:end].copy_(cpu_storage[start:end])

#         cuda_storage[start:end].copy_(pinned_buffer[start:end], non_blocking=True)
#     stream.synchronize()


def create_cuts(size: int, cuts_num: int) -> list[tuple[int, int]]:
    cuts = []
    mod = size % cuts_num
    for i in range(cuts_num):
        if i == 0:
            start = 0
        else:
            start = cuts[i - 1][1]
        if i == cuts_num - 1:
            end = size
        else:
            end = start + size // cuts_num
            if i < mod:
                end += 1
        cuts.append((start, end))
    return cuts
