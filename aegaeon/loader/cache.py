import torch
import torch.nn
import os
import pickle
import glob
import ctypes
from typing import Optional, Dict, Any
from .meta import (
    TensorsMeta,
    TensorsContent,
    ShardingMeta,
    ShardingContent,
    ModelContent,
    ModelMeta,
    CheckPointConfig,
    ParallelType,
)
from .handle import ModelHandle
from .allocator import (
    CPUAllocator,
)
from vllm.tokenizers import get_tokenizer

from aegaeon.logger import init_logger
from aegaeon.utils import CudaRTLibrary

logger = init_logger(__name__)


class QuickCache:

    cpu_allocator: CPUAllocator
    cached_models: Dict[str, ModelHandle]
    cached_tokenizers: Dict[str, Any]

    def __init__(
        self,
        cudart: CudaRTLibrary,
        model_cache_filename: str,
        model_cache_size: int,
        model_cache_snapshot: Optional[str] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA decices avaliable!")

        if not os.path.isfile(model_cache_filename):
            logger.info(
                f"Creating the shared CPU cache from scratch at {model_cache_filename}. "
                "This might take some time."
            )
        else:
            logger.info(f"Found shared model cache at {model_cache_filename}")

        self.model_cache_size = model_cache_size
        self.model_cache_filename = model_cache_filename
        storage = torch.from_file(
            model_cache_filename, shared=True, size=model_cache_size, dtype=torch.uint8
        ).untyped_storage()

        # cudart.cudaHostRegister(ctypes.c_void_p(storage.data_ptr()), storage.nbytes())

        # Tensor.is_pinned() no longer use CUDA info to check if the storage is pinned
        assert storage.is_shared() # and storage.is_pinned()

        self.cpu_allocator = CPUAllocator(model_cache_size, storage)
        self.model_cache_start = storage.data_ptr()
        self.cached_models: Dict[str, ModelHandle] = {}
        self.cached_model_names: Dict[str, str] = {}
        self.cached_tokenizers = {}
        self.model_cache_snapshot_filename = model_cache_snapshot
        if model_cache_snapshot is not None:
            raise NotImplementedError()
            if os.path.isfile(model_cache_filename):
                with open(model_cache_snapshot, "r") as f:
                    logger.info(
                        f"Found previously cached snapshot metadata at {model_cache_snapshot}"
                    )
                    self.cached_models = pickle.load(f)  # Dict[str, ModelHandle]
                    logger.info(f"Cached models: {list(self.cached_models.keys())}")

    def clear_cached_models(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def evict_cached_model(self, model_name: str):
        raise NotImplementedError()

    def evict_lru_model(self) -> str:
        raise NotImplementedError()

    def cache_model(
        self,
        model_name: str,
        model_path: str,
        checkpoint_config: Optional[CheckPointConfig] = None,
    ):
        # if checkpoint_config is None:
        #     checkpoint_config = CheckPointConfig(
        #         model_name, ParallelType.TP, 1, quantization_type=None
        #     )

        if model_path in self.cached_models:
            cached_model = self.cached_models[model_path]
            if cached_model.checkpoint_config == checkpoint_config:
                # cache hit
                return
            else:
                self.evict_cached_model(model_path)

        model_meta = ModelMeta.from_model_path(model_path, checkpoint_config)
        if model_meta is not None:
            if not self.cpu_allocator.can_allocate_model(model_meta):
                raise MemoryError("Insufficient memory to allocate model!")

            sharding_contents = []
            for sharding_meta in model_meta.sharding_metas:
                sharding_content = self.cache_sharding_content(sharding_meta)
                sharding_contents.append(sharding_content)

            name = model_meta.model_path

            model_content: ModelContent = ModelContent(
                name, sharding_contents, checkpoint_config
            )
        else:
            raise NotImplementedError('No *.safetensor files found.')

        self.cached_models[name] = ModelHandle(self.model_cache_start, model_content)
        self.cached_model_names[name] = model_name
        self.cached_tokenizers[name] = get_tokenizer(name, trust_remote_code=True)
        # self.cached_models[name] = model_content
        logger.info(f"quick loader: cached {model_name}")
    
    def uncache_model(self, model_name: str, model_path: str) -> bool:
        """Free all CPU cache slabs for a model and remove it from cached_models."""
        if model_path not in self.cached_models:
            return False
        model_handle = self.cached_models[model_path]
        model_content = model_handle.to_model(self.model_cache_start)
        for sharding in model_content.sharding_contents:
            for tensors in sharding.tensors_contents:
                self.cpu_allocator.free(tensors)
        del self.cached_models[model_path]
        self.cached_model_names.pop(model_path)
        self.cached_tokenizers.pop(model_path, None)
        logger.info(f"quick loader: uncached {model_name}")
        return True
    
    def defrag(self):
        """Compact CPU cache by moving slices."""
        self.cpu_allocator.fragment_collection()
        return self.cached_models.copy()

    def get_tokenizer(self, name: str):
        return self.cached_tokenizers[name]

    def cache_sharding_content(self, sharding_meta: ShardingMeta) -> ShardingContent:
        tensors_contents = []

        for tensors_meta in sharding_meta.tensors_metas:
            tensors_content = self.cache_tensors_content(tensors_meta)
            tensors_contents.append(tensors_content)

        return ShardingContent(sharding_meta.id, tensors_contents)

    def cache_tensors_content(self, tensors_meta: TensorsMeta) -> TensorsContent:
        file_buffer: torch.UntypedStorage = torch.UntypedStorage.from_file(
            tensors_meta.file_path, shared=False, nbytes=tensors_meta.file_size
        ).untyped()[tensors_meta.file_offset :]

        tensors_content = self.cpu_allocator.allocate(tensors_meta)
        load_tensors_file_buffer(tensors_content, file_buffer)

        return tensors_content

    def snapshot(self) -> Dict[str, ModelHandle]:
        if self.model_cache_snapshot_filename is not None:
            raise NotImplementedError()
            # Save metadata of the cached models
            with open(self.model_cache_snapshot_filename, "w") as f:
                pickle.dump(self.cached_models, f)
        return self.cached_models


def load_tensors_file_buffer(
    tensors_content: TensorsContent,
    file_buffer: torch.UntypedStorage,
):
    tensors_storage = tensors_content.tensors_storage
    tensors_meta = tensors_content.tensors_meta
    sorted_tensor_names = sorted(
        list(tensors_meta.file_tensor_info_map.keys()),
        key=lambda n: tensors_meta.file_tensor_info_map[n].data_offsets[0],
    )
    for tensor_name in sorted_tensor_names:
        file_offsets = tensors_meta.file_tensor_info_map[tensor_name].data_offsets
        alligned_offsets = tensors_meta.aligned_tensor_info_map[
            tensor_name
        ].data_offsets
        i0, i1 = file_offsets[0], file_offsets[1]
        j0, j1 = alligned_offsets[0], alligned_offsets[1]
        tensors_storage[j0:j1].copy_(file_buffer[i0:i1])
