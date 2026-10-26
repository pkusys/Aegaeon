import torch
import ctypes

from .meta import TensorsContent, ShardingContent, ModelContent

from aegaeon.logger import init_logger

logger = init_logger(__name__)


class StorageHandle:
    def __init__(
        self,
        model_cache_start: int,
        storage: torch.UntypedStorage,
    ) -> None:
        assert storage.device.type == "cpu"
        self.ptr_ofs: int = storage.data_ptr() - model_cache_start
        self.size: int = storage.nbytes()

    def to_storage(
        self,
        model_cache_start: int,
    ) -> torch.UntypedStorage:
        arr = (ctypes.c_uint8 * self.size).from_address(
            self.ptr_ofs + model_cache_start
        )
        tensor = torch.frombuffer(arr, dtype=torch.uint8)
        return tensor.untyped_storage()


class TensorsHandle:
    def __init__(self, tensors_content: TensorsContent) -> None:
        self.tensors_meta = tensors_content.tensors_meta
        self.slice_info = tensors_content.slice_info

    def to_tensors_content(self, storage: torch.UntypedStorage) -> TensorsContent:
        return TensorsContent(storage, self.tensors_meta, self.slice_info)


class ShardingHandle:
    def __init__(
        self,
        model_cache_start: int,
        sharding_content: ShardingContent,
    ) -> None:
        assert len(sharding_content.tensors_contents) > 0
        assert sharding_content.tensors_contents[0].is_global_storage

        self.id = sharding_content.id
        self.storage_handle = StorageHandle(
            model_cache_start, sharding_content.tensors_contents[0]._storage
        )
        self.tensors_handles = [
            TensorsHandle(t) for t in sharding_content.tensors_contents
        ]

    def to_sharding_content(
        self,
        model_cache_start: int,
    ) -> ShardingContent:
        global_storage = self.storage_handle.to_storage(model_cache_start)
        tensors_contents = [
            h.to_tensors_content(global_storage) for h in self.tensors_handles
        ]
        return ShardingContent(self.id, tensors_contents)


class ModelHandle:
    def __init__(
        self,
        model_cache_start: int,
        model_content: ModelContent,
    ) -> None:
        self.model_name = model_content.model_name
        self.sharding_handles = [
            ShardingHandle(model_cache_start, t)
            for t in model_content.sharding_contents
        ]
        self.checkpoint_config = model_content.checkpoint_config

    def to_model(
        self,
        model_cache_start: int,
    ) -> ModelContent:
        sharding_contents = [
            h.to_sharding_content(model_cache_start) for h in self.sharding_handles
        ]
        return ModelContent(self.model_name, sharding_contents, self.checkpoint_config)

    def get_max_loading_scratch_size(self) -> int:
        """Return the largest temporary GPU slice used by QuickLoader.

        QuickLoader copies one cached safetensors file into a temporary GPU
        slice while ``model.load_weights`` consumes its tensors. The slice is
        released before the next file is copied, so peak scratch space is the
        largest individual tensors file rather than the whole checkpoint.
        """
        return max(
            (
                tensors_handle.tensors_meta.get_storage_size()
                for sharding_handle in self.sharding_handles
                for tensors_handle in sharding_handle.tensors_handles
            ),
            default=0,
        )
