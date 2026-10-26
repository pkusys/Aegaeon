import torch
from typing import Optional, List, Tuple, Generator, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from quick_model_loader.meta import (
    TensorsMeta,
    TensorsContent,
    ShardingMeta,
    ShardingContent,
    ModelContent,
    ModelMeta,
    CheckPointConfig,
)
from quick_model_loader.allocator import (
    CUDAllocator,
    PinnedAllocator,
    CPUAllocator,
)
from quick_model_loader.utils import LRUDict


class QuickModelLoader:
    cuda_allocators: List[CUDAllocator]
    pinned_allocator: PinnedAllocator
    cpu_allocator: CPUAllocator

    loaded_model: Optional[ModelContent]
    cached_models: LRUDict[str, ModelContent]

    def __init__(
        self,
        cuda_storage_sizes: List[int],
        pinned_buffer_size: int,
        models_storage_size: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA decices avaliable!")

        if len(cuda_storage_sizes) > torch.cuda.device_count():
            required_cuda_devices = len(cuda_storage_sizes)
            avaliable_cuda_devices = torch.cuda.device_count()
            raise RuntimeError(
                "Insufficient CUDA devices! "
                + f"Requried: {required_cuda_devices}, Avaliable: {avaliable_cuda_devices}"
            )

        self.cuda_allocators = []
        for i, cuda_storage_size in enumerate(cuda_storage_sizes):
            self.cuda_allocators.append(CUDAllocator(cuda_storage_size, i))

        self.pinned_allocator = PinnedAllocator(pinned_buffer_size)
        self.cpu_allocator = CPUAllocator(models_storage_size)

        self.loaded_model = None
        self.cached_models = LRUDict()

    @property
    def lru_model(self) -> Optional[str]:
        model_name, _ = self.cached_models.lru()
        return model_name

    @classmethod
    def with_models(
        cls,
        cached_models_paths: List[str],
        loaded_model_path: Optional[str] = None,
        cuda_storage_sizes: List[int] = [],
        pinned_buffer_size: Optional[int] = None,
        models_storage_size: Optional[int] = None,
        checkpoint_config: Union[
            None, CheckPointConfig, Dict[str, CheckPointConfig]
        ] = None,
    ):
        model_sizes = []
        for cached_model_path in cached_models_paths:
            if isinstance(checkpoint_config, dict):
                _checkpoint_config = checkpoint_config[cached_model_path]
            else:
                _checkpoint_config = checkpoint_config

            model_meta = ModelMeta.from_model_path(
                cached_model_path, _checkpoint_config
            )
            model_size = model_meta.get_storage_size()
            model_sizes.append(model_size)

        if not loaded_model_path is None:
            if isinstance(checkpoint_config, dict):
                _checkpoint_config = checkpoint_config[loaded_model_path]
            else:
                _checkpoint_config = checkpoint_config

            model_meta = ModelMeta.from_model_path(
                loaded_model_path, _checkpoint_config
            )
            model_size = model_meta.get_storage_size()
            model_sizes.append(model_size)

        if cuda_storage_sizes == []:
            cuda_storage_sizes = [int(max(model_sizes) / 0.8)]

        if pinned_buffer_size is None:
            pinned_buffer_size = max(int(min(model_sizes)), 5 * 1024 * 1024 * 1024)

        if models_storage_size is None:
            models_storage_size = int(sum(model_sizes) / 0.8)

        model_loader = cls(cuda_storage_sizes, pinned_buffer_size, models_storage_size)

        for cached_model_path in cached_models_paths:
            if isinstance(checkpoint_config, dict):
                _checkpoint_config = checkpoint_config[cached_model_path]
            else:
                _checkpoint_config = checkpoint_config

            model_meta = ModelMeta.from_model_path(
                cached_model_path, _checkpoint_config
            )
            if not model_loader.cpu_allocator.can_allocate_model(model_meta):
                continue
            else:
                model_loader.cache_model(cached_model_path, _checkpoint_config)

        if not loaded_model_path is None:
            if isinstance(checkpoint_config, dict):
                _checkpoint_config = checkpoint_config[loaded_model_path]
            else:
                _checkpoint_config = checkpoint_config

            model_loader.load_model(loaded_model_path, _checkpoint_config)
        return model_loader

    def clear_loaded_model(self):
        for cuda_allocator in self.cuda_allocators:
            cuda_allocator.clear()
        self.loaded_model = None

    def clear_cached_models(self):
        self.cpu_allocator.clear()
        self.cached_models.clear()

    def clear(self):
        self.clear_loaded_model()
        self.clear_cached_models()

    def evict_cached_model(self, model_name: str):
        if not model_name in self.cached_models:
            return

        model_content = self.cached_models[model_name]

        for sharding_content in model_content.sharding_contents:
            for tensors_content in sharding_content.tensors_contents:
                self.cpu_allocator.free(tensors_content)

        del self.cached_models[model_name]

    def evict_lru_model(self) -> str:
        if len(self.cached_models) == 0:
            return
        lru_model_name, _ = self.cached_models.lru()

        self.evict_cached_model(lru_model_name)

        return lru_model_name

    def cache_model(
        self, model_name: str, checkpoint_config: Optional[CheckPointConfig] = None
    ):
        if model_name in self.cached_models:
            cached_model = self.cached_models[model_name]
            if cached_model.checkpoint_config == checkpoint_config:
                return
            else:
                self.evict_cached_model(model_name)

        model_meta = ModelMeta.from_model_path(model_name, checkpoint_config)
        if not self.cpu_allocator.can_allocate_model_when_empty(model_meta):
            raise MemoryError("Insufficient memory to allocate model!")

        while not self.cpu_allocator.can_allocate_model(model_meta):
            self.cpu_allocator.fragment_collection()
            if not self.cpu_allocator.can_allocate_model(model_meta):
                self.evict_lru_model()

        sharding_contents = []
        for sharding_meta in model_meta.sharding_metas:
            sharding_content = self.cache_sharding_content(sharding_meta)
            sharding_contents.append(sharding_content)

        name = model_meta.model_path

        model_content: ModelContent = ModelContent(
            name, sharding_contents, checkpoint_config
        )
        self.cached_models[name] = model_content

    def load_model(
        self, model_name: str, checkpoint_config: Optional[CheckPointConfig] = None
    ):
        if (
            self.loaded_model != None
            and self.loaded_model.model_name == model_name
            and self.loaded_model.checkpoint_config == checkpoint_config
        ):
            return

        if not model_name in self.cached_models:
            self.cache_model(model_name, checkpoint_config)
        else:
            cached_model = self.cached_models[model_name]
            if cached_model.checkpoint_config != checkpoint_config:
                self.cache_model(model_name, checkpoint_config)

        self.clear_loaded_model()

        cached_model = self.cached_models[model_name]
        loaded_sharding_contents = []
        shardings_num = len(cached_model.sharding_contents)

        with ThreadPoolExecutor(max_workers=shardings_num) as loading_executor:
            loading_futures = [
                loading_executor.submit(
                    self.load_sharding_content, cached_sharding_content, shardings_num
                )
                for cached_sharding_content in cached_model.sharding_contents
            ]
            for loading_future in loading_futures:
                loaded_sharding_content = loading_future.result()
                loaded_sharding_contents.append(loaded_sharding_content)

        loaded_model_content = ModelContent(
            model_name, loaded_sharding_contents, checkpoint_config=checkpoint_config
        )
        self.loaded_model = loaded_model_content

    def cache_sharding_content(self, sharding_meta: ShardingMeta) -> ShardingContent:
        tensors_contents = []

        for tensors_meta in sharding_meta.tensors_metas:
            tensors_content = self.cache_tensors_content(tensors_meta)
            tensors_contents.append(tensors_content)

        return ShardingContent(sharding_meta.id, tensors_contents)

    def load_sharding_content(
        self, sharding_content: ShardingContent, parallelism: int
    ) -> ShardingContent:
        loaded_tensors_contents = []

        for tensors_content in sharding_content.tensors_contents:
            loaded_tensor_content = self.load_tensors_content(
                tensors_content, sharding_content.id, parallelism
            )
            loaded_tensors_contents.append(loaded_tensor_content)

        return ShardingContent(sharding_content.id, loaded_tensors_contents)

    def cache_tensors_content(self, tensors_meta: TensorsMeta) -> TensorsContent:
        file_buffer: torch.UntypedStorage = torch.UntypedStorage.from_file(
            tensors_meta.file_path, shared=False, nbytes=tensors_meta.file_size
        ).untyped()[tensors_meta.file_offset :]

        tensors_content = self.cpu_allocator.allocate(tensors_meta)
        load_tensors_file_buffer(tensors_content, file_buffer)

        return tensors_content

    def load_tensors_content(
        self, tensors_content: TensorsContent, sharding_id: int, parallelism: int
    ) -> TensorsContent:
        tensors_meta = tensors_content.tensors_meta

        loaded_tensors_content = self.cuda_allocators[sharding_id].allocate(
            tensors_meta
        )

        size = tensors_meta.get_storage_size()
        pinned_size = int(self.pinned_allocator.get_size() / parallelism)
        pinned_offset = sharding_id * pinned_size
        pinned_buffer = self.pinned_allocator.allocate_pinned_buffer(
            pinned_size, pinned_offset
        )

        offset = 0
        while offset < size:
            bound = min(offset + pinned_size, size)
            copy_size = bound - offset
            pipeline_storage_transfer(
                loaded_tensors_content.tensors_storage[offset:bound],
                pinned_buffer[:copy_size],
                tensors_content.tensors_storage[offset:bound],
            )
            offset = bound
        return loaded_tensors_content

    def model_to_tensors(
        self,
        model_name: str,
        device: Union[str, torch.device],
        sharding_id: Optional[int] = None,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        if isinstance(device, str):
            device = torch.device(device)

        if self.loaded_model is not None and self.loaded_model.model_name == model_name:
            tensors_generator = self.loaded_model.to_tensors(sharding_id)
        else:
            if not model_name in self.cached_models:
                self.cache_model(model_name)
            tensors_generator = self.cached_models[model_name].to_tensors(sharding_id)

        for name, tensor in tensors_generator:
            if tensor.device != device:
                tensor_dst = torch.empty_like(tensor, device=device)
                dst_storage = tensor_dst.untyped_storage()
                src_storage = tensor.untyped_storage()
                size = dst_storage.nbytes()
                pinned_buffer = self.pinned_allocator.allocate_pinned_buffer(size)

                pinned_buffer.copy_(src_storage)
                dst_storage.copy_(pinned_buffer)

                yield name, tensor_dst
            else:
                yield name, tensor


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


def pipeline_storage_transfer(
    cuda_storage: torch.UntypedStorage,
    pinned_buffer: torch.UntypedStorage,
    cpu_storage: torch.UntypedStorage,
):
    if not (
        cuda_storage.nbytes() == pinned_buffer.nbytes()
        and pinned_buffer.nbytes() == cpu_storage.nbytes()
    ):
        raise MemoryError(
            "Pipeline transferring requires all straoges and buffer have the same size!"
        )
    size = cuda_storage.nbytes()
    cuts_num = 1 if size <= 1024 * 1024 * 1024 else 16
    cuts = create_cuts(size, cuts_num)

    stream = torch.cuda.current_stream(cuda_storage.device)
    for start, end in cuts:
        pinned_buffer[start:end].copy_(cpu_storage[start:end], non_blocking=True)
        stream.synchronize()
        cuda_storage[start:end].copy_(pinned_buffer[start:end], non_blocking=True)
    stream.synchronize()


def create_cuts(size: int, cuts_num: int) -> List[Tuple[int, int]]:
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
