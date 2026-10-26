import torch
import cupy

from quick_model_loader.meta import TensorsContent, ShardingContent, ModelContent


class TensorHandle:
    def __init__(self, tensor: torch.Tensor) -> None:
        assert "cuda" in tensor.device.type
        tensor_uint8 = tensor.view(dtype=torch.uint8)

        device_id = tensor.device.index

        with cupy.cuda.Device(device_id):
            cupy_array = cupy.asarray(tensor_uint8)
            self.handle = cupy.cuda.runtime.ipcGetMemHandle(cupy_array.data.ptr)
            self.cupy_shape = cupy_array.shape
            self.cupy_dtype = cupy_array.dtype
            self.cupy_nbytes = cupy_array.nbytes
            self.device_id = device_id

            self.tensor_device = tensor.device
            self.tensor_dtype = tensor.dtype
            self.tensor_shape = tensor.shape

    def to_tensor(self) -> torch.Tensor:
        with cupy.cuda.Device(self.device_id):
            shared_data = cupy.cuda.runtime.ipcOpenMemHandle(self.handle)
            self.shared_data = shared_data
            memory = cupy.cuda.UnownedMemory(
                shared_data, size=self.cupy_nbytes, owner=None
            )
            memory_ptr = cupy.cuda.MemoryPointer(memory, offset=0)
            shared_cupy_array = cupy.ndarray(
                shape=self.cupy_shape,
                dtype=self.cupy_dtype,
                memptr=memory_ptr,
            )
            tensor_uint8 = torch.as_tensor(
                shared_cupy_array, dtype=torch.uint8, device=self.tensor_device
            )

            tensor = tensor_uint8.view(dtype=self.tensor_dtype).reshape(
                self.tensor_shape
            )


class StorageHandle:
    def __init__(self, storage: torch.UntypedStorage) -> None:
        assert "cuda" in storage.device.type
        tensor_uint8 = (
            torch.tensor([], dtype=torch.uint8, device=storage.device)
            .set_(storage)
            .view(dtype=torch.uint8)
        )

        device_id = storage.device.index
        with cupy.cuda.Device(device_id):
            cupy_array = cupy.asarray(tensor_uint8)
            self.handle = cupy.cuda.runtime.ipcGetMemHandle(cupy_array.data.ptr)
            self.cupy_shape = cupy_array.shape
            self.cupy_dtype = cupy_array.dtype
            self.cupy_nbytes = cupy_array.nbytes
            self.device_id = device_id

            self.storage_device = storage.device

    def to_storage(self) -> torch.UntypedStorage:
        with cupy.cuda.Device(self.device_id):
            shared_data = cupy.cuda.runtime.ipcOpenMemHandle(self.handle)
            self.shared_data = shared_data
            memory = cupy.cuda.UnownedMemory(
                shared_data, size=self.cupy_nbytes, owner=None
            )
            memory_ptr = cupy.cuda.MemoryPointer(memory, offset=0)
            shared_cupy_array = cupy.ndarray(
                shape=self.cupy_shape,
                dtype=self.cupy_dtype,
                memptr=memory_ptr,
            )
            tensor_uint8 = torch.as_tensor(
                shared_cupy_array, dtype=torch.uint8, device=self.storage_device
            )

            storage = tensor_uint8.untyped_storage()
            return storage


class TensorsHandle:
    def __init__(self, tensors_content: TensorsContent) -> None:
        self.tensors_meta = tensors_content.tensors_meta
        self.slice_info = tensors_content.slice_info

    def to_tensors_content(self, storage: torch.UntypedStorage) -> TensorsContent:
        return TensorsContent(storage, self.tensors_meta, self.slice_info)


class ShardingHandle:
    def __init__(self, sharding_content: ShardingContent) -> None:
        assert len(sharding_content.tensors_contents) > 0

        self.id = sharding_content.id
        self.storage_handle = StorageHandle(
            sharding_content.tensors_contents[0]._global_storage
        )
        self.tensors_handles = [
            TensorsHandle(t) for t in sharding_content.tensors_contents
        ]

    def to_sharding_content(self) -> ShardingContent:
        global_storage = self.storage_handle.to_storage()
        tensors_contents = [
            h.to_tensors_content(global_storage) for h in self.tensors_handles
        ]
        return ShardingContent(self.id, tensors_contents)


class ModelHandle:
    def __init__(self, model_content: ModelContent) -> None:
        self.model_name = model_content.model_name
        self.sharding_handles = [
            ShardingHandle(t) for t in model_content.sharding_contents
        ]
        self.checkpoint_config = model_content.checkpoint_config

    def to_model(self) -> ModelContent:
        sharding_contents = [h.to_sharding_content() for h in self.sharding_handles]
        return ModelContent(self.model_name, sharding_contents, self.checkpoint_config)
