import torch
import json
import glob
import os
from enum import Enum
from typing import List, Tuple, Dict, Generator, Optional
from quick_model_loader._rlib import read_safetensors_meta_as_json

TORCH_VRAM_ALIGN_ADDR = 16


class ParallelType(Enum):
    TP = 1
    PP = 2

    @classmethod
    def from_str(cls, parallel_str: str):
        if parallel_str == "tp":
            return cls.TP
        elif parallel_str == "pp":
            return cls.PP
        else:
            raise TypeError(
                '''Unsupported parallel type! Currently the parallel type 
                should be either one of "tp" or "pp"'''
            )

    def to_str(self) -> str:
        if self == ParallelType.TP:
            return "tp"
        elif self == ParallelType.PP:
            return "pp"
        else:
            raise TypeError(
                '''Unsupported parallel type! Currently the parallel type 
                should be either one of "tp" or "pp"'''
            )


class QuantizationType(Enum):
    FP8 = 1
    A8W8 = 2

    @classmethod
    def from_str(cls, quantization_str: str):
        if quantization_str == "fp8":
            return cls.FP8
        elif quantization_str == "a8w8":
            return cls.A8W8
        else:
            raise TypeError(
                '''Unsupported quantization type! Currently the  
                quantization type should be either "fp8" or "a8w8"'''
            )

    def to_str(self) -> str:
        if self == QuantizationType.FP8:
            return "fp8"
        elif self == QuantizationType.A8W8:
            return "a8w8"
        else:
            raise TypeError(
                '''Unsupported quantization type! Currently the  
                quantization type should be either "fp8" or "a8w8"'''
            )


class CheckPointConfig:
    dir: str
    parallel_type: ParallelType
    parallel_size: int
    quantization_type: Optional[QuantizationType]

    def __init__(
        self,
        dir: str,
        parallel_type: ParallelType,
        parallel_size: int,
        quantization_type: Optional[QuantizationType] = None,
    ) -> None:
        self.dir = dir
        self.parallel_type = parallel_type
        self.parallel_size = parallel_size
        self.quantization_type = quantization_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CheckPointConfig):
            return NotImplemented
        return (
            self.dir,
            self.parallel_type,
            self.parallel_size,
            self.quantization_type,
        ) == (
            other.dir,
            other.parallel_type,
            other.parallel_size,
            other.quantization_type,
        )

    @property
    def sub_dir(self) -> str:
        if self.quantization_type is None:
            quantization_str = ""
        else:
            quantization_str = self.quantization_type.to_str() + "_"
        return quantization_str + self.parallel_type.to_str() + str(self.parallel_size)


class SliceInfo:
    start: int
    end: int

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    @classmethod
    def with_addr_size(cls, addr: int, size: int):
        start = addr
        end = start + size
        return cls(start, end)

    def get_size(self) -> int:
        return self.end - self.start

    def move(self, dst_addr: int):
        new_start = dst_addr
        new_end = new_start + self.get_size()
        self.start = new_start
        self.end = new_end

    def __eq__(self, other) -> bool:
        if not isinstance(other, SliceInfo):
            return NotImplemented
        return (self.start, self.end) == (
            other.start,
            other.end,
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, SliceInfo):
            return NotImplemented
        return (self.start, self.end) < (other.start, other.end)


class TensorInfo:
    dtype: torch.dtype
    shape: List[int]
    data_offsets: Tuple[int, int]

    def __init__(
        self, dtype: torch.dtype, shape: List[int], data_offsets: Tuple[int, int]
    ) -> None:
        self.dtype: torch.dtype = dtype
        self.shape: List[int] = shape
        self.data_offsets: Tuple[int, int] = data_offsets

    def get_size(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    def __str__(self):
        return f"TensorInfo(dtype={self.dtype}, shape={self.shape}, data_offsets={self.data_offsets})"

    def __repr__(self):
        return self.__str__()


class TensorsMeta:
    file_path: str
    file_size: int
    file_offset: int
    file_tensor_info_map: Dict[str, TensorInfo]
    aligned_tensor_info_map: Dict[str, TensorInfo]
    storage_size: int

    def __init__(
        self,
        file_path: str,
        file_size: int,
        file_offset: int,
        file_tensor_info_map: Dict[str, TensorInfo],
    ) -> None:
        self.file_path = file_path
        self.file_size = file_size
        self.file_offset = file_offset
        self.file_tensor_info_map = file_tensor_info_map
        self.create_aligned_tensor_info_map()

    @classmethod
    def from_json_data(cls, json_data: Dict):
        file_path: str = json_data["file_path"]
        size: int = json_data["size"]
        offset: int = json_data["offset"]
        tensor_info_map: Dict[str, TensorInfo] = {}
        for name, tensor_info_dict in json_data["tensor_info_map"].items():
            dtype_str: str = tensor_info_dict["dtype"]
            dtype: torch.dtype = dtype_str_to_dtype(dtype_str)
            shape: List[int] = tensor_info_dict["shape"]
            data_offsets: Tuple[int, int] = tuple(tensor_info_dict["data_offsets"])
            tensor_info = TensorInfo(dtype, shape, data_offsets)
            tensor_info_map[name] = tensor_info
        return cls(file_path, size, offset, tensor_info_map)

    @classmethod
    def from_tensors_file(cls, st_file_path: str):
        json_string = read_safetensors_meta_as_json(st_file_path)
        json_data = json.loads(json_string)
        return cls.from_json_data(json_data)

    def create_aligned_tensor_info_map(self):
        sorted_tensor_names = sorted(
            list(self.file_tensor_info_map.keys()),
            key=lambda n: self.file_tensor_info_map[n].data_offsets[0],
        )
        aligned_tensor_info_map = {}
        start = 0
        for tensor_name in sorted_tensor_names:
            tensor_info = self.file_tensor_info_map[tensor_name]
            tensor_size = tensor_info.get_size()

            align_tensor_start = (
                (start + TORCH_VRAM_ALIGN_ADDR - 1)
                // TORCH_VRAM_ALIGN_ADDR
                * TORCH_VRAM_ALIGN_ADDR
            )
            align_tensor_end = align_tensor_start + tensor_size
            align_tensor_info = TensorInfo(
                tensor_info.dtype,
                tensor_info.shape,
                (align_tensor_start, align_tensor_end),
            )
            aligned_tensor_info_map[tensor_name] = align_tensor_info

            start = align_tensor_end

        self.aligned_tensor_info_map = aligned_tensor_info_map
        self.storage_size = start

    def get_file_storage_size(self) -> int:
        return self.file_size - self.file_offset

    def get_storage_size(self) -> int:
        return self.storage_size

    def __str__(self):
        return (
            f"SafeTensorsMeta(size={self.file_size}, "
            + f"offset={self.file_offset}, "
            + f"tensor_info_map={self.file_tensor_info_map})"
        )

    def __repr__(self):
        return self.__str__()


class TensorsContent:
    _global_storage: torch.UntypedStorage
    tensors_meta: TensorsMeta
    slice_info: SliceInfo

    def __init__(
        self,
        _global_storage: torch.UntypedStorage,
        tensors_meta: TensorsMeta,
        slice_info: SliceInfo,
    ) -> None:
        if tensors_meta.get_storage_size() != slice_info.get_size():
            raise MemoryError(
                "Invalid pair of tensors storage and tensors meta whose sizes are different!"
            )

        self._global_storage: torch.UntypedStorage = _global_storage
        self.tensors_meta: TensorsMeta = tensors_meta
        self.slice_info: SliceInfo = slice_info

    @property
    def tensors_storage(self) -> torch.UntypedStorage:
        start = self.slice_info.start
        end = self.slice_info.end
        return self._global_storage[start:end]

    def get_storage_size(self) -> int:
        return self.tensors_storage.nbytes()

    def to_tensors(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        tensors_storage = self.tensors_storage
        tensor_info_map = self.tensors_meta.aligned_tensor_info_map
        for name, tensor_info in tensor_info_map.items():
            tensor: torch.Tensor = create_tensor_from_storage(
                tensors_storage, tensor_info
            )
            yield name, tensor


class ShardingMeta:
    id: int
    tensors_metas: List[TensorsMeta]

    def __init__(self, id: int, tensor_files: List[str]) -> None:
        self.id = id
        self.tensors_metas = []
        for tensors_file in tensor_files:
            tensors_meta = TensorsMeta.from_tensors_file(tensors_file)
            self.tensors_metas.append(tensors_meta)

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        id: int = 0,
        checkpoint_config: Optional[CheckPointConfig] = None,
    ):
        if checkpoint_config is None and id != 0:
            raise ValueError(
                "There should be only one sharding for a none-checkpoint "
                + f"model. However, current sharding id is {id}"
            )
        elif checkpoint_config is not None and id >= checkpoint_config.parallel_size:
            raise ValueError(
                "Given sharding id is larger than checkpoint's parallel size."
                + f"parallel size: {checkpoint_config.parallel_size}, sharding id: {id}"
            )
        pattern = "*.safetensors"
        if checkpoint_config is not None:
            checkpoint_path = os.path.join(
                model_path,
                checkpoint_config.dir,
                checkpoint_config.sub_dir,
                f"rank{id}",
            )
            files_pattern = os.path.join(checkpoint_path, pattern)
        else:
            files_pattern = os.path.join(model_path, pattern)
        safetensor_files = glob.glob(files_pattern)
        safetensor_files.sort()
        return cls(id, safetensor_files)

    def get_storage_size(self) -> int:
        storage_size = 0

        for tensors_meta in self.tensors_metas:
            storage_size += tensors_meta.get_storage_size()

        return storage_size

    def get_slice_sizes(self) -> List[int]:
        slice_sizes = []
        for tensors_meta in self.tensors_metas:
            size = tensors_meta.get_storage_size()
            slice_sizes.append(size)
        return slice_sizes


class ShardingContent:
    id: int
    tensors_contents: List[TensorsContent]

    def __init__(self, id: int, tensors_contents: List[TensorsContent]) -> None:
        self.id = id
        self.tensors_contents = tensors_contents

    def get_storage_size(self) -> int:
        storage_size = 0
        for tensors_content in self.tensors_contents:
            storage_size += tensors_content.get_storage_size()
        return storage_size

    def to_tensors(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        for tensors_content in self.tensors_contents:
            for tensor_name, tensor in tensors_content.to_tensors():
                yield tensor_name, tensor


class ModelMeta:
    model_path: str
    sharding_metas: List[ShardingMeta]
    checkpoint_config: Optional[CheckPointConfig]

    def __init__(
        self,
        model_path: str,
        sharding_metas: List[ShardingMeta],
        checkpoint_config: Optional[CheckPointConfig] = None,
    ) -> None:
        self.model_path: str = model_path
        self.sharding_metas = sharding_metas
        self.checkpoint_config = checkpoint_config

    @classmethod
    def from_model_path(
        cls, model_path: str, checkpoint_config: Optional[CheckPointConfig] = None
    ):
        sharding_metas = []
        if checkpoint_config is None:
            sharding_metas.append(ShardingMeta.from_model_path(model_path))
        else:
            for i in range(checkpoint_config.parallel_size):
                sharding_metas.append(
                    ShardingMeta.from_model_path(model_path, i, checkpoint_config)
                )
        return cls(model_path, sharding_metas, checkpoint_config)

    def get_storage_size(self) -> int:
        storage_size = 0

        for sharding_meta in self.sharding_metas:
            storage_size += sharding_meta.get_storage_size()

        return storage_size

    def get_slice_sizes(self) -> List[int]:
        slice_sizes = []

        for sharding_meta in self.sharding_metas:
            slice_sizes += sharding_meta.get_slice_sizes()

        return slice_sizes


class ModelContent:
    model_name: str
    sharding_contents: List[ShardingContent]
    checkpoint_config: Optional[CheckPointConfig]

    def __init__(
        self,
        model_name: str,
        sharding_contents: List[ShardingContent],
        checkpoint_config: Optional[CheckPointConfig] = None,
    ) -> None:
        self.model_name: str = model_name
        self.sharding_contents: List[ShardingContent] = sharding_contents
        self.checkpoint_config = checkpoint_config

    def get_storage_size(self) -> int:
        storage_size = 0
        for sharding_content in self.sharding_contents:
            storage_size += sharding_content.get_storage_size()
        return storage_size

    def to_tensors(
        self, sharding_id: Optional[int] = None
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        if sharding_id is None:
            for sharding_content in self.sharding_contents:
                for tensor_name, tensor in sharding_content.to_tensors():
                    yield tensor_name, tensor
        else:
            if sharding_id >= len(self.sharding_contents):
                raise RuntimeError("Try to tensorlize not existed sharding contents!")
            sharding_content = self.sharding_contents[sharding_id]
            for tensor_name, tensor in sharding_content.to_tensors():
                yield tensor_name, tensor


def create_tensor_from_storage(
    storage: torch.UntypedStorage,
    tensor_info: TensorInfo,
) -> torch.Tensor:
    start_idx = tensor_info.data_offsets[0]
    end_idx = tensor_info.data_offsets[1]
    storage_slice = storage[start_idx:end_idx]
    tensor = (
        torch.tensor([], dtype=torch.uint8, device=storage.device)
        .set_(storage_slice)
        .view(dtype=tensor_info.dtype)
        .reshape(tensor_info.shape)
    )

    return tensor


def dtype_str_to_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "BOOL":
        return torch.bool
    elif dtype_str == "U8":
        return torch.uint8
    elif dtype_str == "I8":
        return torch.int8
    elif dtype_str == "F8_E5M2":
        try:
            return torch.float8_e5m2
        except:
            raise TypeError("Your torch version doesn't support type float8_e5m2")
    elif dtype_str == "F8_E4M3":
        try:
            return torch.float8_e4m3fn
        except:
            raise TypeError("Your torch version doesn't support float8_e4m3fn")
    elif dtype_str == "I16":
        return torch.int16
    elif dtype_str == "U16":
        try:
            return torch.uint16
        except:
            raise TypeError("Your torch version doesn't support type uint16")
    elif dtype_str == "F16":
        return torch.float16
    elif dtype_str == "BF16":
        try:
            return torch.bfloat16
        except:
            raise TypeError("Your torch version doesn't support type bfloat16")
    elif dtype_str == "I32":
        return torch.int32
    elif dtype_str == "U32":
        try:
            return torch.uint32
        except:
            raise TypeError("Your torch version doesn't support type uint32")
    elif dtype_str == "F32":
        return torch.float32
    elif dtype_str == "F64":
        return torch.float64
    elif dtype_str == "I64":
        return torch.int64
    elif dtype_str == "U64":
        try:
            return torch.uint64
        except:
            raise TypeError("Your torch version doesn't support type uint64")
    else:
        raise TypeError(f"Your torch version doesn't support type {dtype_str}")
