import torch
from enum import Enum
from typing import List, Optional, Union
from bisect import bisect_left
from .meta import (
    TensorsContent,
    TensorsMeta,
    ModelMeta,
    SliceInfo,
    TORCH_VRAM_ALIGN_ADDR,
)


class DeviceType(Enum):
    CUDA = 1
    PINNED = 2
    CPU = 3


class Device:
    def __init__(
        self, device_type: DeviceType, device_id: Optional[int] = None
    ) -> None:
        self.device_type = device_type
        self.device_id = device_id

    @classmethod
    def from_str(cls, device_str: str):
        if device_str.startswith("cuda"):
            if ":" in device_str:
                _, device_id = device_str.split(":")
                return cls(DeviceType.CUDA, int(device_id))
            else:
                return cls(DeviceType.CUDA)
        elif device_str == "pinned":
            return cls(DeviceType.PINNED)
        elif device_str == "cpu":
            return cls(DeviceType.CPU)
        else:
            raise TypeError(
                """Unsupported device type! Currently the device type 
                should be either "cuda", "pinned", "cpu", or "cuda:X"."""
            )

    def to_torch_device_str(self) -> str:
        if self.device_type == DeviceType.CUDA:
            return f"cuda:{self.device_id}" if self.device_id is not None else "cuda"
        else:
            return "cpu"

    def is_pinned(self) -> bool:
        if self.device_type == DeviceType.PINNED:
            return True
        else:
            return False


class MemoryTable:
    size: int
    table: List[SliceInfo]

    def __init__(self, size: int) -> None:
        self.size = size
        self.table = []

    def clear(self):
        self.table.clear()

    def append(self, slice_info: SliceInfo):
        # Check whether the appending is legal or not
        if slice_info.start < 0:
            raise MemoryError("Illegal memory allocation! Memory address should >= 0!")
        if slice_info.end > self.size:
            raise MemoryError("Illegal memory allocation! Exceed memory size!")
        if slice_info.start % TORCH_VRAM_ALIGN_ADDR != 0:
            raise MemoryError("Illegal memory allocation! Memory not alligned!")

        index = get_insert_index(self.table, slice_info)

        if index is None:
            raise MemoryError("Illegal memory allocation! Memory has been allocated!")

        self.table.insert(index, slice_info)

    def delete(self, slice: SliceInfo):
        index = self.search_allocated(slice)
        # Check whether the deletion is legal or not
        if index == None:
            raise MemoryError("Try to free unallocated memory!")

        del self.table[index]

    def move(self, slice_info: SliceInfo, dst_addr: int):
        if not self.can_move(slice_info, dst_addr):
            raise RuntimeError("Illegal slice movement!")

        self.delete(slice_info)
        slice_info.move(dst_addr)
        self.append(slice_info)

    def search_allocated(self, slice_info: SliceInfo) -> Optional[int]:
        index = bisect_left(self.table, slice_info)

        if index == len(self.table):
            return None

        if self.table[index] != slice_info:
            return None

        return index

    def find_free_addr(self, size: int) -> Optional[int]:
        # Try to allocate space between the slice hole
        for i in range(len(self.table) - 1):
            start = self.table[i].end
            end = self.table[i + 1].start
            addr = find_feasible_addr(start, end, size)
            if addr != None:
                return addr

        # Try to allocate space after the last slice or from the very beginning
        if len(self.table) == 0:
            start = 0
        else:
            start = self.table[-1].end
        end = self.size
        addr = find_feasible_addr(start, end, size)

        return addr

    def can_allocate(self, slice_sizes: List[int]) -> bool:
        total_size = sum(slice_sizes)
        if total_size > self.get_free_size():
            return False

        original_table = [t for t in self.table]
        can_allocate = True

        for slice_size in slice_sizes:
            addr = self.find_free_addr(slice_size)
            if addr == None:
                can_allocate = False
                break
            else:
                slice_info = SliceInfo.with_addr_size(addr, slice_size)
                self.append(slice_info)

        self.table = original_table
        return can_allocate

    def can_allocate_when_empty(self, slice_sizes: List[int]) -> bool:
        required_size = 0

        for size in slice_sizes:
            addr = (
                (required_size + TORCH_VRAM_ALIGN_ADDR - 1)
                // TORCH_VRAM_ALIGN_ADDR
                * TORCH_VRAM_ALIGN_ADDR
            )
            required_size = addr + size

        return self.size >= required_size

    def can_move(self, slice_info: SliceInfo, dst_addr: int) -> bool:
        if dst_addr < 0:
            return False

        if dst_addr % TORCH_VRAM_ALIGN_ADDR != 0:
            return False

        size = slice_info.get_size()
        if dst_addr + size > self.size:
            return False

        index = self.search_allocated(slice_info)
        if index is None:
            return False

        table = [t for t in self.table]
        del table[index]

        size = slice_info.get_size()
        new_slice_info = SliceInfo.with_addr_size(dst_addr, size)
        new_index = get_insert_index(table, new_slice_info)
        if new_index is None:
            return False

        return True

    def get_allocated_size(self) -> int:
        allocated_size = 0
        for slice_info in self.table:
            allocated_size += slice_info.get_size()
        return allocated_size

    def get_free_size(self) -> int:
        return self.size - self.get_allocated_size()

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index) -> SliceInfo:
        return self.table[index]


class Allocator:
    storage: torch.UntypedStorage
    device: Device
    memory_table: MemoryTable

    def __init__(
        self,
        size: int,
        device_or_stoage: Union[str, Device, torch.UntypedStorage],
    ) -> None:
        if size <= 0:
            raise MemoryError(
                "The storage size should > 0 when initialize an Allocator"
            )

        if isinstance(device_or_stoage, str):
            self.device = Device.from_str(device_or_stoage)
        elif isinstance(device_or_stoage, Device):
            self.device = device_or_stoage
        elif isinstance(device_or_stoage, torch.UntypedStorage):
            self.device = Device.from_str(str(device_or_stoage.device))
            self.storage = device_or_stoage
            self.memory_table = MemoryTable(self.storage.nbytes())
            return
        else:
            raise TypeError(
                f"The device {device_or_stoage}({type(device_or_stoage)}) should be str, Device enum or "
                "an existing storage when initializing an Allocator"
            )

        tensor: torch.Tensor = torch.zeros(
            size, dtype=torch.uint8, device=self.device.to_torch_device_str()
        )
        if self.device.is_pinned():
            tensor = tensor.pin_memory()
        tensor[:] = tensor[:]

        self.storage: torch.UntypedStorage = tensor.untyped_storage()
        self.memory_table = MemoryTable(self.storage.nbytes())

    def clear(self):
        self.memory_table.clear()

    def allocate(self, tensors_meta: TensorsMeta) -> TensorsContent:
        size = tensors_meta.get_storage_size()
        slice_info = self.allocate_slice(size)
        return TensorsContent(self.storage, tensors_meta, slice_info)

    def free(self, tensors_content: TensorsContent):
        slice_info = tensors_content.slice_info
        self.free_slice(slice_info)

    def get_size(self) -> int:
        return self.memory_table.size

    def get_allocated_size(self) -> int:
        return self.memory_table.get_allocated_size()

    def get_free_size(self) -> int:
        return self.memory_table.get_free_size()

    def allocate_slice(self, size: int) -> SliceInfo:
        addr = self.memory_table.find_free_addr(size)
        if addr == None:
            raise MemoryError("Insufficient memory to allocate model storage slice")
        slice_info = SliceInfo.with_addr_size(addr, size)
        self.memory_table.append(slice_info)
        return slice_info

    def free_slice(self, slice_info: SliceInfo):
        self.memory_table.delete(slice_info)

    def can_allocate_model(self, model_meta: ModelMeta) -> bool:
        slice_sizes = model_meta.get_slice_sizes()

        return self.memory_table.can_allocate(slice_sizes)

    def can_allocate_model_when_empty(self, model_meta: ModelMeta) -> bool:
        slice_sizes = model_meta.get_slice_sizes()

        return self.memory_table.can_allocate_when_empty(slice_sizes)

    def move_slice(self, slice_info: SliceInfo, dst_addr: int):
        if slice_info.start == dst_addr:
            return

        src_start = slice_info.start
        src_end = slice_info.end

        self.memory_table.move(slice_info, dst_addr)

        dst_start = slice_info.start
        dst_end = slice_info.end

        if dst_start >= src_end or src_start >= dst_end:  # Not interleaved
            self.storage[dst_start:dst_end].copy_(self.storage[src_start:src_end])
        else:
            tmp_storage = torch.zeros(
                slice_info.get_size(), dtype=torch.uint8, device=self.storage.device
            ).untyped_storage()
            tmp_storage.copy_(self.storage[src_start:src_end])
            self.storage[dst_start:dst_end].copy_(tmp_storage)

    def fragment_collection(self):
        for i in range(len(self.memory_table)):
            slice_info = self.memory_table[i]
            slice_size = slice_info.get_size()
            start = 0 if i == 0 else self.memory_table[i - 1].end
            end = (
                self.memory_table.size
                if i == len(self.memory_table) - 1
                else self.memory_table[i + 1].start
            )
            new_addr = find_feasible_addr(start, end, slice_size)
            if not new_addr is None:
                self.move_slice(slice_info, new_addr)


class CUDAllocator(Allocator):
    def __init__(self, size: int, storage: torch.UntypedStorage) -> None:
        super().__init__(size, device_or_stoage=storage)


class CPUAllocator(Allocator):
    def __init__(
        self, size: int, storage: Optional[torch.UntypedStorage] = None
    ) -> None:
        device_or_stoage = storage
        if device_or_stoage is None:
            device_or_stoage = Device(DeviceType.CPU)
        super().__init__(size, device_or_stoage=device_or_stoage)


class PinnedAllocator(Allocator):
    def __init__(self, size: int) -> None:
        super().__init__(size, device_or_stoage=Device(DeviceType.PINNED))

    def allocate_pinned_buffer(
        self, size: int, offset: int = 0
    ) -> torch.UntypedStorage:
        if size + offset > self.storage.nbytes():
            raise MemoryError("Insufficient pinned memory to allocate pinned buffer")
        pinned_buffer: torch.UntypedStorage = self.storage[offset : offset + size]
        return pinned_buffer


def find_feasible_addr(start: int, end: int, size: int) -> Optional[int]:
    addr_candi = (
        (start + TORCH_VRAM_ALIGN_ADDR - 1)
        // TORCH_VRAM_ALIGN_ADDR
        * TORCH_VRAM_ALIGN_ADDR
    )

    if end - addr_candi >= size:
        return addr_candi
    else:
        return None


def is_interleaved(slice0: SliceInfo, slice1: SliceInfo) -> bool:
    start0, end0 = slice0.start, slice0.end
    start1, end1 = slice1.start, slice1.end

    # Check if slices are interleaved
    return not (start1 >= end0 or start0 >= end1)


def get_insert_index(table: List[SliceInfo], slice: SliceInfo) -> Optional[int]:
    index = bisect_left(table, slice)
    if index < len(table):
        if is_interleaved(slice, table[index]):
            return None
    if index > 0:
        if is_interleaved(slice, table[index - 1]):
            return None
    return index
