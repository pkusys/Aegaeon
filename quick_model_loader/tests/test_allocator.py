import pytest
import torch
from quick_model_loader.allocator import Allocator
from quick_model_loader.meta import (
    TensorInfo,
    TensorsMeta,
    SliceInfo,
)


def test_simple_allocate_slice():
    allocator = Allocator(1024, "cpu")

    slice_0_info = allocator.allocate_slice(4)
    slice_1_info = allocator.allocate_slice(8)
    slice_2_info = allocator.allocate_slice(16)
    slice_3_info = allocator.allocate_slice(32)
    slice_4_info = allocator.allocate_slice(944)

    assert slice_0_info.get_size() == 4
    assert slice_1_info.get_size() == 8
    assert slice_2_info.get_size() == 16
    assert slice_3_info.get_size() == 32
    assert slice_4_info.get_size() == 944

    assert slice_0_info.start == 0
    assert slice_1_info.start == 16
    assert slice_2_info.start == 32
    assert slice_3_info.start == 48
    assert slice_4_info.start == 80

    with pytest.raises(
        MemoryError, match="Insufficient memory to allocate model storage slice"
    ):
        allocator.allocate_slice(1)


def test_free_slice():
    allocator = Allocator(1024, "cpu")
    slice_0_info = allocator.allocate_slice(4)
    slice_1_info = allocator.allocate_slice(8)
    slice_2_info = allocator.allocate_slice(16)

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_0_info.start, slice_0_info.end - 1))

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_1_info.start, slice_1_info.end - 1))

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_2_info.start, slice_2_info.end - 1))

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_0_info.start + 1, slice_0_info.end))

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_1_info.start + 1, slice_1_info.end))

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free_slice(SliceInfo(slice_2_info.start + 1, slice_2_info.end))

    allocator.free_slice(slice_0_info)
    allocator.free_slice(slice_1_info)
    allocator.free_slice(slice_2_info)

    assert len(allocator.memory_table) == 0


def test_allocate_slice_after_free():
    allocator = Allocator(1024, "cpu")
    slice_0_info = allocator.allocate_slice(4)
    slice_1_info = allocator.allocate_slice(8)
    slice_2_info = allocator.allocate_slice(16)
    slice_3_info = allocator.allocate_slice(32)

    assert slice_0_info.start == 0
    assert slice_1_info.start == 16
    assert slice_2_info.start == 32
    assert slice_3_info.start == 48

    assert slice_0_info.get_size() == 4
    assert slice_1_info.get_size() == 8
    assert slice_2_info.get_size() == 16
    assert slice_3_info.get_size() == 32

    allocator.free_slice(slice_1_info)
    assert len(allocator.memory_table) == 3

    slice_4_info = allocator.allocate_slice(4)
    assert slice_4_info.start == 16
    assert slice_4_info.get_size() == 4

    slice_5_info = allocator.allocate_slice(3)
    assert slice_5_info.start == 80
    assert slice_5_info.get_size() == 3

    slice_6_info = allocator.allocate_slice(64)
    assert slice_6_info.start == 96
    assert slice_6_info.get_size() == 64


def test_allocate_slice_aligned():
    allocator = Allocator(1024, "cpu")
    slice_0_info = allocator.allocate_slice(6)
    slice_1_info = allocator.allocate_slice(12)
    slice_2_info = allocator.allocate_slice(32)

    assert slice_0_info.start == 0
    assert slice_1_info.start == 16
    assert slice_2_info.start == 32

    assert slice_0_info.get_size() == 6
    assert slice_1_info.get_size() == 12
    assert slice_2_info.get_size() == 32
    allocator.free_slice(slice_1_info)

    slice_3_info = allocator.allocate_slice(18)
    assert slice_3_info.get_size() == 18
    assert slice_3_info.start == 64


def test_simple_allocate():
    allocator = Allocator(1024, "cpu")

    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensor1_info = TensorInfo(torch.float16, [4, 4], (18, 50))
    tensor2_info = TensorInfo(torch.float16, [5, 5], (50, 100))

    tensors_012_meta = TensorsMeta(
        "test",
        100,
        0,
        {"tensor0": tensor0_info, "tensor1": tensor1_info, "tensor2": tensor2_info},
    )

    tensors_012_content = allocator.allocate(tensors_012_meta)
    assert len(allocator.memory_table) == 1
    assert tensors_012_content.slice_info.start == 0
    assert tensors_012_content.tensors_storage.nbytes() == 114
    assert (
        tensors_012_content.tensors_storage.nbytes()
        == tensors_012_content.slice_info.get_size()
    )
    assert (
        tensors_012_content.tensors_storage.nbytes()
        == tensors_012_meta.get_storage_size()
    )

    tensor3_info = TensorInfo(torch.float16, [3], (0, 6))
    tensor4_info = TensorInfo(torch.float16, [4], (6, 14))
    tensor5_info = TensorInfo(torch.float16, [5], (14, 24))

    tensors_345_meta = TensorsMeta(
        "test",
        12,
        0,
        {"tensor3": tensor3_info, "tensor4": tensor4_info, "tensor5": tensor5_info},
    )
    tensors_345_content = allocator.allocate(tensors_345_meta)
    assert len(allocator.memory_table) == 2
    assert tensors_345_content.slice_info.start == 128
    assert tensors_345_content.tensors_storage.nbytes() == 42
    assert (
        tensors_345_content.tensors_storage.nbytes()
        == tensors_345_content.slice_info.get_size()
    )
    assert (
        tensors_345_content.tensors_storage.nbytes()
        == tensors_345_meta.get_storage_size()
    )


def test_free():
    allocator = Allocator(1024, "cpu")

    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensors_0_meta = TensorsMeta("test", 18, 0, {"tensor0": tensor0_info})
    tensors_0_content = allocator.allocate(tensors_0_meta)

    tensor1_info = TensorInfo(torch.float16, [4, 4], (0, 32))
    tensors_1_meta = TensorsMeta("test", 32, 0, {"tensor1": tensor1_info})
    tensors_1_content = allocator.allocate(tensors_1_meta)

    tensor2_info = TensorInfo(torch.float16, [5, 5], (0, 50))
    tensors_2_meta = TensorsMeta("test", 50, 0, {"tensor2": tensor2_info})
    tensors_2_content = allocator.allocate(tensors_2_meta)

    assert len(allocator.memory_table) == 3

    allocator.free(tensors_0_content)
    allocator.free(tensors_1_content)
    allocator.free(tensors_2_content)

    assert len(allocator.memory_table) == 0

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free(tensors_0_content)

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free(tensors_1_content)

    with pytest.raises(MemoryError, match="Try to free unallocated memory!"):
        allocator.free(tensors_2_content)

    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensor1_info = TensorInfo(torch.float16, [4, 4], (18, 50))
    tensor2_info = TensorInfo(torch.float16, [5, 5], (50, 100))

    tensors_012_meta = TensorsMeta(
        "test",
        100,
        0,
        {"tensor0": tensor0_info, "tensor1": tensor1_info, "tensor2": tensor2_info},
    )

    tensors_012_content = allocator.allocate(tensors_012_meta)
    assert len(allocator.memory_table) == 1
    assert tensors_012_content.slice_info.start == 0
    assert tensors_012_content.tensors_storage.nbytes() == 114
    assert (
        tensors_012_content.tensors_storage.nbytes()
        == tensors_012_content.slice_info.get_size()
    )
    assert (
        tensors_012_content.tensors_storage.nbytes()
        == tensors_012_meta.get_storage_size()
    )


def test_allocate_after_free():
    allocator = Allocator(1024, "cpu")

    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensors_0_meta = TensorsMeta("test", 18, 0, {"tensor0": tensor0_info})
    allocator.allocate(tensors_0_meta)

    tensor1_info = TensorInfo(torch.float16, [4, 4], (0, 32))
    tensors_1_meta = TensorsMeta("test", 32, 0, {"tensor1": tensor1_info})
    tensors_1_content = allocator.allocate(tensors_1_meta)

    tensor2_info = TensorInfo(torch.float16, [5, 5], (0, 50))
    tensors_2_meta = TensorsMeta("test", 50, 0, {"tensor2": tensor2_info})
    allocator.allocate(tensors_2_meta)

    allocator.free(tensors_1_content)
    assert len(allocator.memory_table) == 2

    tensor3_info = TensorInfo(torch.float16, [6], (0, 12))
    tensors_3_meta = TensorsMeta("test", 12, 0, {"tensor3": tensor3_info})
    tensors_3_content = allocator.allocate(tensors_3_meta)
    assert tensors_3_content.slice_info.start == 32

    tensor4_info = TensorInfo(torch.float16, [9], (0, 18))
    tensors_4_meta = TensorsMeta("test", 18, 0, {"tensor4": tensor4_info})
    tensors_4_content = allocator.allocate(tensors_4_meta)
    assert tensors_4_content.slice_info.start == 128

    tensor5_info = TensorInfo(torch.float16, [3], (0, 6))
    tensors_5_meta = TensorsMeta("test", 6, 0, {"tensor5": tensor5_info})
    tensors_5_content = allocator.allocate(tensors_5_meta)
    assert tensors_5_content.slice_info.start == 48


def test_allocate_aligned():
    allocator = Allocator(1024, "cpu")

    tensor0_info = TensorInfo(torch.float16, [3], (0, 6))
    tensors_0_meta = TensorsMeta("test", 6, 0, {"tensor0": tensor0_info})
    tensors0_content = allocator.allocate(tensors_0_meta)

    tensor1_info = TensorInfo(torch.float32, [3], (0, 12))
    tensors_1_meta = TensorsMeta("test", 12, 0, {"tensor1": tensor1_info})
    tensors1_content = allocator.allocate(tensors_1_meta)

    tensor2_info = TensorInfo(torch.float64, [4], (0, 32))
    tensors_2_meta = TensorsMeta("test", 32, 0, {"tensor2": tensor2_info})
    tensors2_content = allocator.allocate(tensors_2_meta)

    assert tensors0_content.slice_info.start == 0
    assert tensors1_content.slice_info.start == 16
    assert tensors2_content.slice_info.start == 32

    allocator.free(tensors1_content)

    tensor3_info = TensorInfo(torch.float16, [18], (0, 18))
    tensors_3_meta = TensorsMeta("test", 18, 0, {"tensor3": tensor3_info})
    tensors3_content = allocator.allocate(tensors_3_meta)

    assert tensors3_content.slice_info.start == 64

    tensor4_info = TensorInfo(torch.float16, [14], (0, 14))
    tensors_4_meta = TensorsMeta("test", 14, 0, {"tensor4": tensor4_info})
    tensors4_content = allocator.allocate(tensors_4_meta)

    assert tensors4_content.slice_info.start == 16


def test_move_slice():
    allocator = Allocator(64, "cpu")
    storage = allocator.storage

    slice_size = 16
    slice_info = allocator.allocate_slice(slice_size)
    assert slice_info.start == 0
    assert slice_info.end == 16

    tensor_storage = storage[slice_info.start : slice_info.end]
    tensor_src = torch.tensor(list(range(8)), dtype=torch.float16)
    tensor_storage.copy_(tensor_src.untyped_storage())
    tensor = (
        torch.tensor([], dtype=torch.uint8, device=tensor_src.device)
        .set_(tensor_storage)
        .view(dtype=tensor_src.dtype)
        .reshape(tensor_src.shape)
    )
    assert torch.allclose(tensor, tensor_src)

    allocator.move_slice(slice_info, 16)
    assert slice_info.start == 16
    assert slice_info.end == 32
    tensor_storage = storage[slice_info.start : slice_info.end]
    tensor = (
        torch.tensor([], dtype=torch.uint8, device=tensor_src.device)
        .set_(tensor_storage)
        .view(dtype=tensor_src.dtype)
        .reshape(tensor_src.shape)
    )
    assert torch.allclose(tensor, tensor_src)

    allocator.move_slice(slice_info, 32)
    assert slice_info.start == 32
    assert slice_info.end == 48
    tensor_storage = storage[slice_info.start : slice_info.end]
    tensor = (
        torch.tensor([], dtype=torch.uint8, device=tensor_src.device)
        .set_(tensor_storage)
        .view(dtype=tensor_src.dtype)
        .reshape(tensor_src.shape)
    )
    assert torch.allclose(tensor, tensor_src)

    allocator.move_slice(slice_info, 48)
    assert slice_info.start == 48
    assert slice_info.end == 64
    tensor_storage = storage[slice_info.start : slice_info.end]
    tensor = (
        torch.tensor([], dtype=torch.uint8, device=tensor_src.device)
        .set_(tensor_storage)
        .view(dtype=tensor_src.dtype)
        .reshape(tensor_src.shape)
    )
    assert torch.allclose(tensor, tensor_src)


def test_fragments_collection():
    allocator = Allocator(1024, "cpu")

    tensor0_info = TensorInfo(torch.float16, [3], (0, 6))
    tensors_0_meta = TensorsMeta("test", 6, 0, {"tensor0": tensor0_info})
    tensors0_content = allocator.allocate(tensors_0_meta)
    tensor0_src = torch.tensor([0, 1, 2], dtype=torch.float16)
    tensors0_content.tensors_storage.copy_(tensor0_src.untyped_storage())
    tensor0 = (
        torch.tensor([], dtype=torch.uint8, device=tensor0_src.device)
        .set_(tensors0_content.tensors_storage)
        .view(dtype=tensor0_src.dtype)
        .reshape(tensor0_src.shape)
    )
    assert torch.allclose(tensor0, tensor0_src)

    tensor1_info = TensorInfo(torch.float32, [3], (0, 12))
    tensors_1_meta = TensorsMeta("test", 12, 0, {"tensor1": tensor1_info})
    tensors1_content = allocator.allocate(tensors_1_meta)
    tensor1_src = torch.tensor([3, 4, 5], dtype=torch.float32)
    tensors1_content.tensors_storage.copy_(tensor1_src.untyped_storage())
    tensor1 = (
        torch.tensor([], dtype=torch.uint8, device=tensor1_src.device)
        .set_(tensors1_content.tensors_storage)
        .view(dtype=tensor1_src.dtype)
        .reshape(tensor1_src.shape)
    )
    assert torch.allclose(tensor1, tensor1_src)

    tensor2_info = TensorInfo(torch.float64, [4], (0, 32))
    tensors_2_meta = TensorsMeta("test", 32, 0, {"tensor2": tensor2_info})
    tensors2_content = allocator.allocate(tensors_2_meta)
    tensor2_src = torch.tensor([6, 7, 8, 9], dtype=torch.float64)
    tensors2_content.tensors_storage.copy_(tensor2_src.untyped_storage())
    tensor2 = (
        torch.tensor([], dtype=torch.uint8, device=tensor2_src.device)
        .set_(tensors2_content.tensors_storage)
        .view(dtype=tensor2_src.dtype)
        .reshape(tensor2_src.shape)
    )
    assert torch.allclose(tensor2, tensor2_src)

    allocator.free(tensors1_content)
    allocator.fragment_collection()

    assert tensors0_content.slice_info.start == 0
    assert tensors0_content.slice_info.get_size() == 6
    assert tensors2_content.slice_info.start == 16
    assert tensors2_content.slice_info.get_size() == 32

    tensor0 = (
        torch.tensor([], dtype=torch.uint8, device=tensor0_src.device)
        .set_(tensors0_content.tensors_storage)
        .view(dtype=tensor0_src.dtype)
        .reshape(tensor0_src.shape)
    )
    assert torch.allclose(tensor0, tensor0_src)

    tensor2 = (
        torch.tensor([], dtype=torch.uint8, device=tensor2_src.device)
        .set_(tensors2_content.tensors_storage)
        .view(dtype=tensor2_src.dtype)
        .reshape(tensor2_src.shape)
    )
    assert torch.allclose(tensor2, tensor2_src)

    allocator.free(tensors0_content)
    allocator.fragment_collection()

    assert tensors2_content.slice_info.start == 0
    assert tensors2_content.slice_info.get_size() == 32

    tensor2 = (
        torch.tensor([], dtype=torch.uint8, device=tensor2_src.device)
        .set_(tensors2_content.tensors_storage)
        .view(dtype=tensor2_src.dtype)
        .reshape(tensor2_src.shape)
    )
    assert torch.allclose(tensor2, tensor2_src)
