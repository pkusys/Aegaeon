import torch
import glob
import os
from typing import Dict
from safetensors import safe_open
from quick_model_loader import QuickModelLoader
from quick_model_loader.meta import ModelContent, TensorInfo, TensorsMeta
from quick_model_loader.model_loader import load_tensors_file_buffer

model_loader = QuickModelLoader([512], 512, 1024)
model_loader_mid = QuickModelLoader([512], 512, 512)
model_loader_small = QuickModelLoader([512], 512, 320)


def get_model_allocated_storage_size(model_content: ModelContent) -> int:
    allocated_storage_size = 0
    for sharding_content in model_content.sharding_contents:
        for tensors_content in sharding_content.tensors_contents:
            tensors_storage = tensors_content.tensors_storage
            allocated_storage_size += tensors_storage.nbytes()
    return allocated_storage_size


def tensors_are_equal(tensor0: torch.Tensor, tensor1: torch.Tensor) -> bool:
    if tensor0.shape != tensor1.shape:
        return False
    if tensor0.device != tensor1.device:
        return False
    return torch.allclose(tensor0, tensor1)


def safe_open_model_content(
    model_content: ModelContent, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    tensors_dict = {}

    model_path = model_content.model_name
    pattern = "*.safetensors"
    tensors_paths = glob.glob(os.path.join(model_path, pattern))

    for tensors_path in tensors_paths:
        with safe_open(tensors_path, "pt", device) as f:
            for k in f.keys():
                tensors_dict[k] = f.get_tensor(k)

    return tensors_dict


def check_model_content(model_content: ModelContent, device: str = "cpu"):
    tensors_dict = safe_open_model_content(model_content, device)
    count = 0

    for tensor_name, tensor in model_content.to_tensors():
        tensor_to_cmp = tensors_dict[tensor_name]
        assert tensors_are_equal(tensor, tensor_to_cmp)
        count += 1

    assert count == len(tensors_dict)


def test_cache_model0():
    model_loader.cache_model("test_models/model0")
    model0_content = model_loader.cached_models["test_models/model0"]

    assert model0_content.model_name == "test_models/model0"
    assert len(model0_content.sharding_contents) == 1
    assert len(model0_content.sharding_contents[0].tensors_contents) == 2
    assert model0_content.get_storage_size() == 58
    assert get_model_allocated_storage_size(model0_content) == 58

    check_model_content(model0_content, "cpu")

    model_loader.clear()


def test_cache_model1():
    model_loader.cache_model("test_models/model1")
    model1_content = model_loader.cached_models["test_models/model1"]

    assert model1_content.model_name == "test_models/model1"
    assert len(model1_content.sharding_contents) == 1
    assert len(model1_content.sharding_contents[0].tensors_contents) == 2
    assert model1_content.get_storage_size() == 58
    assert get_model_allocated_storage_size(model1_content) == 58

    check_model_content(model1_content, "cpu")

    model_loader.clear()


def test_cache_model2():
    model_loader.cache_model("test_models/model2")
    model2_content = model_loader.cached_models["test_models/model2"]

    assert model2_content.model_name == "test_models/model2"
    assert len(model2_content.sharding_contents) == 1
    assert len(model2_content.sharding_contents[0].tensors_contents) == 2
    assert model2_content.get_storage_size() == 168
    assert get_model_allocated_storage_size(model2_content) == 168

    check_model_content(model2_content, "cpu")

    model_loader.clear()


def test_cache_model3():
    model_loader.cache_model("test_models/model3")
    model3_content = model_loader.cached_models["test_models/model3"]

    assert model3_content.model_name == "test_models/model3"
    assert len(model3_content.sharding_contents) == 1
    assert len(model3_content.sharding_contents[0].tensors_contents) == 3
    assert model3_content.get_storage_size() == 218
    assert get_model_allocated_storage_size(model3_content) == 218

    check_model_content(model3_content, "cpu")

    model_loader.clear()


def test_load_model0():
    model_loader.load_model("test_models/model0")
    model0_content = model_loader.loaded_model

    assert model0_content.model_name == "test_models/model0"
    assert len(model0_content.sharding_contents) == 1
    assert len(model0_content.sharding_contents[0].tensors_contents) == 2
    assert model0_content.get_storage_size() == 58
    assert get_model_allocated_storage_size(model0_content) == 58

    check_model_content(model0_content, "cuda")

    model_loader.clear()


def test_load_model1():
    model_loader.load_model("test_models/model1")
    model1_content = model_loader.loaded_model

    assert model1_content.model_name == "test_models/model1"
    assert len(model1_content.sharding_contents) == 1
    assert len(model1_content.sharding_contents[0].tensors_contents) == 2
    assert model1_content.get_storage_size() == 58
    assert get_model_allocated_storage_size(model1_content) == 58

    check_model_content(model1_content, "cuda")

    model_loader.clear()


def test_load_model2():
    model_loader.load_model("test_models/model2")
    model2_content = model_loader.loaded_model

    assert model2_content.model_name == "test_models/model2"
    assert len(model2_content.sharding_contents) == 1
    assert len(model2_content.sharding_contents[0].tensors_contents) == 2
    assert model2_content.get_storage_size() == 168
    assert get_model_allocated_storage_size(model2_content) == 168

    check_model_content(model2_content, "cuda")

    model_loader.clear()


def test_load_model3():
    model_loader.load_model("test_models/model3")
    model3_content = model_loader.loaded_model

    assert model3_content.model_name == "test_models/model3"
    assert len(model3_content.sharding_contents) == 1
    assert len(model3_content.sharding_contents[0].tensors_contents) == 3
    assert model3_content.get_storage_size() == 218
    assert get_model_allocated_storage_size(model3_content) == 218

    check_model_content(model3_content, "cuda")

    model_loader.clear()


def test_load_safetensors_file_buffer():
    allocator = model_loader.cpu_allocator

    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensor1_info = TensorInfo(torch.float32, [2, 2], (18, 34))
    tensor2_info = TensorInfo(torch.float64, [2], (34, 50))

    tensor0 = torch.tensor(list(range(9)), dtype=torch.float16).reshape((3, 3))
    tensor1 = torch.tensor([9, 10, 11, 12], dtype=torch.float32).reshape((2, 2))
    tensor2 = torch.tensor([13, 14], dtype=torch.float64).reshape((2))

    storage: torch.UntypedStorage = torch.tensor(
        [0] * 50, dtype=torch.int8
    ).untyped_storage()
    storage[0:18].copy_(tensor0.untyped_storage())
    storage[18:34].copy_(tensor1.untyped_storage())
    storage[34:50].copy_(tensor2.untyped_storage())

    meta = TensorsMeta(
        "test",
        50,
        0,
        {"tensor0": tensor0_info, "tensor1": tensor1_info, "tensor2": tensor2_info},
    )
    tensors_content = allocator.allocate(meta)
    load_tensors_file_buffer(tensors_content, storage)

    new_tensor0_info = tensors_content.tensors_meta.aligned_tensor_info_map["tensor0"]
    new_tensor1_info = tensors_content.tensors_meta.aligned_tensor_info_map["tensor1"]
    new_tensor2_info = tensors_content.tensors_meta.aligned_tensor_info_map["tensor2"]

    assert new_tensor0_info.dtype == torch.float16
    assert new_tensor0_info.shape == [3, 3]
    assert new_tensor0_info.data_offsets == (0, 18)

    assert new_tensor1_info.dtype == torch.float32
    assert new_tensor1_info.shape == [2, 2]
    assert new_tensor1_info.data_offsets == (32, 48)

    assert new_tensor2_info.dtype == torch.float64
    assert new_tensor2_info.shape == [2]
    assert new_tensor2_info.data_offsets == (48, 64)

    for tensor_name, tensor in tensors_content.to_tensors():
        if tensor_name == "tensor0":
            assert tensors_are_equal(tensor, tensor0)
        if tensor_name == "tensor1":
            assert tensors_are_equal(tensor, tensor1)
        if tensor_name == "tensor2":
            assert tensors_are_equal(tensor, tensor2)

    model_loader.clear()


def test_fragment_collection():
    model0 = "test_models/model0"
    model_loader.cache_model(model0)

    model1 = "test_models/model1"
    model_loader.cache_model(model1)

    model2 = "test_models/model2"
    model_loader.cache_model(model2)

    model3 = "test_models/model3"
    model_loader.cache_model(model3)

    model_loader.evict_cached_model(model0)
    model_loader.cpu_allocator.fragment_collection()

    model1_content: ModelContent = model_loader.cached_models[model1]
    model1_tensors_contents = model1_content.sharding_contents[0].tensors_contents
    assert min([t.slice_info.start for t in model1_tensors_contents]) == 0
    assert max([t.slice_info.end for t in model1_tensors_contents]) == 72
    check_model_content(model1_content, "cpu")

    model2_content: ModelContent = model_loader.cached_models[model2]
    model2_tensors_contents = model2_content.sharding_contents[0].tensors_contents
    assert min([t.slice_info.start for t in model2_tensors_contents]) == 80
    assert max([t.slice_info.end for t in model2_tensors_contents]) == 262
    check_model_content(model2_content, "cpu")

    model3_content: ModelContent = model_loader.cached_models[model3]
    model3_tensors_contents = model3_content.sharding_contents[0].tensors_contents
    assert min([t.slice_info.start for t in model3_tensors_contents]) == 272
    assert max([t.slice_info.end for t in model3_tensors_contents]) == 516
    check_model_content(model3_content, "cpu")

    model_loader.evict_cached_model(model2)
    model_loader.cpu_allocator.fragment_collection()

    model1_content: ModelContent = model_loader.cached_models[model1]
    model1_tensors_contents = model1_content.sharding_contents[0].tensors_contents
    assert min([t.slice_info.start for t in model1_tensors_contents]) == 0
    assert max([t.slice_info.end for t in model1_tensors_contents]) == 72
    check_model_content(model1_content, "cpu")

    model3_content: ModelContent = model_loader.cached_models[model3]
    model3_tensors_contents = model3_content.sharding_contents[0].tensors_contents
    assert min([t.slice_info.start for t in model3_tensors_contents]) == 80
    assert max([t.slice_info.end for t in model3_tensors_contents]) == 324
    check_model_content(model3_content, "cpu")

    model_loader.clear()


def test_lru_model():
    model0 = "test_models/model0"
    model_loader.cache_model(model0)
    assert model_loader.lru_model == model0

    model1 = "test_models/model1"
    model_loader.cache_model(model1)
    assert model_loader.lru_model == model0

    model2 = "test_models/model2"
    model_loader.cache_model(model2)
    assert model_loader.lru_model == model0

    model3 = "test_models/model3"
    model_loader.cache_model(model3)
    assert model_loader.lru_model == model0

    for _ in model_loader.model_to_tensors(model0, "cpu"):
        pass
    assert model_loader.lru_model == model1

    for _ in model_loader.model_to_tensors(model1, "cpu"):
        pass
    assert model_loader.lru_model == model2

    for _ in model_loader.model_to_tensors(model2, "cpu"):
        pass
    assert model_loader.lru_model == model3

    for _ in model_loader.model_to_tensors(model3, "cpu"):
        pass
    assert model_loader.lru_model == model0

    model_loader.load_model(model0)
    assert model_loader.lru_model == model1

    model_loader.load_model(model1)
    assert model_loader.lru_model == model2

    model_loader.load_model(model2)
    assert model_loader.lru_model == model3

    model_loader.load_model(model3)
    assert model_loader.lru_model == model0

    model_loader.clear()


def test_evict_lru_model():
    model0 = "test_models/model0"
    model_loader.cache_model(model0)

    model1 = "test_models/model1"
    model_loader.cache_model(model1)

    model2 = "test_models/model2"
    model_loader.cache_model(model2)

    model3 = "test_models/model3"
    model_loader.cache_model(model3)

    evicted_model = model_loader.evict_lru_model()
    assert evicted_model == model0
    assert list(model_loader.cached_models.keys()) == [model1, model2, model3]

    evicted_model = model_loader.evict_lru_model()
    assert evicted_model == model1
    assert list(model_loader.cached_models.keys()) == [model2, model3]

    evicted_model = model_loader.evict_lru_model()
    assert evicted_model == model2
    assert list(model_loader.cached_models.keys()) == [model3]

    evicted_model = model_loader.evict_lru_model()
    assert evicted_model == model3
    assert list(model_loader.cached_models.keys()) == []

    model_loader.clear()


def test_auto_evict_small():
    model0 = "test_models/model0"
    model_loader_small.cache_model(model0)

    assert list(model_loader_small.cached_models.keys()) == [model0]
    # using .store to avoid trigger lru when testing
    model0_content = model_loader_small.cached_models.store[model0]
    check_model_content(model0_content, "cpu")

    model1 = "test_models/model1"
    model_loader_small.cache_model(model1)

    assert list(model_loader_small.cached_models.keys()) == [model0, model1]
    model0_content = model_loader_small.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model1_content = model_loader_small.cached_models.store[model1]
    check_model_content(model1_content, "cpu")

    model2 = "test_models/model2"
    model_loader_small.cache_model(model2)

    assert list(model_loader_small.cached_models.keys()) == [model1, model2]
    model1_content = model_loader_small.cached_models.store[model1]
    check_model_content(model1_content, "cpu")
    model2_content = model_loader_small.cached_models.store[model2]
    check_model_content(model2_content, "cpu")

    model3 = "test_models/model3"
    model_loader_small.cache_model(model3)

    assert list(model_loader_small.cached_models.keys()) == [model3]
    model3_content = model_loader_small.cached_models.store[model3]
    check_model_content(model3_content, "cpu")

    model_loader_small.cache_model(model0)
    assert list(model_loader_small.cached_models.keys()) == [model0]
    model0_content = model_loader_small.cached_models.store[model0]
    check_model_content(model0_content, "cpu")

    model1 = "test_models/model1"
    model_loader_small.cache_model(model1)

    assert list(model_loader_small.cached_models.keys()) == [model0, model1]
    model0_content = model_loader_small.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model1_content = model_loader_small.cached_models.store[model1]
    check_model_content(model1_content, "cpu")

    model_loader_small.load_model(model0)
    model0_content = model_loader_small.loaded_model
    check_model_content(model0_content, "cuda")
    model1_content = model_loader_small.cached_models.store[model1]
    check_model_content(model1_content, "cpu")

    model_loader_small.cache_model(model2)

    assert list(model_loader_small.cached_models.keys()) == [model0, model2]
    model0_content = model_loader_small.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model2_content = model_loader_small.cached_models.store[model2]
    check_model_content(model2_content, "cpu")

    model_loader_small.load_model(model3)

    assert list(model_loader_small.cached_models.keys()) == [model3]
    model3_content = model_loader_small.cached_models.store[model3]
    check_model_content(model3_content, "cpu")
    model3_content = model_loader_small.loaded_model
    check_model_content(model3_content, "cuda")

    model_loader_small.clear()


def test_auto_evict_mid():
    model0 = "test_models/model0"
    model_loader_mid.cache_model(model0)

    assert list(model_loader_mid.cached_models.keys()) == [model0]
    # using .store to avoid trigger lru when testing
    model0_content = model_loader_mid.cached_models.store[model0]
    check_model_content(model0_content, "cpu")

    model1 = "test_models/model1"
    model_loader_mid.cache_model(model1)

    assert list(model_loader_mid.cached_models.keys()) == [model0, model1]
    model0_content = model_loader_mid.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model1_content = model_loader_mid.cached_models.store[model1]
    check_model_content(model1_content, "cpu")

    model2 = "test_models/model2"
    model_loader_mid.cache_model(model2)

    assert list(model_loader_mid.cached_models.keys()) == [model0, model1, model2]
    model0_content = model_loader_mid.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model1_content = model_loader_mid.cached_models.store[model1]
    check_model_content(model1_content, "cpu")
    model2_content = model_loader_mid.cached_models.store[model2]
    check_model_content(model2_content, "cpu")

    model3 = "test_models/model3"
    model_loader_mid.cache_model(model3)

    assert list(model_loader_mid.cached_models.keys()) == [model2, model3]

    model2_content = model_loader_mid.cached_models.store[model2]
    check_model_content(model2_content, "cpu")
    model3_content = model_loader_mid.cached_models.store[model3]
    check_model_content(model3_content, "cpu")

    model_loader_mid.cache_model(model0)
    model_loader_mid.cache_model(model1)

    assert list(model_loader_mid.cached_models.keys()) == [model3, model0, model1]
    model0_content = model_loader_mid.cached_models.store[model0]
    check_model_content(model0_content, "cpu")
    model1_content = model_loader_mid.cached_models.store[model1]
    check_model_content(model1_content, "cpu")
    model3_content = model_loader_mid.cached_models.store[model3]
    check_model_content(model3_content, "cpu")

    model_loader_mid.load_model(model3)
    assert list(model_loader_mid.cached_models.keys()) == [model0, model1, model3]

    model3_content = model_loader_mid.loaded_model
    check_model_content(model3_content, "cuda")

    model_loader_mid.load_model(model2)
    assert list(model_loader_mid.cached_models.keys()) == [model3, model2]
    model2_content = model_loader_mid.cached_models.store[model2]
    check_model_content(model2_content, "cpu")
    model3_content = model_loader_mid.cached_models.store[model3]
    check_model_content(model3_content, "cpu")
    model2_content = model_loader_mid.loaded_model
    check_model_content(model2_content, "cuda")

    model_loader_mid.clear()


def test_real_model():
    model_path = os.getenv("TEST_MODEL")
    if model_path is None:
        return

    _GB = 1024 * 1024 * 1024
    model_loader = QuickModelLoader(24 * _GB, 10 * _GB, 48 * _GB)
    model_loader.cache_model(model_path)
    model_content = model_loader.cached_models[model_path]

    check_model_content(model_content, "cpu")

    model_loader.load_model(model_path)
    model_content = model_loader.loaded_model

    check_model_content(model_content, "cuda")
