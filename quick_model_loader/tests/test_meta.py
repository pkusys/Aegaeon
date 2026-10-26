import torch
from safetensors.torch import save_file
from quick_model_loader.meta import TensorsMeta, TensorInfo


def test_read_meta():
    tensor0 = torch.tensor([0, 1, 2, 3], dtype=torch.float16)
    tensor1 = torch.tensor([4, 5, 6], dtype=torch.float16)
    tensor2 = torch.tensor([7, 8, 9], dtype=torch.float16)

    tensors_dict = {"tensor0": tensor0, "tensor1": tensor1, "tensor2": tensor2}
    save_file(tensors_dict, "test_models/test.safetensors")

    meta = TensorsMeta.from_tensors_file("test_models/test.safetensors")
    assert meta.get_storage_size() == 16 + 16 + 6  # 10 * 2 bytes
    assert len(meta.file_tensor_info_map) == 3  # 3 tensors
    assert len(meta.aligned_tensor_info_map) == 3

    tensor0_file_info = meta.file_tensor_info_map["tensor0"]
    tensor1_file_info = meta.file_tensor_info_map["tensor1"]
    tensor2_file_info = meta.file_tensor_info_map["tensor2"]

    assert tensor0_file_info.dtype == torch.float16
    assert tensor1_file_info.dtype == torch.float16
    assert tensor2_file_info.dtype == torch.float16

    assert tensor0_file_info.shape == [4]
    assert tensor1_file_info.shape == [3]
    assert tensor2_file_info.shape == [3]

    assert tensor0_file_info.get_size() == 8
    assert tensor1_file_info.get_size() == 6
    assert tensor2_file_info.get_size() == 6

    assert tensor0_file_info.data_offsets == (0, 8)
    assert tensor1_file_info.data_offsets == (8, 14)
    assert tensor2_file_info.data_offsets == (14, 20)

    tensor0_aligned_info = meta.aligned_tensor_info_map["tensor0"]
    tensor1_aligned_info = meta.aligned_tensor_info_map["tensor1"]
    tensor2_aligned_info = meta.aligned_tensor_info_map["tensor2"]

    assert tensor0_aligned_info.dtype == torch.float16
    assert tensor1_aligned_info.dtype == torch.float16
    assert tensor2_aligned_info.dtype == torch.float16

    assert tensor0_aligned_info.shape == [4]
    assert tensor1_aligned_info.shape == [3]
    assert tensor2_aligned_info.shape == [3]

    assert tensor0_aligned_info.get_size() == 8
    assert tensor1_aligned_info.get_size() == 6
    assert tensor2_aligned_info.get_size() == 6

    assert tensor0_aligned_info.data_offsets == (0, 8)
    assert tensor1_aligned_info.data_offsets == (16, 22)
    assert tensor2_aligned_info.data_offsets == (32, 38)


def test_meta_memory_aligned():
    tensor0_info = TensorInfo(torch.float16, [3, 3], (0, 18))
    tensor1_info = TensorInfo(torch.float32, [2, 2], (18, 34))
    tensor2_info = TensorInfo(torch.float64, [2], (34, 50))
    meta = TensorsMeta(
        "test",
        50,
        0,
        {"tensor0": tensor0_info, "tensor1": tensor1_info, "tensor2": tensor2_info},
    )

    assert meta.get_storage_size() == 64

    aligned_tensor_info_map = meta.aligned_tensor_info_map
    aligned_tensor0_info = aligned_tensor_info_map["tensor0"]
    aligned_tensor1_info = aligned_tensor_info_map["tensor1"]
    aligned_tensor2_info = aligned_tensor_info_map["tensor2"]

    assert aligned_tensor0_info.dtype == torch.float16
    assert aligned_tensor0_info.shape == [3, 3]
    assert aligned_tensor0_info.data_offsets == (0, 18)

    assert aligned_tensor1_info.dtype == torch.float32
    assert aligned_tensor1_info.shape == [2, 2]
    assert aligned_tensor1_info.data_offsets == (32, 48)

    assert aligned_tensor2_info.dtype == torch.float64
    assert aligned_tensor2_info.shape == [2]
    assert aligned_tensor2_info.data_offsets == (48, 64)


def test_read_model0_meta():
    meta1 = TensorsMeta.from_tensors_file(
        "test_models/model0/model-00001-of-00002.safetensors"
    )

    assert meta1.get_file_storage_size() == 12  # 4 + 4 + 4 bytes
    assert len(meta1.file_tensor_info_map) == 2  # 2 tensors

    tensor0_file_info = meta1.file_tensor_info_map["tensor0"]
    tensor1_file_info = meta1.file_tensor_info_map["tensor1"]

    assert tensor0_file_info.dtype == torch.float32
    assert tensor1_file_info.dtype == torch.float32

    assert tensor0_file_info.shape == [1]
    assert tensor1_file_info.shape == [2]

    assert tensor0_file_info.data_offsets == (0, 4)
    assert tensor1_file_info.data_offsets == (4, 12)

    assert meta1.get_storage_size() == 24  # 16 + 4 + 4 bytes
    assert len(meta1.aligned_tensor_info_map) == 2  # 2 tensors

    tensor0_aligned_info = meta1.aligned_tensor_info_map["tensor0"]
    tensor1_aligned_info = meta1.aligned_tensor_info_map["tensor1"]

    assert tensor0_aligned_info.dtype == torch.float32
    assert tensor1_aligned_info.dtype == torch.float32

    assert tensor0_aligned_info.shape == [1]
    assert tensor1_aligned_info.shape == [2]

    assert tensor0_aligned_info.data_offsets == (0, 4)
    assert tensor1_aligned_info.data_offsets == (16, 24)

    meta2 = TensorsMeta.from_tensors_file(
        "test_models/model0/model-00002-of-00002.safetensors"
    )

    assert meta2.get_file_storage_size() == 26  # 8 + 9 * 2 bytes
    assert len(meta2.file_tensor_info_map) == 2  # 2 tensors

    tensor2_file_info = meta2.file_tensor_info_map["tensor2"]
    tensor3_file_info = meta2.file_tensor_info_map["tensor3"]

    assert tensor2_file_info.dtype == torch.float16
    assert tensor3_file_info.dtype == torch.float16

    assert tensor2_file_info.shape == [2, 2]
    assert tensor3_file_info.shape == [3, 3]

    assert tensor2_file_info.data_offsets == (0, 8)
    assert tensor3_file_info.data_offsets == (8, 26)

    assert meta2.get_storage_size() == 34  # 16 + 9 * 2 bytes
    assert len(meta2.aligned_tensor_info_map) == 2  # 2 tensors

    tensor2_aligned_info = meta2.aligned_tensor_info_map["tensor2"]
    tensor3_aligned_info = meta2.aligned_tensor_info_map["tensor3"]

    assert tensor2_aligned_info.dtype == torch.float16
    assert tensor3_aligned_info.dtype == torch.float16

    assert tensor2_aligned_info.shape == [2, 2]
    assert tensor3_aligned_info.shape == [3, 3]

    assert tensor2_aligned_info.data_offsets == (0, 8)
    assert tensor3_aligned_info.data_offsets == (16, 34)


def test_read_model1_meta():
    meta1 = TensorsMeta.from_tensors_file(
        "test_models/model1/model-00001-of-00002.safetensors"
    )

    assert meta1.get_file_storage_size() == 26  # 8 + 9 * 2 bytes
    assert len(meta1.file_tensor_info_map) == 2  # 2 tensors

    tensor0_file_info = meta1.file_tensor_info_map["tensor0"]
    tensor1_file_info = meta1.file_tensor_info_map["tensor1"]

    assert tensor0_file_info.dtype == torch.float16
    assert tensor1_file_info.dtype == torch.float16

    assert tensor0_file_info.shape == [2, 2]
    assert tensor1_file_info.shape == [3, 3]

    assert tensor0_file_info.data_offsets == (0, 8)
    assert tensor1_file_info.data_offsets == (8, 26)

    assert meta1.get_storage_size() == 34  # 16 + 9 * 2 bytes
    assert len(meta1.aligned_tensor_info_map) == 2  # 2 tensors

    tensor2_aligned_info = meta1.aligned_tensor_info_map["tensor0"]
    tensor3_aligned_info = meta1.aligned_tensor_info_map["tensor1"]

    assert tensor2_aligned_info.dtype == torch.float16
    assert tensor3_aligned_info.dtype == torch.float16

    assert tensor2_aligned_info.shape == [2, 2]
    assert tensor3_aligned_info.shape == [3, 3]

    assert tensor2_aligned_info.data_offsets == (0, 8)
    assert tensor3_aligned_info.data_offsets == (16, 34)

    meta2 = TensorsMeta.from_tensors_file(
        "test_models/model1/model-00002-of-00002.safetensors"
    )

    assert meta2.get_file_storage_size() == 12  # 4 + 4 + 4 bytes
    assert len(meta2.file_tensor_info_map) == 2  # 2 tensors

    tensor2_file_info = meta2.file_tensor_info_map["tensor2"]
    tensor3_file_info = meta2.file_tensor_info_map["tensor3"]

    assert tensor2_file_info.dtype == torch.float32
    assert tensor3_file_info.dtype == torch.float32

    assert tensor2_file_info.shape == [1]
    assert tensor3_file_info.shape == [2]

    assert tensor2_file_info.data_offsets == (0, 4)
    assert tensor3_file_info.data_offsets == (4, 12)

    assert meta2.get_storage_size() == 24  # 16 + 4 + 4 bytes
    assert len(meta2.aligned_tensor_info_map) == 2  # 2 tensors

    tensor2_aligned_info = meta2.aligned_tensor_info_map["tensor2"]
    tensor3_aligned_info = meta2.aligned_tensor_info_map["tensor3"]

    assert tensor2_aligned_info.dtype == torch.float32
    assert tensor3_aligned_info.dtype == torch.float32

    assert tensor2_aligned_info.shape == [1]
    assert tensor3_aligned_info.shape == [2]

    assert tensor2_aligned_info.data_offsets == (0, 4)
    assert tensor3_aligned_info.data_offsets == (16, 24)


def test_read_model2_meta():
    meta1 = TensorsMeta.from_tensors_file(
        "test_models/model2/model-00001-of-00002.safetensors"
    )

    assert meta1.get_file_storage_size() == 82  # 64 + 9 * 2 bytes
    assert len(meta1.file_tensor_info_map) == 2  # 2 tensors

    tensor0_file_info = meta1.file_tensor_info_map["tensor0"]
    tensor1_file_info = meta1.file_tensor_info_map["tensor1"]

    assert tensor0_file_info.dtype == torch.float32
    assert tensor1_file_info.dtype == torch.float16

    assert tensor0_file_info.shape == [4, 4]
    assert tensor1_file_info.shape == [3, 3]

    assert tensor0_file_info.data_offsets == (0, 64)
    assert tensor1_file_info.data_offsets == (64, 82)

    assert meta1.get_storage_size() == 82  # 64 + 9 * 2 bytes
    assert len(meta1.aligned_tensor_info_map) == 2  # 2 tensors

    tensor0_aligned_info = meta1.aligned_tensor_info_map["tensor0"]
    tensor1_aligned_info = meta1.aligned_tensor_info_map["tensor1"]

    assert tensor0_aligned_info.dtype == torch.float32
    assert tensor1_aligned_info.dtype == torch.float16

    assert tensor0_aligned_info.shape == [4, 4]
    assert tensor1_aligned_info.shape == [3, 3]

    assert tensor0_aligned_info.data_offsets == (0, 64)
    assert tensor1_aligned_info.data_offsets == (64, 82)

    meta2 = TensorsMeta.from_tensors_file(
        "test_models/model2/model-00002-of-00002.safetensors"
    )

    assert meta2.get_file_storage_size() == 86  # 32 + 27 * 2 bytes
    assert len(meta2.file_tensor_info_map) == 2  # 2 tensors

    tensor2_file_info = meta2.file_tensor_info_map["tensor2"]
    tensor3_file_info = meta2.file_tensor_info_map["tensor3"]

    assert tensor2_file_info.dtype == torch.float32
    assert tensor3_file_info.dtype == torch.float16

    assert tensor2_file_info.shape == [2, 2, 2]
    assert tensor3_file_info.shape == [3, 3, 3]

    assert tensor2_file_info.data_offsets == (0, 32)
    assert tensor3_file_info.data_offsets == (32, 86)

    assert meta2.get_storage_size() == 86  # 32 + 27 * 2 bytes
    assert len(meta2.aligned_tensor_info_map) == 2  # 2 tensors

    tensor2_aligned_info = meta2.aligned_tensor_info_map["tensor2"]
    tensor3_aligned_info = meta2.aligned_tensor_info_map["tensor3"]

    assert tensor2_aligned_info.dtype == torch.float32
    assert tensor3_aligned_info.dtype == torch.float16

    assert tensor2_aligned_info.shape == [2, 2, 2]
    assert tensor3_aligned_info.shape == [3, 3, 3]

    assert tensor2_aligned_info.data_offsets == (0, 32)
    assert tensor3_aligned_info.data_offsets == (32, 86)


def test_read_model3_meta():
    meta1 = TensorsMeta.from_tensors_file(
        "test_models/model3/model-00001-of-00003.safetensors"
    )

    assert meta1.get_file_storage_size() == 8  # 4 + 4 bytes
    assert len(meta1.file_tensor_info_map) == 2  # 2 tensors

    tensor0_file_info = meta1.file_tensor_info_map["tensor0"]
    tensor1_file_info = meta1.file_tensor_info_map["tensor1"]

    assert tensor0_file_info.dtype == torch.float32
    assert tensor1_file_info.dtype == torch.float32

    assert tensor0_file_info.shape == [1]
    assert tensor1_file_info.shape == [1, 1]

    assert tensor0_file_info.data_offsets == (0, 4)
    assert tensor1_file_info.data_offsets == (4, 8)

    assert meta1.get_storage_size() == 20  # 16 + 4 bytes
    assert len(meta1.aligned_tensor_info_map) == 2  # 2 tensors

    meta2 = TensorsMeta.from_tensors_file(
        "test_models/model3/model-00002-of-00003.safetensors"
    )

    assert meta2.get_file_storage_size() == 34  # 16 + 9 * 2 bytes
    assert len(meta2.file_tensor_info_map) == 2  # 2 tensors

    tensor2_file_info = meta2.file_tensor_info_map["tensor2"]
    tensor3_file_info = meta2.file_tensor_info_map["tensor3"]

    assert tensor2_file_info.dtype == torch.float32
    assert tensor3_file_info.dtype == torch.float16

    assert tensor2_file_info.shape == [2, 2]
    assert tensor3_file_info.shape == [3, 3]

    assert tensor2_file_info.data_offsets == (0, 16)
    assert tensor3_file_info.data_offsets == (16, 34)

    assert meta2.get_storage_size() == 34  # 16 + 9 * 2 bytes
    assert len(meta2.aligned_tensor_info_map) == 2  # 2 tensors

    meta3 = TensorsMeta.from_tensors_file(
        "test_models/model3/model-00003-of-00003.safetensors"
    )

    assert meta3.get_file_storage_size() == 164  # 64 + 5 * 5 * 4 bytes
    assert len(meta3.file_tensor_info_map) == 2  # 2 tensors

    tensor4_file_info = meta3.file_tensor_info_map["tensor4"]
    tensor5_file_info = meta3.file_tensor_info_map["tensor5"]

    assert tensor4_file_info.dtype == torch.float32
    assert tensor5_file_info.dtype == torch.float32

    assert tensor4_file_info.shape == [4, 4]
    assert tensor5_file_info.shape == [5, 5]

    assert tensor4_file_info.data_offsets == (0, 64)
    assert tensor5_file_info.data_offsets == (64, 164)

    assert meta3.get_storage_size() == 164  # 64 + 5 * 5 * 4 bytes
    assert len(meta3.aligned_tensor_info_map) == 2  # 2 tensors
