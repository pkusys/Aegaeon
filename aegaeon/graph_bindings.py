"""Archive ABI and CUDA tensor bindings (no device allocations)."""

LAYOUT_ABI_VERSION = 3
ATTENTION_POLICY = "flash-single-split-v1"


def cuda_tensor_bindings(value):
    import torch

    result = {}
    visited = set()

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                result[path] = {
                    "pointer": item.data_ptr(),
                    "shape": list(item.shape),
                    "stride": list(item.stride()),
                    "dtype": str(item.dtype),
                }
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
        elif hasattr(item, "__dict__"):
            visit(vars(item), path)

    visit(value, "")
    return result


def validate_bindings(expected, actual, label):
    if expected != actual:
        keys = sorted(set(expected) | set(actual))
        changed = [key for key in keys if expected.get(key) != actual.get(key)]
        details = {
            key: {"saved": expected.get(key), "actual": actual.get(key)}
            for key in changed[:5]
        }
        raise RuntimeError(f"Foundry {label} CUDA bindings mismatch: {details}")


def single_split_attention(metadata):
    """Keep FlashAttention's split count invariant across eager and capture."""
    for item in metadata.values():
        if hasattr(item, "max_num_splits"):
            item.max_num_splits = 1
    return metadata


def relocate_cuda_buffers(model, allocator):
    """Move CUDA buffer storage into the deterministic model arena.

    Copy each underlying storage once and retain the original shape, stride,
    offset and aliases. Tensor objects are rebound in place, so shared RoPE
    modules and any existing references see the relocated buffer too.
    """
    import torch

    arena_start = allocator.storage.data_ptr()
    arena_end = arena_start + allocator.storage.nbytes()
    relocated = {}
    copied = 0
    for _, buffer in sorted(model.named_buffers(), key=lambda pair: pair[0]):
        if not buffer.is_cuda or not buffer.numel():
            continue
        storage = buffer.untyped_storage()
        pointer = storage.data_ptr()
        if arena_start <= pointer and pointer + storage.nbytes() <= arena_end:
            continue
        key = (pointer, storage.nbytes())
        if key not in relocated:
            target = allocator.allocate((storage.nbytes(),), torch.uint8)
            source = torch.empty(0, dtype=torch.uint8, device=buffer.device).set_(
                storage
            )
            target.copy_(source)
            relocated[key] = target.untyped_storage()
            copied += storage.nbytes()
        buffer.set_(
            relocated[key], buffer.storage_offset(), buffer.size(), buffer.stride()
        )
    return copied
