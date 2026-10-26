import torch

from aegaeon.utils import DTYPE_SIZE, MB, VRAM_ALIGN, prod


def _aligned(addr: int) -> int:
    aligned = addr + VRAM_ALIGN - 1
    return aligned - aligned % VRAM_ALIGN


class Allocator:
    """
    A simplistic GPU memory allocator for managing tensor allocations
    without actually involving CUDA memory management.
    """

    storage: torch.UntypedStorage
    device: torch.device

    def __init__(self, storage: torch.UntypedStorage) -> None:
        assert (
            storage.device.type == "cuda"
        ), "only CUDA devices are supported for Aegaeon Allocator"

        self.device = storage.device
        self.storage = storage
        self.top = 0
        self.bottom = []
        self._size = storage.nbytes()

    @classmethod
    def with_size(
        cls,
        size: int,
        device: torch.device,
    ):
        storage = torch.empty(size, dtype=torch.uint8, device=device).untyped_storage()
        return cls(storage)

    def can_allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        aligned: bool = True,
    ) -> bool:
        alloc_size = prod(shape) * DTYPE_SIZE[dtype]
        start = _aligned(self.top) if aligned else self.top
        end = start + alloc_size
        if end > self.size:
            return False
        return True

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        raw: bool = False,
        aligned: bool = True,
    ) -> torch.Tensor | torch.UntypedStorage:
        alloc_size = prod(shape) * DTYPE_SIZE[dtype]
        start = _aligned(self.top) if aligned else self.top
        end = start + alloc_size
        if end > self._size:
            raise ValueError(
                f"Insufficient memory. "
                f"Tried to allocate {alloc_size / MB:.2f} MB, "
                f"available {(self._size - start) / MB:.2f} MB."
            )

        storage = self.storage[start:end]
        self.top = end

        if raw:
            return storage
        tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(storage)
            .view(dtype=dtype)
            .reshape(shape)
        )
        return tensor

    def push(self):
        self.bottom.append(self.top)

    def pop(self):
        self.top = self.bottom.pop()

    def clear(self):
        self.top = 0

    def size(self) -> int:
        return self._size

    def allocated_size(self) -> int:
        return self.top

    def free_size(self) -> int:
        return self._size - self.top

    @torch.no_grad()
    def relocate_tensors(self, tensors: list[torch.Tensor]) -> int:
        """Compact arena-backed tensors and rebind them to the new storage.

        Tensors are moved in ascending-address order so an in-place downward
        compaction cannot overwrite a source that has not been copied yet.
        Returning the copied byte count lets callers report switch cost without
        scanning tensor contents.
        """
        ordered = sorted(tensors, key=lambda tensor: tensor.data_ptr())
        relocated_bytes = 0
        for tensor in ordered:
            relocated = self.allocate(tensor.shape, tensor.dtype)
            assert isinstance(relocated, torch.Tensor)
            relocated.copy_(tensor, non_blocking=True)
            tensor.set_(relocated)
            relocated_bytes += tensor.numel() * tensor.element_size()

        # This is an O(number-of-tensors) pointer check. A tensor left at its
        # old address would be silently overwritten by the next arena user.
        arena_start = self.storage.data_ptr()
        compact_end = arena_start + self.allocated_size()
        escaped = [
            tensor
            for tensor in ordered
            if tensor.numel()
            and not (
                arena_start <= tensor.data_ptr()
                and tensor.data_ptr() + tensor.numel() * tensor.element_size()
                <= compact_end
            )
        ]
        if escaped:
            raise RuntimeError(
                f"Failed to relocate {len(escaped)} tensors into the compact arena"
            )
        return relocated_bytes


_alloc: Allocator | None = None


def initialize_alloc(size: int, device: torch.device):
    global _alloc
    assert _alloc is None, "initialize_alloc should be invoked only once"
    _alloc = Allocator.with_size(size, device)


def has_alloc() -> bool:
    """Return whether this process currently owns a QuickLoader GPU arena."""
    return _alloc is not None


def release_alloc() -> None:
    """Release the QuickLoader arena so ordinary PyTorch allocations can run.

    Clearing an allocator only rewinds its cursor; it does not release the large
    backing CUDA storage. DefaultModelLoader must not coexist with that arena on
    a 40 GiB A100 when loading an 18 GiB Qwen3.5 checkpoint.
    """
    global _alloc
    _alloc = None


def get_alloc() -> Allocator:
    global _alloc
    assert _alloc is not None, "initialize_alloc has not been invoked"
    return _alloc
