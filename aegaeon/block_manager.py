import math
import ray
import asyncio
import torch
import os

from typing import (
    List,
    Callable,
    Tuple,
    Dict,
    TypeAlias,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from enum import Enum
from dataclasses import dataclass
from collections import deque, defaultdict

from aegaeon.config import BLOCK_SIZE, ParallelConfig
from aegaeon.cache_transfer import (
    byte_extent_for_kernel_blocks,
    kernel_block_extent_for_attention_page,
    logical_blocks_for_tokens,
    num_skipped_sliding_window_blocks,
    sliding_window_resident_block_bound,
)
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.utils import Counter, CudaRTLibrary, rebuild_cuda_event, prod
from aegaeon.device_registry import DeviceSpec
from aegaeon.model_registry import ModelSpec

logger = init_logger(__name__)

# Sleep for this many seconds between the daemon runs
SLEEP_IN_DAEMON_LOOP = 8

# Sleep for this many seconds between attempts to reclaim a move event
SLEEP_WHEN_RECLAIM_PENDS = 0.01

"""
XXX: migration between GPU memory is currently not implemented. The
following description talks about migration at the principle level
but not its actual implementation.

TODO: the whole block_shape mechanism currently does not consider model
parallelism. Add support for it once the non-parallel case runs.

## Synchonization

The block manager need to deal with 4 types of synchronization:

Requirement 1: When calling step(), we need to ensure that all blocks that
    are related to the request are on GPU, and have finished movement if
    we've issued a swap-in/migration operation on them before.

Requirement 2: When issuing movements, we need to ensure that the target blocks
    (i.e. blocks that we are copying to) are free. If there were another
    movements which use them as source, we must wait for those operations
    to finish.

Requirement 3: When issuing movements, the source blocks must be ready.
    For example, if we have issued swapping-in on request #0 before and now 
    are about to migrate it, then the swap-in operation must have finished.

Requirement 4: When calling switch(), we need to ensure that all blocks that
    are related to previous requests are not on GPU, and have finished movement if
    we've issued a swap-out/migration operation on them before.

To facilitate these synchronization efficiently, we rely on CUDA events.
After every movement operation, we push a CUDA event into the corresponding
CUDA stream. We maintain a dict called `move_events`, which maps request_ids,
to a list of MoveEvents which include the cuda events related to that request. 
These events are then dispatched by the block manager to workers, which synchronizes
their streams correctly for the events. To be exact:

1. When calling step() / switch(), we iterate through all requests in the batch, retrieve
the CUDA event from the worker-local `move_event_table` (populated when swap_blocks
and migrate_blocks are invoked on the worker side), and wait for it to finish. 
This ensures Requirement 1 / 4.

2. When issuing a movement, we call event.wait() on the corresponding movement
event and use the corresponding stream (i.e., if we are swapping-in then it is the
swap-in stream) as the argument. Wait() makes all future work submitted to the given 
stream wait for this event, so the movement operation that we're issuing will wait 
for the previous movement operation to finish.
This ensures Requirement 3.

3. For Requirement 2, a possible solution is to mark the CUDA event of every block
and wait for them to finish. However, this is not efficient. Instead, for
each cache space (CPU/GPUs), we maintain two lists: `free_blocks_list` and
`move_blocks_list`. The former contains blocks that are we are sure to
be free (it does not contain any useful data), and the latter contains blocks
involving movement operations that may not have finished. 

Model switches (and thus registering a new GPU cache space) serve as periodical
barriers for the block manager to reclaim blocks from move_blocks_list
synchronously: all movements to a GPU space must have finished before a switch,
and all movements from it no longer relevant. If unfortunately the free_blocks_list
is still insufficient at some allocation, `_get_free_blocks()` will perform
reclaimation itself by gathering events and asynchronously querying them.
This effectively pends the corresponding task without choking the entire
control plane (for example, other engines can still operate unless they
also want to allocate on this blocked location). For details involving race
conditions from workers freeing events and multiple tasks reclaiming the
same move_blocks_list, refer to `_reclaim_XXX()`.
This ensures Requirement 2 with a small overall cost. 
"""

""">=0 for GPU engine ID, -1 for CPU"""
BlockLocation: TypeAlias = int

BLOCK_LOCATION_CPU: BlockLocation = -1

# vLLM v1 reserves physical block 0 as NULL_BLOCK_ID. Attention metadata uses
# it for padded block-table entries, and recurrent GDN layers read column zero
# as their state-page index. It must never be allocated to a real request.
VLLM_NULL_BLOCK_ID = 0

# Released sliding-window pages must remain as logical table columns so the
# attention backend can keep indexing by absolute token position. Use a
# negative control-plane sentinel because CPU block zero is a valid allocation;
# workers translate it to vLLM's zero-filled physical null block.
LOGICAL_NULL_BLOCK_ID = -1


def is_gpu(location: BlockLocation) -> bool:
    return location >= 0


@dataclass
class CacheGroupConfig:
    """Scheduler-visible allocation policy for one vLLM KV-cache group.

    ``block_size`` is vLLM's KV-manager (physical) block size, not necessarily
    the attention kernel's block size. Enlarging it is what avoids padding one
    small attention page up to the size of a large GDN state page.
    """

    kind: str
    block_size: int
    kernel_block_size: int
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.kernel_block_size <= 0:
            raise ValueError(
                "Cache block sizes must both be positive: "
                f"physical={self.block_size}, kernel={self.kernel_block_size}"
            )
        if self.kind == "attention":
            if self.block_size % self.kernel_block_size:
                raise ValueError(
                    f"Attention kernel block size {self.kernel_block_size} must "
                    f"divide physical block size {self.block_size}"
                )
        elif self.kind == "mamba_none":
            if self.block_size != self.kernel_block_size:
                raise ValueError(
                    "Mamba/GDN state pages cannot be split into kernel blocks"
                )
            if self.sliding_window is not None:
                raise ValueError("Mamba/GDN cache groups cannot have a sliding window")
        else:
            raise NotImplementedError(f"Unsupported cache group kind: {self.kind}")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive, got {self.sliding_window}"
            )

    def blocks_for_tokens(self, num_tokens: int) -> int:
        """Return a safe upper bound on resident physical pages.

        A sliding-window query can straddle two physical pages. The worst case
        is one current page plus the pages covering the preceding
        ``sliding_window - 1`` tokens. Schedulers need this resident bound;
        the logical table still grows with the absolute sequence position and
        is sized by :meth:`table_blocks_for_tokens`.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if self.kind == "attention":
            if self.sliding_window is None:
                return self.table_blocks_for_tokens(num_tokens)
            return sliding_window_resident_block_bound(
                num_tokens, self.block_size, self.sliding_window
            )
        if self.kind == "mamba_none":
            # Prefix caching is disabled: one recurrent-state page per request.
            return 1
        raise NotImplementedError(f"Unsupported cache group kind: {self.kind}")

    def table_blocks_for_tokens(self, num_tokens: int) -> int:
        """Return logical block-table columns required for absolute positions."""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if self.kind == "attention":
            return logical_blocks_for_tokens(num_tokens, self.block_size)
        if self.kind == "mamba_none":
            return 1
        raise NotImplementedError(f"Unsupported cache group kind: {self.kind}")

    def num_skipped_blocks(self, processed_computed_tokens: int) -> int:
        """Return leading pages wholly outside the next attention window.

        ``processed_computed_tokens`` is the committed KV-token count and the
        absolute position of the next decode token. A window of W therefore
        retains W-1 committed tokens for that next query. Only complete
        physical pages can be released.
        """
        if processed_computed_tokens < 0:
            raise ValueError(
                "processed_computed_tokens must be non-negative, got "
                f"{processed_computed_tokens}"
            )
        if self.kind != "attention" or self.sliding_window is None:
            return 0
        return num_skipped_sliding_window_blocks(
            processed_computed_tokens,
            self.block_size,
            self.sliding_window,
        )

    def block_is_skipped(
        self, processed_computed_tokens: int, logical_block_index: int
    ) -> bool:
        if logical_block_index < 0:
            raise ValueError(
                f"logical_block_index must be non-negative, got {logical_block_index}"
            )
        return logical_block_index < self.num_skipped_blocks(processed_computed_tokens)

    @property
    def kernel_blocks_per_physical_block(self) -> int:
        return self.block_size // self.kernel_block_size

    def valid_kernel_blocks(self, cached_tokens: int, block_index: int) -> int:
        """Return the live kernel-page count in one physical page.

        Use :meth:`kernel_block_extent` when its start offset is also needed.
        """
        if cached_tokens < 0:
            raise ValueError(f"cached_tokens must be non-negative, got {cached_tokens}")
        if block_index < 0:
            raise ValueError(f"block_index must be non-negative, got {block_index}")
        if self.kind == "mamba_none":
            return 1

        _, count = self.kernel_block_extent(cached_tokens, block_index)
        return count

    def kernel_block_extent(
        self, cached_tokens: int, block_index: int
    ) -> Tuple[int, int]:
        """Return ``(first kernel page, live page count)`` for swap."""
        if self.kind == "mamba_none":
            return (0, 1)
        return kernel_block_extent_for_attention_page(
            cached_tokens,
            block_index,
            self.block_size,
            self.kernel_block_size,
            self.sliding_window,
        )


@dataclass(frozen=True)
class CacheGroupRuntimeConfig:
    """Execution-side block-size contract for one KV-cache group.

    ``kv_block_size`` is the token span owned by one physical block allocated
    by the cache manager. ``kernel_block_size`` is the smaller granularity used
    by the attention backend. Hybrid models may deliberately make the former
    larger so that an attention page occupies the same bytes as a large
    recurrent-state page without padding every 16-token attention block.
    """

    kind: str
    kv_block_size: int
    kernel_block_size: int
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kv_block_size <= 0 or self.kernel_block_size <= 0:
            raise ValueError(
                "KV-manager and kernel block sizes must both be positive: "
                f"kv={self.kv_block_size}, kernel={self.kernel_block_size}"
            )
        if self.kind == "attention":
            if self.kv_block_size % self.kernel_block_size:
                raise ValueError(
                    f"Attention kernel block size {self.kernel_block_size} must "
                    f"divide KV-manager block size {self.kv_block_size}"
                )
        elif self.kind == "mamba_none":
            if self.kv_block_size != self.kernel_block_size:
                raise ValueError(
                    "Mamba/GDN state pages cannot be split into virtual kernel "
                    "blocks"
                )
            if self.sliding_window is not None:
                raise ValueError("Mamba/GDN cache groups cannot have a sliding window")
        else:
            raise NotImplementedError(f"Unsupported cache group kind: {self.kind}")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive, got {self.sliding_window}"
            )

    @property
    def blocks_per_kv_block(self) -> int:
        """Number of kernel blocks represented by one physical block id."""
        return self.kv_block_size // self.kernel_block_size

    def bytes_for_kernel_blocks(
        self, page_size_bytes: int, valid_kernel_blocks: int
    ) -> int:
        """Translate a valid kernel-page prefix to a backing-tensor prefix."""
        _, extent_bytes = self.byte_extent_for_kernel_blocks(
            page_size_bytes, 0, valid_kernel_blocks
        )
        return extent_bytes

    def byte_extent_for_kernel_blocks(
        self,
        page_size_bytes: int,
        first_kernel_block: int,
        num_kernel_blocks: int,
    ) -> Tuple[int, int]:
        return byte_extent_for_kernel_blocks(
            page_size_bytes,
            self.blocks_per_kv_block,
            first_kernel_block,
            num_kernel_blocks,
        )

    def to_kernel_block_ids(self, kv_block_ids: Sequence[int]) -> List[int]:
        """Expand physical block ids into the ids used by attention kernels.

        This mirrors vLLM's virtual-block mapping. For example, with a
        272-token physical block and a 16-token kernel block, physical id 3
        maps to kernel ids 51 through 67.
        """
        if self.kind != "attention":
            return list(kv_block_ids)
        ratio = self.blocks_per_kv_block
        kernel_ids: List[int] = []
        for block_id in kv_block_ids:
            if block_id < 0:
                kernel_ids.extend([VLLM_NULL_BLOCK_ID] * ratio)
            else:
                kernel_ids.extend(block_id * ratio + offset for offset in range(ratio))
        return kernel_ids

    def slot_for_position(self, kv_block_ids: Sequence[int], position: int) -> int:
        """Map a token position to a slot in the kernel-shaped cache view."""
        if self.kind != "attention":
            raise ValueError(f"Cache group {self.kind} has no token slot mapping")
        if position < 0:
            raise ValueError(f"Token position must be non-negative, got {position}")
        table_index = position // self.kv_block_size
        if table_index >= len(kv_block_ids):
            raise IndexError(
                f"Position {position} needs physical table column {table_index}, "
                f"but the table has {len(kv_block_ids)} columns"
            )
        physical_id = kv_block_ids[table_index]
        if physical_id < 0:
            raise RuntimeError(
                f"Position {position} maps to released logical table column "
                f"{table_index}"
            )
        physical_offset = position % self.kv_block_size
        kernel_id = (
            physical_id * self.blocks_per_kv_block
            + physical_offset // self.kernel_block_size
        )
        return kernel_id * self.kernel_block_size + (
            physical_offset % self.kernel_block_size
        )


@dataclass
class BlockTable:
    # Location of the blocks
    location: BlockLocation
    # Block shape
    block_shape: Tuple[int, ...]
    # Allocation policy and block ids, indexed by vLLM cache-group id.
    cache_groups: Tuple[CacheGroupConfig, ...]
    group_blocks: List[List[int]]

    @property
    def blocks(self) -> List[int]:
        """Flatten physical block ids across all cache groups.

        Physical ids are unique across groups because every group allocates
        from the same pool. This flattened view is used by swap/free paths;
        model execution must use ``group_blocks`` instead.
        """
        return [
            block
            for blocks in self.group_blocks
            for block in blocks
            if block != LOGICAL_NULL_BLOCK_ID
        ]

    def replace_flat_blocks(self, blocks: List[int]) -> None:
        expected = sum(
            block != LOGICAL_NULL_BLOCK_ID
            for group in self.group_blocks
            for block in group
        )
        if len(blocks) != expected:
            raise ValueError(f"Expected {expected} blocks, got {len(blocks)}")
        cursor = 0
        replacement: List[List[int]] = []
        for group in self.group_blocks:
            replacement_group: List[int] = []
            for old_block in group:
                if old_block == LOGICAL_NULL_BLOCK_ID:
                    replacement_group.append(LOGICAL_NULL_BLOCK_ID)
                else:
                    replacement_group.append(blocks[cursor])
                    cursor += 1
            replacement.append(replacement_group)
        self.group_blocks = replacement

    def release_skipped_blocks(self, processed_computed_tokens: int) -> List[int]:
        """Null and return resident pages wholly before each group's window."""
        released: List[int] = []
        for group, blocks in zip(self.cache_groups, self.group_blocks):
            skipped_blocks = min(
                group.num_skipped_blocks(processed_computed_tokens), len(blocks)
            )
            for block_index in range(skipped_blocks):
                block_id = blocks[block_index]
                if block_id == LOGICAL_NULL_BLOCK_ID:
                    continue
                released.append(block_id)
                blocks[block_index] = LOGICAL_NULL_BLOCK_ID
        return released

    def transfer_entries(
        self, processed_computed_tokens: int
    ) -> List[Tuple[int, int, int, int, int]]:
        """Return retained pages as ``(group, index, id, first, count)``.

        Every returned page needs a physical allocation at the destination.
        ``count`` may be zero when a retained page has no cached history to
        copy but will receive the next token, notably for ``sliding_window=1``.
        The caller must omit those entries from DMA while still assigning their
        destination block ids. The source table is not mutated.
        """
        entries: List[Tuple[int, int, int, int, int]] = []
        for group_id, (group, blocks) in enumerate(
            zip(self.cache_groups, self.group_blocks)
        ):
            for block_index, block_id in enumerate(blocks):
                if block_id == LOGICAL_NULL_BLOCK_ID or group.block_is_skipped(
                    processed_computed_tokens, block_index
                ):
                    continue
                first_block, block_count = group.kernel_block_extent(
                    processed_computed_tokens, block_index
                )
                entries.append(
                    (
                        group_id,
                        block_index,
                        block_id,
                        first_block,
                        block_count,
                    )
                )
        return entries

    def moved_group_blocks(
        self,
        processed_computed_tokens: int,
        target_block_ids: Sequence[int],
    ) -> List[List[int]]:
        """Build a destination logical table from a compact live-page list."""
        entries = self.transfer_entries(processed_computed_tokens)
        if len(entries) != len(target_block_ids):
            raise ValueError(
                f"Expected {len(entries)} target blocks, got {len(target_block_ids)}"
            )
        moved = [[LOGICAL_NULL_BLOCK_ID] * len(blocks) for blocks in self.group_blocks]
        for entry, target_id in zip(entries, target_block_ids):
            group_id, block_index, _, _, _ = entry
            moved[group_id][block_index] = target_id
        return moved

    def swap_metadata(
        self, cached_tokens: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return metadata for non-empty DMA extents.

        Destination-only zero-copy entries are deliberately excluded. Move
        planning uses :meth:`transfer_entries` directly so it can allocate
        those pages before filtering the actual copy lists.
        """
        entries = [
            entry
            for entry in self.transfer_entries(cached_tokens)
            if entry[4] > 0
        ]
        return (
            [entry[0] for entry in entries],
            [entry[3] for entry in entries],
            [entry[4] for entry in entries],
        )


@dataclass(eq=False)
class MoveEvent:
    # Serial for ordering move events.
    serial: int
    # Requests involved in this movement
    requests: List[Request]
    # Block shape
    block_shape: Tuple[int, ...]
    # Source block location
    source_location: BlockLocation
    # Target block location
    target_location: BlockLocation
    # List of source blocks
    source_blocks: List[int]

    # Corresponding CUDA events (ray futures of event handles, one for each Worker)
    event_handle_refs: List[ray.ObjectRef]
    _event_handles: Optional[list] = None

    async def _gather_event_handles(self) -> None:
        if self._event_handles is None:
            handles = await asyncio.gather(*self.event_handle_refs)
            # this checks for multiple callers
            if self._event_handles is None:
                self._event_handles = handles

    def event_handles(self) -> Optional[list]:
        return self._event_handles

    def is_swap(self) -> bool:
        return is_gpu(self.source_location) != is_gpu(self.target_location)

    def engine_id(self) -> int:
        """Get the ID of the engine that is responsible for the movement.

        There are 3 types of movement: (1) swap in, (2) swap out, and (3)
        migrate in. For (1) and (2) this is the GPU side; for (3) its the
        target GPU side.
        """
        match (is_gpu(self.source_location), is_gpu(self.target_location)):
            case (_, True):
                return self.target_location
            case (True, False):
                return self.source_location
            case (False, False):
                raise NotImplementedError


class BlockManager:
    """A Block Manager that maintains all key-value caches at the block-level.

    The block manager is first initialized only for the unified CPU swap space.
    GPU cache spaces can then be registered (and replaced) subsequently.

    For subroutines and algorithms related to swapping, please refer to
    the header comment block.

    ## Event loops
    - Daemon
        - log_block_usage
    """

    def __init__(
        self,
        cpu_num_slabs: int,
        cpu_slab_size_bytes: int,
    ):
        self.block_size = BLOCK_SIZE
        self.dtype_size = 2
        self.cudart = CudaRTLibrary()
        self.event_counter = Counter()
        self.device_type = None
        self.parallel_config = None

        # CPU allocator
        self.cpu_num_slabs = cpu_num_slabs
        self.cpu_slab_size_bytes = cpu_slab_size_bytes

        # Mapping: block shape -> list of blocks
        self.cpu_free_blocks: Dict[Tuple[int, ...], deque[int]] = defaultdict(deque)
        self.cpu_free_slabs: list[int] = [i for i in range(self.cpu_num_slabs)]

        # Mapping: slab index -> number of used blocks within
        self.cpu_slab_usage: list[int] = [0 for _ in range(self.cpu_num_slabs)]
        # Mapping: slab index -> shape of blocks in the slab
        self.cpu_block_shape: list[Tuple[int, ...]] = [
            None for _ in range(self.cpu_num_slabs)
        ]

        # GPU allocator
        # Mapping: engine_id -> block_shape
        self.gpu_block_shape: Dict[int, Tuple[int, ...]] = {}
        # Mapping: engine_id -> num_max_blocks
        self.gpu_num_max_blocks: Dict[int, int] = {}
        # Mapping: engine_id -> scheduler-visible cache-group policies
        self.gpu_cache_groups: Dict[int, Tuple[CacheGroupConfig, ...]] = {}
        # Mapping: engine_id -> free_blocks
        self.gpu_free_blocks: Dict[int, List[int]] = {}
        # Mapping: engine_id -> async func
        self.remote_callables: Dict[int, Callable] = {}

        # Movement
        # Mapping: block_shape -> list of MoveEvents (from cpu)
        self.move_events_cpu: Dict[Tuple[int, ...], deque[MoveEvent]] = defaultdict(
            deque
        )
        # Mapping: block_shape -> number of moving blocks (from cpu)
        self.move_cpu_num_blocks: Dict[Tuple[int, ...], int] = defaultdict(int)
        # Mapping: block_shape -> lastest reclaim event
        self.move_cpu_locks: Dict[Tuple[int, ...], Optional[asyncio.Event]] = (
            defaultdict(lambda: None)
        )
        # Mapping: engine_id -> list of MoveEvents (from gpu)
        self.move_events_gpu: Dict[int, deque[MoveEvent]] = defaultdict(deque)
        # Mapping: engine_id -> number of moving blocks (from gpu)
        self.move_gpu_num_blocks: Dict[int, int] = {}
        # Mapping: engine_id -> lastest reclaim event
        self.move_gpu_locks: Dict[int, Optional[asyncio.Event]] = defaultdict(
            lambda: None
        )
        # Mapping: request_id -> latest CUDA events (as ray future), one for each Worker
        self.latest_events: Dict[int, List[ray.ObjectRef]] = {}
        # List of freed request IDs whose resources are not yet cleared in workers
        self.zombie_request_ids: List[int] = []

        # Global block table
        # Mapping: request_id => BlockTable
        self.block_tables: Dict[int, BlockTable] = {}

    def _set_device(self, device_type: DeviceSpec):
        self.device_type = device_type

    def _set_parallel_config(self, parallel_config: ParallelConfig):
        self.parallel_config = parallel_config

    def _get_block_size_bytes(self, block_shape: Tuple[int, ...]) -> int:
        return prod(block_shape) * self.dtype_size

    def _get_slab_range(self, slab: int, block_size_bytes: int) -> Tuple[int, int]:
        start = math.ceil(slab * self.cpu_slab_size_bytes / block_size_bytes)
        end = math.floor((slab + 1) * self.cpu_slab_size_bytes / block_size_bytes)
        return (start, end)

    def _reclaim_gpu_move_blocks_once(
        self,
        move_event: MoveEvent,
    ) -> int:
        self.move_gpu_num_blocks[move_event.source_location] -= len(
            move_event.source_blocks
        )
        self.gpu_free_blocks[move_event.source_location].extend(
            move_event.source_blocks
        )
        logger.debug(
            f"Reclaimed {len(move_event.source_blocks)} blocks for engine #{move_event.source_location}"
        )
        return len(move_event.source_blocks)

    def _reclaim_cpu_move_blocks_once(
        self,
        move_event: MoveEvent,
    ) -> int:
        block_size_bytes = self._get_block_size_bytes(move_event.block_shape)
        self.move_cpu_num_blocks[move_event.block_shape] -= len(
            move_event.source_blocks
        )
        self.cpu_free_blocks[move_event.block_shape].extend(move_event.source_blocks)
        # ..update slab usage
        for block in move_event.source_blocks:
            slab = block * block_size_bytes // self.cpu_slab_size_bytes
            self.cpu_slab_usage[slab] -= 1
        return len(move_event.source_blocks)

    def reset(self):
        self.event_counter = Counter()

        self.cpu_free_blocks.clear()
        self.cpu_free_slabs = [i for i in range(self.cpu_num_slabs)]

        self.cpu_slab_usage = [0 for _ in range(self.cpu_num_slabs)]
        self.cpu_block_shape = [None for _ in range(self.cpu_num_slabs)]

        self.gpu_block_shape.clear()
        self.gpu_num_max_blocks.clear()
        self.gpu_cache_groups.clear()
        self.gpu_free_blocks.clear()

        self.remote_callables.clear()

        self.move_events_cpu.clear()
        self.move_cpu_num_blocks.clear()
        self.move_cpu_locks.clear()
        self.move_events_gpu.clear()
        self.move_gpu_num_blocks.clear()
        self.move_gpu_locks.clear()
        self.latest_events.clear()
        self.zombie_request_ids.clear()

        self.block_tables.clear()

    def register_gpu_space(
        self,
        engine_id: int,
        num_max_blocks: int,
        block_shape: Tuple[int, ...],
        cache_groups: List[Tuple[str, int, int, Optional[int]]],
        remote_callable: Callable,
    ) -> None:
        """Register a new GPU cache space with the given config.

        Also reclaims all blocks moving from and to the old cache space.
        """

        if self.gpu_free_blocks.get(engine_id, None) is not None:
            # Replacing an old GPU space; make sure that no requests are still in the old space
            for req_id, block_table in self.block_tables.items():
                if block_table.location == engine_id:
                    raise ValueError(
                        f"Error registering new GPU cache space for engine_id {engine_id}; "
                        f"Request #{req_id} still resides in the cache."
                    )

        if engine_id in self.move_events_gpu:
            self.move_events_gpu[engine_id].clear()
        self.move_gpu_num_blocks[engine_id] = 0

        if (old_block_shape := self.gpu_block_shape.get(engine_id, None)) is not None:
            # ..reclaim CPU blocks moving to the old space
            move_events_cpu = self.move_events_cpu[old_block_shape]
            num_events = 0
            num_blocks = 0
            for _ in range(len(move_events_cpu)):
                move_event = move_events_cpu[0]

                if move_event.target_location == engine_id:
                    move_events_cpu.popleft()
                    num_events += 1
                    num_blocks += self._reclaim_cpu_move_blocks_once(move_event)
                else:
                    move_events_cpu.rotate(-1)

            if num_events > 0:
                logger.debug(
                    f"Reclaimed {num_events} events, {num_blocks} CPU blocks for block shape {block_shape}"
                )
                # self._reclaim_cpu_slabs()

        if num_max_blocks <= 1:
            raise ValueError(
                "GPU cache must contain at least one usable block in addition "
                "to vLLM's reserved NULL_BLOCK_ID"
            )
        self.gpu_block_shape[engine_id] = block_shape
        normalized_groups = tuple(
            CacheGroupConfig(
                kind=kind,
                block_size=int(block_size),
                kernel_block_size=int(kernel_block_size),
                sliding_window=(
                    None if sliding_window is None else int(sliding_window)
                ),
            )
            for kind, block_size, kernel_block_size, sliding_window in cache_groups
        )
        if not normalized_groups:
            raise ValueError("GPU cache must expose at least one cache group")
        self.gpu_cache_groups[engine_id] = normalized_groups
        # Match vLLM's BlockPool invariant: physical block zero is the null
        # block, so expose only num_max_blocks - 1 blocks to Aegaeon's
        # scheduler and allocator. Keeping the physical IDs unchanged is
        # required because workers index the raw cache tensors directly.
        self.gpu_num_max_blocks[engine_id] = num_max_blocks - 1
        self.gpu_free_blocks[engine_id] = list(
            range(VLLM_NULL_BLOCK_ID + 1, num_max_blocks)
        )
        self.remote_callables[engine_id] = remote_callable

    async def start_event_loop(self):
        async def daemon_loop():
            while True:
                await asyncio.sleep(SLEEP_IN_DAEMON_LOOP)
                self.log_block_usage()
                self.clear_zombie_requests()

        await daemon_loop()

    async def _reclaim_blocks(
        self,
        move_events: deque[MoveEvent],
        target_serial: int,
        reclaim_callback: Callable[[MoveEvent], int],
    ) -> Tuple[int, int]:
        num_events = 0
        num_blocks = 0
        while len(move_events) > 0 and move_events[0].serial <= target_serial:
            # XXX: pop the move_event out; this ensures that multiple concurrent
            # invocations of this function does not access the same event more than once
            move_event = move_events.popleft()
            events: Optional[list] = None
            while True:
                if any(req.is_finished for req in move_event.requests):
                    # XXX: request finished -> this event must have finished;
                    # also, the CUDA event may have been freed, so we must NOT rebuild
                    break

                if (handles := move_event.event_handles()) is None:
                    # ..remote not ready
                    await asyncio.wait(
                        [move_event._gather_event_handles()],
                        timeout=SLEEP_WHEN_RECLAIM_PENDS,
                    )
                    continue

                if events is None:
                    # ..get CUDA events
                    events = [
                        rebuild_cuda_event(self.cudart, handle, is_worker=False)
                        for handle in handles
                    ]

                if all(self.cudart.cudaEventQuery(event) for event in events):
                    # ..all events done
                    break
                else:
                    await asyncio.sleep(SLEEP_WHEN_RECLAIM_PENDS)
            if events is not None:
                # ..clean up
                for event in events:
                    self.cudart.cudaEventDestroy(event)

            # ..free to reclaim this move event!
            num_events += 1
            num_blocks += reclaim_callback(move_event)

        return (num_events, num_blocks)

    async def _reclaim_gpu_move_blocks(
        self,
        location: BlockLocation,
    ):
        """Reclaim all moved GPU blocks at the given location.

        This function may asynchronously spin until all move events from the location
        are reclaimed.
        """
        move_events = self.move_events_gpu[location]
        if len(move_events) > 0:
            # ..reclaiming some new events
            lock = asyncio.Event()
            target_serial = move_events[-1].serial
            self.move_gpu_locks[location] = lock
        else:
            lock = self.move_gpu_locks[location]
            if lock is not None:
                # ..some other task is reclaiming already; wait for them
                await lock.wait()
            return

        num_events, num_blocks = await self._reclaim_blocks(
            move_events, target_serial, self._reclaim_gpu_move_blocks_once
        )
        logger.debug(
            f"Reclaimed {num_events} events, {num_blocks} GPU blocks for engine #{location}"
        )
        lock.set()

    async def _reclaim_cpu_move_blocks(
        self,
        block_shape: Tuple[int, ...],
    ):
        """Reclaim all moved CPU blocks with the given shape.

        This function may asynchronously spin until all move events from the location
        are reclaimed.
        """
        num_events = 0
        num_blocks = 0
        move_events = self.move_events_cpu[block_shape]
        if len(move_events) > 0:
            # ..reclaiming some new events
            lock = asyncio.Event()
            target_serial = move_events[-1].serial
            self.move_cpu_locks[block_shape] = lock
        else:
            lock = self.move_cpu_locks[block_shape]
            if lock is not None:
                # ..some other task is reclaiming already; wait for them
                await lock.wait()
            return

        num_events, num_blocks = await self._reclaim_blocks(
            move_events, target_serial, self._reclaim_cpu_move_blocks_once
        )

        logger.debug(
            f"Reclaimed {num_events} events, {num_blocks} CPU blocks for block shape {block_shape}"
        )
        self._reclaim_cpu_slabs()
        lock.set()

    def _reclaim_cpu_slabs(self) -> bool:
        """Reclaim unused cpu slabs, adding them to the free slab list."""
        reclaimed = 0
        for slab, usage in enumerate(self.cpu_slab_usage):
            block_shape = self.cpu_block_shape[slab]
            if usage == 0 and block_shape is not None:
                # Reclaim this slab
                reclaimed += 1
                block_size_bytes = self._get_block_size_bytes(block_shape)

                cpu_free_blocks = self.cpu_free_blocks[block_shape]
                blocks_start, blocks_end = self._get_slab_range(slab, block_size_bytes)
                for _ in range(len(cpu_free_blocks)):
                    block = cpu_free_blocks[0]
                    # ..filters those within the range
                    if blocks_start <= block and block < blocks_end:
                        cpu_free_blocks.popleft()
                    else:
                        cpu_free_blocks.rotate(-1)
                self.cpu_free_slabs.append(slab)
                self.cpu_block_shape[slab] = None
        logger.debug(f"Reclaimed {reclaimed} CPU slabs")
        return reclaimed > 0

    async def _get_free_blocks(
        self,
        num_blocks: int,
        location: BlockLocation,
        block_shape: Tuple[int, ...],
    ) -> List[int]:
        """Get free blocks at the given location.

        If free blocks are not immediately available, reclaim move events, which
        may spin and block the current caller task.
        """
        block_size_bytes = self._get_block_size_bytes(block_shape)
        if is_gpu(location):
            # GPU allocation
            assert location in self.gpu_free_blocks, f"unknown gpu location {location}"

            gpu_free_blocks = self.gpu_free_blocks[location]
            if len(gpu_free_blocks) < num_blocks:
                # Need to "flush" moving GPU blocks, i.e. make sure all
                # move-out operations have finished for the engine, then move their source blocks
                # to the free list.
                await self._reclaim_gpu_move_blocks(location)
                if len(gpu_free_blocks) < num_blocks:
                    raise ResourceWarning(
                        f"not enough free blocks on GPU, requested {num_blocks}, available {len(gpu_free_blocks)}"
                    )

            blocks = gpu_free_blocks[:num_blocks]
            gpu_free_blocks[:] = gpu_free_blocks[num_blocks:]

        else:
            # rec = num_blocks
            # num_blocks = 1
            # CPU allocation
            cpu_free_blocks = self.cpu_free_blocks[block_shape]

            # Always try reserving new slabs first; if that fails, reclaim moving blocks.
            def ensure_slab_blocks(
                block_size_bytes: int, num_blocks_needed: int
            ) -> bool:
                # Test if free slabs contains the given number of blocks
                if num_blocks_needed <= 0:
                    return True
                for slab in self.cpu_free_slabs:
                    assert self.cpu_block_shape[slab] is None
                    blocks_start, blocks_end = self._get_slab_range(
                        slab, block_size_bytes
                    )
                    num_blocks_needed -= blocks_end - blocks_start
                    if num_blocks_needed <= 0:
                        return True
                return False

            if not ensure_slab_blocks(
                block_size_bytes, num_blocks - len(cpu_free_blocks)
            ):
                # ..cannot allocate using existing free blocks and slabs
                # Need to "flush" moving CPU blocks, i.e. make sure all
                # moving-in operations have finished, then move their source blocks
                # to the free list
                await self._reclaim_cpu_move_blocks(block_shape)

                # XXX: cpu_free_blocks is potentailly shrinked by _reclaim_cpu_move_blocks
                if not ensure_slab_blocks(
                    block_size_bytes, num_blocks - len(cpu_free_blocks)
                ):
                    raise ResourceWarning(
                        f"not enough free blocks on CPU, requested {num_blocks}, available {len(cpu_free_blocks)} "
                        f"and {len(self.cpu_free_slabs)} slabs"
                    )

            # ..use free slabs if needed
            while len(cpu_free_blocks) < num_blocks:
                slab = self.cpu_free_slabs.pop()
                blocks_start, blocks_end = self._get_slab_range(slab, block_size_bytes)
                self.cpu_block_shape[slab] = block_shape
                cpu_free_blocks.extend(range(blocks_start, blocks_end))

            blocks = []
            for _ in range(num_blocks):
                block = cpu_free_blocks.popleft()
                # ..also update slab usage
                slab = block * block_size_bytes // self.cpu_slab_size_bytes
                self.cpu_slab_usage[slab] += 1
                blocks.append(block)

            # Hack: Return the same block for multiple times
            # blocks = blocks * rec

        return blocks

    async def _get_free_blocks_batched(
        self,
        num_blocks_list: List[int],
        location: BlockLocation,
        block_shape: Tuple[int, ...],
    ) -> List[List[int]]:
        free_blocks = await self._get_free_blocks(
            sum(num_blocks_list), location, block_shape
        )
        cur = 0
        ret = []
        for num in num_blocks_list:
            ret.append(free_blocks[cur : cur + num])
            cur += num
        return ret

    def get_gpu_num_max_blocks(self, engine_id: int) -> int:
        assert engine_id in self.gpu_num_max_blocks, f"unknown gpu location {engine_id}"
        return self.gpu_num_max_blocks[engine_id]

    def get_gpu_num_max_blocks_from_model(
        self, engine_id: int, model: ModelSpec
    ) -> int:
        assert engine_id in self.gpu_num_max_blocks, f"unknown gpu location {engine_id}"
        # The worker has already planned the cache from the model's real
        # per-layer specs. The legacy homogeneous model formula is incorrect
        # for hybrid cache groups, so the registered physical pool is the
        # authoritative capacity.
        return self.gpu_num_max_blocks[engine_id]

    @staticmethod
    def _get_group_blocks_needed(
        num_tokens: int, cache_groups: Tuple[CacheGroupConfig, ...]
    ) -> List[int]:
        return [group.blocks_for_tokens(num_tokens) for group in cache_groups]

    @staticmethod
    def _get_group_table_blocks_needed(
        num_tokens: int, cache_groups: Tuple[CacheGroupConfig, ...]
    ) -> List[int]:
        return [group.table_blocks_for_tokens(num_tokens) for group in cache_groups]

    def get_num_blocks_needed_for_length(self, engine_id: int, num_tokens: int) -> int:
        """Return physical pool blocks needed across all cache groups."""
        assert engine_id in self.gpu_cache_groups, f"unknown gpu location {engine_id}"
        return sum(
            self._get_group_blocks_needed(num_tokens, self.gpu_cache_groups[engine_id])
        )

    def get_num_blocks_needed(self, request: Request, engine_id: int) -> int:
        """Get physical pool blocks needed for a request on an engine."""
        return self.get_num_blocks_needed_for_request(request, engine_id)

    def get_num_blocks_needed_for_request(
        self,
        request: Request,
        engine_id: int,
        additional_tokens: int = 0,
    ) -> int:
        """Return cache pages needed for a request plus scheduler lookahead.

        Unchunked prefill needs a writable logical page for every prompt
        position. Decode admission only needs the bounded resident set for a
        sliding-window group, while full attention continues to grow normally.
        """
        if additional_tokens < 0:
            raise ValueError(
                f"additional_tokens must be non-negative, got {additional_tokens}"
            )
        num_tokens = (
            request.get_input_len()
            + request.get_output_len()
            + additional_tokens
        )
        if request.is_prefill_stage():
            # Aegaeon performs an unchunked prefill. All prompt positions must
            # have writable slots during that forward, even for SW layers;
            # skipped pages become releasable only after prefill commits.
            return sum(
                self._get_group_table_blocks_needed(
                    num_tokens, self.gpu_cache_groups[engine_id]
                )
            )
        if additional_tokens == 0:
            # Admission for the next step must use its actual writable pages.
            # A worst-case window alignment can exceed this step's demand and
            # would incorrectly reject a request that can execute alone.
            position = request.get_first_new_token_index()
            return sum(
                max(
                    0,
                    group.table_blocks_for_tokens(num_tokens)
                    - group.num_skipped_blocks(position),
                )
                for group in self.gpu_cache_groups[engine_id]
            )
        return self.get_num_blocks_needed_for_length(engine_id, num_tokens)

    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of blocks needed for a request already in GPU"""
        block_table = self.block_tables[request.request_id]
        assert is_gpu(
            block_table.location
        ), f"request {request.request_id} is not on GPU when calling get_num_append_blocks_needed"
        num_tokens = request.get_input_len() + request.get_output_len()
        needed_by_group = self._get_group_table_blocks_needed(
            num_tokens, block_table.cache_groups
        )
        return sum(
            needed - len(current)
            for needed, current in zip(needed_by_group, block_table.group_blocks)
        )

    # async def allocate_blocks(
    #     self,
    #     request: Request,
    #     engine_id: int,
    # ):
    #     """Allocate blocks for a request on GPU for the engine_id."""

    #     # Make sure the request is not already allocated or its blocks are on GPU for the given engine.
    #     assert (
    #         request.request_id not in self.block_tables
    #         or self.block_tables[request.request_id].location == engine_id
    #     ), (f"request {request.request_id} is allocated but is at {self.block_tables[request.request_id].location}; "
    #         f"please migrate it to engine #{engine_id} before allocating more blocks")

    #     num_blocks_needed = self.get_num_blocks_needed(request)
    #     block_shape = self.gpu_block_shape[engine_id]

    #     if request.request_id not in self.block_tables:
    #         # ..this request has not been allocated before
    #         self.block_tables[request.request_id] = BlockTable(
    #             engine_id,
    #             block_shape,
    #             await self._get_free_blocks(
    #                 num_blocks_needed, engine_id, block_shape))
    #     else:
    #         blocks = self.block_tables[request.request_id].blocks
    #         num_blocks_cur = len(blocks)
    #         if num_blocks_cur < num_blocks_needed:
    #             blocks.extend(await self._get_free_blocks(
    #                 num_blocks_needed - num_blocks_cur, engine_id, block_shape))

    async def allocate_blocks_batched(
        self, batched_requests: BatchedRequests, engine_id: int
    ):
        """Allocate blocks for a batch of requests."""
        if len(batched_requests) == 0:
            return

        num_blocks_list = []
        group_append_counts: List[List[int]] = []
        location = engine_id
        block_shape = self.gpu_block_shape[engine_id]
        cache_groups = self.gpu_cache_groups[engine_id]

        for request in batched_requests.requests:
            # Make sure the request is not already allocated or its blocks are on GPU for the given engine.
            assert (
                request.request_id not in self.block_tables
                or self.block_tables[request.request_id].location == engine_id
            ), (
                f"request {request.request_id} is allocated but is at {self.block_tables[request.request_id].location}; "
                f"please migrate it to engine #{engine_id} before allocating more blocks"
            )

            num_tokens = request.get_input_len() + request.get_output_len()
            if request.request_id not in self.block_tables:
                # ..this request has not been allocated before
                needed_by_group = self._get_group_table_blocks_needed(
                    num_tokens, cache_groups
                )
                append_counts = needed_by_group
            else:
                table = self.block_tables[request.request_id]
                if table.cache_groups != cache_groups:
                    raise ValueError(
                        f"request {request.request_id} cache groups do not match "
                        f"engine #{engine_id}"
                    )
                # The previous step has completed before this request can be
                # scheduled again. Retire whole SW pages before allocating the
                # next logical column so their physical ids can be reused by
                # this same batch.
                released = table.release_skipped_blocks(
                    request.get_first_new_token_index()
                )
                if released:
                    self.gpu_free_blocks[engine_id].extend(released)
                needed_by_group = self._get_group_table_blocks_needed(
                    num_tokens, cache_groups
                )
                append_counts = [
                    needed - len(current)
                    for needed, current in zip(needed_by_group, table.group_blocks)
                ]
            if any(count < 0 for count in append_counts):
                raise ValueError(
                    f"request {request.request_id} cache allocation shrank: "
                    f"append_counts={append_counts}"
                )
            group_append_counts.append(append_counts)
            num_blocks_list.append(sum(append_counts))

        # Allocate in batch
        free_blocks_list = await self._get_free_blocks_batched(
            num_blocks_list, location, block_shape
        )

        for request, free_blocks, append_counts in zip(
            batched_requests.requests, free_blocks_list, group_append_counts
        ):
            cursor = 0
            new_group_blocks = []
            for count in append_counts:
                new_group_blocks.append(free_blocks[cursor : cursor + count])
                cursor += count
            if request.request_id not in self.block_tables:
                self.block_tables[request.request_id] = BlockTable(
                    engine_id,
                    block_shape,
                    cache_groups,
                    new_group_blocks,
                )
            else:
                table = self.block_tables[request.request_id]
                for current, new_blocks in zip(table.group_blocks, new_group_blocks):
                    current.extend(new_blocks)

    def free_blocks(self, request_id: int):
        """Free blocks for a request."""
        assert request_id in self.block_tables, f"request {request_id} not allocated"
        block_table = self.block_tables.pop(request_id)
        block_size_bytes = self._get_block_size_bytes(block_table.block_shape)

        if is_gpu(block_table.location):
            self.gpu_free_blocks[block_table.location].extend(block_table.blocks)
        else:
            self.cpu_free_blocks[block_table.block_shape].extend(block_table.blocks)
            # Update slab usage
            for block in block_table.blocks:
                slab = block * block_size_bytes // self.cpu_slab_size_bytes
                self.cpu_slab_usage[slab] -= 1
        self.latest_events.pop(request_id, None)
        self.zombie_request_ids.append(request_id)

    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for request in requests:
            self.free_blocks(request.request_id)

    def get_block_table(self, request_id: int) -> BlockTable:
        """Get the block table for a request"""
        assert request_id in self.block_tables, f"request {request_id} not allocated"
        return self.block_tables[request_id]

    def are_all_requests_at_location(
        self, requests: List[Request], location: BlockLocation
    ) -> bool:
        """Return whether every request has an allocated table at ``location``."""
        return bool(requests) and all(
            (block_table := self.block_tables.get(request.request_id)) is not None
            and block_table.location == location
            for request in requests
        )

    def is_all_requests_on_gpu(self, requests: BatchedRequests):
        """Check if all requests in a batch are on GPU"""
        return all(
            is_gpu(self.block_tables[request.request_id].location)
            for request in requests.requests
        )

    def log_block_usage(self):
        log_lines = []
        log_lines.append("---------- Block Usage ----------")
        for engine_id, gpu_free_list in self.gpu_free_blocks.items():
            num_max_blocks = self.get_gpu_num_max_blocks(engine_id)
            num_free_blocks = len(gpu_free_list)
            num_move_blocks = self.move_gpu_num_blocks[engine_id]
            log_lines.append(
                f"[gpu (engine {engine_id})] {num_free_blocks} / {num_max_blocks} "
                f"({num_free_blocks / num_max_blocks * 100:.2f}%) "
                f"free; "
                f"{num_move_blocks} / {num_max_blocks} "
                f"({num_move_blocks / num_max_blocks * 100:.2f}%) "
                f"moving; "
            )

        for block_shape, cpu_free_list in self.cpu_free_blocks.items():
            slabs: List[int] = []
            for i, shape in enumerate(self.cpu_block_shape):
                if shape == block_shape:
                    slabs.append(i)

            num_free_blocks = len(cpu_free_list)
            num_used_blocks = sum(self.cpu_slab_usage[i] for i in slabs)
            num_move_blocks = self.move_cpu_num_blocks[block_shape]
            num_max_blocks = num_used_blocks + num_free_blocks

            if num_max_blocks == 0:
                continue
            log_lines.append(
                f"[cpu (shape {block_shape}, {len(slabs)} slabs)] {num_free_blocks} / {num_max_blocks} "
                f"({num_free_blocks / num_max_blocks * 100:.2f}%) "
                f"free; "
                f"{num_move_blocks} / {num_max_blocks} "
                f"({num_move_blocks / num_max_blocks * 100:.2f}%) "
                f"moving; "
            )
        log_lines.append("---------------------------------")
        logger.info("\n".join(log_lines))

    def clear_zombie_requests(self):
        from aegaeon.llm import Controller

        Controller.remote_call_all_workers_async(
            "clear_request_resource_batched", self.zombie_request_ids
        )
        self.zombie_request_ids.clear()

    async def move_requests(
        self,
        requests: List[Request],
        target_location: BlockLocation,
    ):
        """
        Move blocks for a batch of requests to `target_location`.

        Depending on the current location of each request, this method may:
        - Do nothing, if the request is already at the target location
        - Issue swap_blocks, if the move is between CPU and GPU
        - Issue migrate_blocks, if the move is between two GPUs
        """
        swapped: List[Request] = []
        swap_cur_location: Optional[BlockLocation] = None
        swap_block_shape: Optional[Tuple[int, ...]] = None
        # Copy lists contain only pages still visible to attention. The release
        # list additionally contains newly obsolete source pages; all source
        # ids remain unavailable until the chained CUDA event completes.
        swap_source_block_ids = []
        swap_target_block_ids = []
        swap_release_source_block_ids = []
        swap_group_ids = []
        swap_first_kernel_blocks = []
        swap_valid_kernel_blocks = []
        swap_plans = []
        seen_request_ids = set()
        for request in requests:
            if request.request_id in seen_request_ids:
                continue
            seen_request_ids.add(request.request_id)
            assert (
                block_table := self.block_tables.get(request.request_id, None)
            ), f"request {request.request_id} not allocated"
            cur_location = block_table.location
            if cur_location == target_location:
                continue
            elif is_gpu(cur_location) and is_gpu(target_location):
                raise NotImplementedError("Migration is not implemented")
            else:
                if swap_cur_location is not None:
                    assert (
                        block_table.location == swap_cur_location
                    ), f"request {request.request_id} is on {block_table.location}; expected {swap_cur_location}"
                if swap_block_shape is not None:
                    assert (
                        block_table.block_shape == swap_block_shape
                    ), f"request {request.request_id}'s shape is not {swap_block_shape}"
                swap_cur_location = block_table.location
                swap_block_shape = block_table.block_shape
                swapped.append(request)
                # The GPU side of a swap is the user; i.e., its worker will later access the blocks
                swap_engine_id = (
                    target_location if is_gpu(target_location) else swap_cur_location
                )
                if is_gpu(target_location):
                    target_groups = self.gpu_cache_groups[target_location]
                    if block_table.cache_groups != target_groups:
                        raise ValueError(
                            f"request {request.request_id} cache groups do not match "
                            f"target engine #{target_location}"
                        )

                processed_computed_tokens = request.get_first_new_token_index()
                transfer_entries = block_table.transfer_entries(
                    processed_computed_tokens
                )
                swap_plans.append(
                    (block_table, processed_computed_tokens, transfer_entries)
                )
                swap_release_source_block_ids += block_table.blocks

        if not swapped:
            return False

        # Reserve the entire destination before changing any source table.
        # Per-request reservation could commit an earlier request and then
        # fail a later one, losing the earlier source ids before a MoveEvent
        # exists to reclaim them.
        target_block_ids = await self._get_free_blocks(
            sum(len(entries) for _, _, entries in swap_plans),
            target_location,
            swap_block_shape,
        )
        cursor = 0
        destination_tables = []
        for block_table, processed_computed_tokens, transfer_entries in swap_plans:
            num_blocks = len(transfer_entries)
            new_block_ids = target_block_ids[cursor : cursor + num_blocks]
            cursor += num_blocks
            # Some retained pages need a destination allocation without
            # carrying any cached bytes. A W=1 sliding-window page is the
            # canonical case: all prior KV is invisible, but the next token
            # still writes into the current logical page. Keep it in the
            # destination table while excluding it from the DMA lists.
            copy_entries = [
                (entry, target_block_id)
                for entry, target_block_id in zip(
                    transfer_entries, new_block_ids
                )
                if entry[4] > 0
            ]
            swap_source_block_ids += [entry[2] for entry, _ in copy_entries]
            swap_target_block_ids += [target for _, target in copy_entries]
            swap_group_ids += [entry[0] for entry, _ in copy_entries]
            swap_first_kernel_blocks += [
                entry[3] for entry, _ in copy_entries
            ]
            swap_valid_kernel_blocks += [
                entry[4] for entry, _ in copy_entries
            ]

            # Preserve absolute logical columns, replacing pages before a
            # sliding window with null entries. Only retained pages receive
            # destination allocations.
            destination_tables.append(
                block_table.moved_group_blocks(
                    processed_computed_tokens, new_block_ids
                )
            )

        for (block_table, _, _), group_blocks in zip(swap_plans, destination_tables):
            block_table.group_blocks = group_blocks
            block_table.location = target_location

        # Swapping
        if swapped:
            is_swap_in = True if is_gpu(target_location) else False
            source_event_handles = [
                self.latest_events.get(request.request_id, None) for request in swapped
            ]
            event_refs = self.remote_callables[swap_engine_id](
                "swap_blocks",
                [req.request_id for req in swapped],
                swap_source_block_ids,
                swap_target_block_ids,
                swap_group_ids,
                swap_first_kernel_blocks,
                swap_valid_kernel_blocks,
                source_event_handles,
                is_swap_in,
            )
            # ray.get(event_refs)

            # Update the move states
            move_event = MoveEvent(
                next(self.event_counter),
                swapped,
                swap_block_shape,
                swap_cur_location,
                target_location,
                swap_release_source_block_ids,
                event_refs,
            )
            if is_gpu(swap_cur_location):
                self.move_gpu_num_blocks[swap_cur_location] += len(
                    swap_release_source_block_ids
                )
                self.move_events_gpu[swap_cur_location].append(move_event)
            else:
                self.move_cpu_num_blocks[swap_block_shape] += len(
                    swap_release_source_block_ids
                )
                self.move_events_cpu[swap_block_shape].append(move_event)
            for request in swapped:
                self.latest_events[request.request_id] = event_refs

        # Migrated
        pass

        return bool(swapped)

    def _migrate_requests(
        self,
        requests: List[Request],
        target_location: BlockLocation,
    ):
        raise NotImplementedError()
        """Migrate blocks for a batch of requests to `target_location`.
        It is assumed that `requests` reside at the same source GPU location.
        """
        if len(requests) == 0:
            return

        _block_table = self.block_tables[requests[0].request_id]
        block_shape = _block_table.block_shape
        cur_location = _block_table.location
        assert is_gpu(target_location) and is_gpu(
            cur_location
        ), f"both target location {target_location} and current location {cur_location} should be on GPU"
        # The target engine is the user; i.e., its worker will later access the blocks
        engine_id = target_location

        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        for request in requests:
            assert (
                block_table := self.block_tables.get(request.request_id, None)
            ), f"request {request.request_id} not allocated"
            assert (
                block_table.location == cur_location
            ), f"request {request.request_id} is on {target_location}; expected {cur_location}"
            assert (
                block_table.block_shape == block_shape
            ), f"request {request.request_id}'s shape is not {block_shape}"

            old_block_ids = block_table.blocks
            num_blocks = len(old_block_ids)
            new_block_ids = self._get_free_blocks(
                num_blocks, target_location, block_shape, engine_id
            )
            source_block_ids += old_block_ids
            target_block_ids += new_block_ids

            # Update the block table
            block_table.replace_flat_blocks(new_block_ids)
            block_table.location = target_location

        # Gather CUDA events
        source_events = [
            self.latest_events.get(request.request_id, None) for request in requests
        ]
        event_refs = self.remote_callables[engine_id](
            "migrate_blocks",
            requests,
            cur_location,
            source_block_ids,
            target_block_ids,
            source_events,
        )
        # ..this adds the migration event to the source engine as a move-out event
        self.remote_callables[cur_location](
            "migrate_blocks_source", requests, event_ref
        )

        # Update the move states
        move_event = MoveEvent(
            block_shape,
            cur_location,
            target_location,
            old_block_ids,
            event_ref,
        )
        self.move_gpu_num_blocks[cur_location] += num_blocks
        self.move_events_gpu[cur_location].add(move_event)
        for request in requests:
            self.latest_events[request.request_id] = event_ref
