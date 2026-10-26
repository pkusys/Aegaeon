"""Pure helpers for planning partial KV-cache transfers."""


def logical_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Return block-table columns needed to address absolute token positions."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return (num_tokens + block_size - 1) // block_size


def sliding_window_resident_block_bound(
    num_tokens: int, block_size: int, sliding_window: int
) -> int:
    """Return the safe per-request resident-page bound for admission control.

    The next query can make a W-token window straddle one more physical page
    than ``ceil(W / block_size)``. The bound is the current page plus enough
    pages for the W-1 committed tokens that precede it.
    """
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    logical_blocks = logical_blocks_for_tokens(num_tokens, block_size)
    if logical_blocks == 0:
        return 0
    max_resident_blocks = (sliding_window - 1 + block_size - 1) // block_size + 1
    return min(logical_blocks, max_resident_blocks)


def num_skipped_sliding_window_blocks(
    processed_computed_tokens: int, block_size: int, sliding_window: int
) -> int:
    """Return leading complete pages invisible to the next SW query."""
    if processed_computed_tokens < 0:
        raise ValueError(
            "processed_computed_tokens must be non-negative, got "
            f"{processed_computed_tokens}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    skipped_tokens = max(0, processed_computed_tokens - sliding_window + 1)
    return skipped_tokens // block_size


def kernel_block_extent_for_attention_page(
    cached_tokens: int,
    physical_block_index: int,
    physical_block_size: int,
    kernel_block_size: int,
    sliding_window: int | None = None,
) -> tuple[int, int]:
    """Return the live small-page extent within one physical attention page.

    Full attention produces a prefix on the final page. Sliding-window
    attention can additionally produce a suffix on the oldest live page; its
    intersection with any physical page is still one contiguous DMA extent.
    """
    if cached_tokens < 0:
        raise ValueError(f"cached_tokens must be non-negative, got {cached_tokens}")
    if physical_block_index < 0:
        raise ValueError(
            "physical_block_index must be non-negative, got " f"{physical_block_index}"
        )
    if physical_block_size <= 0 or kernel_block_size <= 0:
        raise ValueError("physical and kernel block sizes must be positive")
    if physical_block_size % kernel_block_size:
        raise ValueError(
            f"Kernel block size {kernel_block_size} must divide physical "
            f"block size {physical_block_size}"
        )
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    block_start = physical_block_index * physical_block_size
    block_end = block_start + physical_block_size
    # ``cached_tokens`` is the committed KV-token count and the absolute
    # position of the next query. That query supplies its own K/V, so an
    # inclusive window of W needs only the preceding W-1 cached tokens.
    live_start = (
        0 if sliding_window is None else max(0, cached_tokens - sliding_window + 1)
    )
    extent_start = max(block_start, live_start)
    extent_end = min(block_end, cached_tokens)
    if extent_start >= extent_end:
        return (0, 0)

    relative_start = extent_start - block_start
    relative_end = extent_end - block_start
    first_kernel_block = relative_start // kernel_block_size
    end_kernel_block = (relative_end + kernel_block_size - 1) // kernel_block_size
    return first_kernel_block, end_kernel_block - first_kernel_block


def valid_kernel_blocks_for_attention_page(
    cached_tokens: int,
    physical_block_index: int,
    physical_block_size: int,
    kernel_block_size: int,
) -> int:
    """Count valid small pages in a full-attention physical-page prefix."""
    _, count = kernel_block_extent_for_attention_page(
        cached_tokens,
        physical_block_index,
        physical_block_size,
        kernel_block_size,
    )
    return count


def prefix_bytes_for_kernel_blocks(
    page_size_bytes: int,
    kernel_blocks_per_page: int,
    valid_kernel_blocks: int,
) -> int:
    """Return bytes in a contiguous valid prefix of a backing page."""
    if page_size_bytes <= 0:
        raise ValueError(f"page_size_bytes must be positive, got {page_size_bytes}")
    if kernel_blocks_per_page <= 0:
        raise ValueError(
            "kernel_blocks_per_page must be positive, got " f"{kernel_blocks_per_page}"
        )
    if not 0 <= valid_kernel_blocks <= kernel_blocks_per_page:
        raise ValueError(
            f"valid kernel blocks {valid_kernel_blocks} outside [0, "
            f"{kernel_blocks_per_page}]"
        )
    if page_size_bytes % kernel_blocks_per_page:
        raise ValueError(
            f"Backing page size {page_size_bytes} is not divisible by "
            f"{kernel_blocks_per_page} kernel blocks"
        )
    return page_size_bytes // kernel_blocks_per_page * valid_kernel_blocks


def byte_extent_for_kernel_blocks(
    page_size_bytes: int,
    kernel_blocks_per_page: int,
    first_kernel_block: int,
    num_kernel_blocks: int,
) -> tuple[int, int]:
    """Translate a small-page extent to a byte offset and byte length."""
    if first_kernel_block < 0:
        raise ValueError(
            f"first_kernel_block must be non-negative, got {first_kernel_block}"
        )
    end_kernel_block = first_kernel_block + num_kernel_blocks
    if end_kernel_block > kernel_blocks_per_page:
        raise ValueError(
            f"Kernel-block extent [{first_kernel_block}, {end_kernel_block}) "
            f"exceeds page size {kernel_blocks_per_page}"
        )
    bytes_before = prefix_bytes_for_kernel_blocks(
        page_size_bytes,
        kernel_blocks_per_page,
        first_kernel_block,
    )
    extent_bytes = prefix_bytes_for_kernel_blocks(
        page_size_bytes,
        kernel_blocks_per_page,
        num_kernel_blocks,
    )
    return bytes_before, extent_bytes
