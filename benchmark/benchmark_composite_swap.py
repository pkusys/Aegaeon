"""Benchmark Aegaeon's real segmented composite-cache copy pattern.

Example on the user's NUMA-local A100 setup::

    CUDA_VISIBLE_DEVICES=0 taskset -c 0-9,20-29 \
      python benchmark/benchmark_composite_swap.py --device cuda:0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegaeon.cache_transfer import (
    byte_extent_for_kernel_blocks,
    kernel_block_extent_for_attention_page,
)

MIB = 1024 * 1024


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def make_extent_plan(
    length: int,
    physical_block_size: int,
    kernel_block_size: int,
    segment_page_bytes: int,
    fixed_pages: int,
    sliding_window: int | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    attention_pages = math.ceil(length / physical_block_size)
    total_pages = fixed_pages + attention_pages
    full = [(0, segment_page_bytes)] * total_pages
    optimized = [(0, segment_page_bytes)] * fixed_pages
    kernel_pages = physical_block_size // kernel_block_size
    for block_index in range(attention_pages):
        first, count = kernel_block_extent_for_attention_page(
            length,
            block_index,
            physical_block_size,
            kernel_block_size,
            sliding_window,
        )
        optimized.append(
            byte_extent_for_kernel_blocks(
                segment_page_bytes,
                kernel_pages,
                first,
                count,
            )
        )
    return full, optimized


def submit_copies(
    cpu_pages: torch.Tensor,
    gpu_segments: list[torch.Tensor],
    extents: list[tuple[int, int]],
    is_swap_in: bool,
) -> None:
    segment_page_bytes = gpu_segments[0].shape[1]
    for segment_index, gpu_pages in enumerate(gpu_segments):
        for page_id, (byte_offset, byte_length) in enumerate(extents):
            if byte_length == 0:
                continue
            cpu_extent = cpu_pages[page_id].narrow(
                0,
                segment_index * segment_page_bytes + byte_offset,
                byte_length,
            )
            gpu_extent = gpu_pages[page_id].narrow(0, byte_offset, byte_length)
            if is_swap_in:
                gpu_extent.copy_(cpu_extent, non_blocking=True)
            else:
                cpu_extent.copy_(gpu_extent, non_blocking=True)


def measure(
    cpu_pages: torch.Tensor,
    gpu_segments: list[torch.Tensor],
    extents: list[tuple[int, int]],
    is_swap_in: bool,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    stream = torch.cuda.Stream(device=gpu_segments[0].device)
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            submit_copies(cpu_pages, gpu_segments, extents, is_swap_in)
    stream.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            submit_copies(cpu_pages, gpu_segments, extents, is_swap_in)
            end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), percentile(samples, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[1, 128, 256, 512, 528, 529, 1057],
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--fixed-pages", type=int, default=3)
    parser.add_argument("--physical-page-bytes", type=int, default=17_301_504)
    parser.add_argument("--physical-block-size", type=int, default=528)
    parser.add_argument("--kernel-block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int)
    args = parser.parse_args()

    if args.physical_page_bytes % args.segments:
        raise ValueError("physical page bytes must divide evenly across segments")
    segment_page_bytes = args.physical_page_bytes // args.segments
    device = torch.device(args.device)
    print(torch.cuda.get_device_name(device))

    for length in args.lengths:
        full, optimized = make_extent_plan(
            length,
            args.physical_block_size,
            args.kernel_block_size,
            segment_page_bytes,
            args.fixed_pages,
            args.sliding_window,
        )
        cpu_pages = torch.empty(
            (len(full), args.physical_page_bytes),
            dtype=torch.uint8,
            pin_memory=True,
        )
        gpu_segments = [
            torch.empty(
                (len(full), segment_page_bytes), dtype=torch.uint8, device=device
            )
            for _ in range(args.segments)
        ]
        old_bytes = sum(length for _, length in full) * args.segments
        new_bytes = sum(length for _, length in optimized) * args.segments
        for direction, is_swap_in in (("H2D", True), ("D2H", False)):
            old_p50, old_p95 = measure(
                cpu_pages,
                gpu_segments,
                full,
                is_swap_in,
                args.warmup,
                args.iters,
            )
            new_p50, new_p95 = measure(
                cpu_pages,
                gpu_segments,
                optimized,
                is_swap_in,
                args.warmup,
                args.iters,
            )
            speedup = (old_p50 - new_p50) / old_p50 * 100
            print(
                f"L={length:5d} {direction} bytes={old_bytes / MIB:6.1f}->"
                f"{new_bytes / MIB:6.1f}MiB p50={old_p50:7.4f}->"
                f"{new_p50:7.4f}ms ({speedup:6.2f}%) "
                f"p95={old_p95:7.4f}->{new_p95:7.4f}ms"
            )


if __name__ == "__main__":
    main()
