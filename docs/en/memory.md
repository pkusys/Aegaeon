# Memory and KV Cache

Aegaeon caches model weights separately from request KV state, so it can switch models while preserving in-progress requests.

| Region | Contents | Main setting |
|---|---|---|
| CPU weight cache | Model weights shared by node Workers | `model_cache_size_gb` |
| CPU KV cache | Request state between execution steps | `cpu_num_slabs` × `cpu_slab_size_bytes` |
| GPU KV budget | Capacity estimate after subtracting active and prefetched weights | `memory_utilization` |
| QuickLoader GPU arena | Space for weights and KV allocations, allocated when QuickLoader first loads a model | `AEGAEON_ARENA_FRACTION` |

These budgets still need headroom for Ray, tokenizers, other mapped files, and temporary tensors. Settings expressed in GB/GiB use `1024^3` bytes.

## Weight cache

A positive `model_cache_size_gb` sets a fixed CPU weight-cache budget. With zero, Aegaeon estimates the budget from startup models, or uses 20 GiB if none are loaded at startup. Deployment can evict models with no active requests to make room for another model.

## KV cache

BlockManager allocates request KV blocks and coordinates CPU/GPU transfers. In P/D mode, Prefill and Decode exchange request state through the node's shared CPU KV cache. [Decode Work Stealing](work-stealing.md) uses that same cache when transferring batch ownership between Decode engines.

## Shared memory

CPU caches use shared, pinned host memory, so their budgets must fit both available host memory and the backing filesystem (`/dev/shm` by default), with sufficient service or container memlock limits. Independent services need separate shared-memory directories; `AEGAEON_SHM_DIR` selects the directory at startup.

Choose budgets in the [configuration reference](configuration.md). For allocation errors, see [Troubleshooting](troubleshooting.md).
