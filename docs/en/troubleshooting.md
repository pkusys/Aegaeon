# Troubleshooting

Start with `aegaeon doctor --check-cuda`, the service log, and `ray status`.

| Symptom | Check |
|---|---|
| Dependency or extension import failure | Activate the installation environment and rebuild extensions after changing Torch/CUDA |
| Ray actors wait for resources | Match node resources, visible GPUs, and `worker_num_cpus` to the topology |
| CPU cache allocation fails | Check shared memory and reduce cache/slab budgets |
| GPU OOM | Reduce batch/context sizes and leave room for weights, prefetch, and workspaces |
| Missing profiles | Match model/device aliases and collect [profiles](profiling.md) |
| HTTP 400 / 422 | Check model names, [generation values](generation.md), and `/docs` schemas |
| HTTP 503 | Check READY replicas in the runtime snapshot |
| HTTP 507 | Check node cache capacity and deployment results |

## Host registration and memlock

For pinned-memory or host-registration errors:

```bash
df -h /dev/shm
ulimit -Sl
ulimit -Hl
```

Adjust service/container shared-memory capacity and memlock limits, then restart the service.

## Foundry

- Set `AEGAEON_FOUNDRY_HOOK_PATH` to the installed `libcuda_hook.so` before startup.
- Add Conda's `lib` directory to `LD_LIBRARY_PATH` for Boost and `libstdc++`.
- For missing CUDA driver symbols, use a matching library from NVIDIA's [forward-compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html), set `AEGAEON_CUDA_COMPAT_PATH`, and prepend it to `LD_LIBRARY_PATH`.
- Use one `seq_len_buckets` entry and matching SAVE/LOAD configurations. Recreate archives after runtime or layout changes.

When reporting a problem, include the command, configuration, dependency versions, and traceback.
