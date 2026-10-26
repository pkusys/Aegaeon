# 故障排查

先运行 `aegaeon doctor --check-cuda`，并检查服务日志及 `ray status`。

| 现象 | 检查项 |
|---|---|
| 依赖或扩展导入失败 | 激活安装环境，更换 Torch/CUDA 后重新编译扩展 |
| Ray actor 等待资源 | 核对节点资源、可见 GPU 和 `worker_num_cpus` 与拓扑是否匹配 |
| CPU 缓存分配失败 | 检查共享内存，减小缓存或 slab 预算 |
| GPU OOM | 减小 batch 或上下文长度，为权重、预取和工作区留出空间 |
| 缺少性能数据 | 核对模型与设备别名，采集[性能数据](profiling.md) |
| HTTP 400 / 422 | 检查模型名称、[生成参数](generation.md)及 `/docs` 中的 schema |
| HTTP 503 | 在运行时快照中检查 READY 副本 |
| HTTP 507 | 检查节点缓存容量和部署结果 |

## 主机内存注册与 memlock

出现锁页内存或主机内存注册错误时：

```bash
df -h /dev/shm
ulimit -Sl
ulimit -Hl
```

调整服务或容器的共享内存容量及 memlock 限制，然后重启服务。

## Foundry

- 启动前将 `AEGAEON_FOUNDRY_HOOK_PATH` 设置为已安装的 `libcuda_hook.so` 路径。
- 将 Conda 的 `lib` 目录加入 `LD_LIBRARY_PATH`，以提供 Boost 和 `libstdc++`。
- 缺少 CUDA driver 符号时，使用 NVIDIA [前向兼容软件包](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)中的匹配动态库，设置 `AEGAEON_CUDA_COMPAT_PATH`，并将其加到 `LD_LIBRARY_PATH` 首位。
- `seq_len_buckets` 使用一个条目，SAVE/LOAD 配置保持一致。运行时或布局更改后重新生成存档。

反馈问题时附上执行命令、配置、依赖版本和 traceback。
