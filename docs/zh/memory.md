# 内存与 KV Cache

Aegaeon 分别缓存模型权重和请求的 KV 状态，因此可以在保留未完成请求状态的同时切换模型。

| 区域 | 内容 | 主要配置 |
|---|---|---|
| CPU 权重缓存 | 节点内 Worker 共享的模型权重 | `model_cache_size_gb` |
| CPU KV 缓存 | 执行步骤之间保存的请求状态 | `cpu_num_slabs` × `cpu_slab_size_bytes` |
| GPU KV 预算 | 容量估算中扣除当前和预取权重后的空间 | `memory_utilization` |
| QuickLoader GPU arena | 权重与 KV 的分配空间，在 QuickLoader 首次加载模型时分配 | `AEGAEON_ARENA_FRACTION` |

设置这些预算时，还需为 Ray、tokenizer、其他映射文件及临时张量留出余量。GB/GiB 配置以 `1024^3` 字节计。

## 权重缓存

`model_cache_size_gb` 大于 0 时指定固定的 CPU 权重缓存预算；为 0 时根据启动模型估算，未加载启动模型时使用 20 GiB。部署新模型时，可以驱逐没有活跃请求的模型以腾出空间。

## KV 缓存

BlockManager 分配请求的 KV block，并协调 CPU/GPU 搬运。P/D 模式下，Prefill 与 Decode 通过节点共享的 CPU KV 缓存交换请求状态。[Decode Work Stealing](work-stealing.md) 也通过这份缓存在 Decode 引擎之间转移 batch 的归属。

## 共享内存

CPU 缓存使用共享的页锁定主机内存，预算必须同时满足可用主机内存、承载缓存文件的文件系统容量（默认 `/dev/shm`）以及服务或容器的 memlock 限制。独立服务需要使用不同的共享内存目录，启动时通过 `AEGAEON_SHM_DIR` 选择。

预算字段见[配置参考](configuration.md)，分配错误见[故障排查](troubleshooting.md)。
