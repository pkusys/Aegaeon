# 部署

每个已配置的 Aegaeon 节点运行一个 Controller，节点内的 Engine 共享 CPU 缓存。当前支持 TP=1，因此每个 Engine 使用一张 GPU。常规部署中，每个物理 Ray 节点对应一个 Aegaeon 节点；`server.nnodes` 统计节点数，而非 GPU 数。

多 GPU 主机通常设置 `nnodes: 1`。通过 Python 集成时，可分别设置 `NodeConfig.cuda_visible_devices` 和 `NodeConfig.shm_dir`，在同一主机上隔离多个逻辑节点。

## 执行模式

| 模式 | Server 配置 |
|---|---|
| Simple | `num_engines > 0`，P/D 数量均为 0 |
| P/D | `num_engines: 0`，至少一个 Prefill 和一个 Decode Engine |

单 GPU、P/D 和多节点布局见[配置示例](examples/README.md)。在各节点准备相同的模型文件和匹配的性能数据。

## Ray 集群

单机可使用 `--ray-address local`。两台主机分别启动 Ray：

```bash
# Head node
ray start --head --port=6789 --resources='{"node_0":1}'
```

```bash
# Worker node
ray start --address=HEAD_IP:6789 --resources='{"node_1":1}'
```

在配置中设置 `server.nnodes: 2`，然后从 Head 节点启动 API：

```bash
aegaeon start --config /srv/aegaeon/config.yaml --ray-address HEAD_IP:6789
```

## 资源

每个节点需要为每个配置的 Engine 提供一张可见 GPU。Ray CPU 预算需覆盖每个 Worker 的 `server.worker_num_cpus` 个逻辑 CPU，以及 Controller 所需的容量。`ray status` 显示集群的 CPU 和 GPU 资源。

如需覆盖默认 [SLO 设置](scheduling.md#slo-设置)，在启动前为 API 和所有 Ray 节点设置相应环境变量。独立服务使用各自的 Ray 实例和共享内存目录。停止前台服务会断开其 Ray driver；已有 Ray 集群需单独管理。
