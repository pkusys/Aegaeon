# 模型管理

Aegaeon 将模型副本保存在各节点的 CPU 权重缓存中，并在就绪副本之间路由请求。每个请求的 Prefill 和 Decode 生命周期始终位于同一个 Aegaeon 节点。

## 副本状态

| 状态 | 含义 |
|---|---|
| `absent` | 无副本；放置快照中省略此状态 |
| `loading` | 正在部署 |
| `ready` | 接收新请求 |
| `draining` | 拒绝新请求；已有请求结束后重试卸载 |
| `failed` | 部署失败 |

## 部署与路由

```bash
aegaeon deploy Qwen/Qwen3-4B --replicas 2
aegaeon undeploy Qwen/Qwen3-4B --nodes node_1
```

`--replicas`（HTTP 请求中的 `replica_count`）设置就绪副本数的下限目标。减少副本数使用 undeploy。`--nodes`（`node_ids`）直接选择节点，优先于副本数目标。

`server.model_placement_policy: least-loaded` 根据缓存容量、驱逐代价和请求负载选择节点；`server.request_routing_policy: least-outstanding` 在就绪节点中选择未完成请求最少的节点。两者均支持 `round-robin`。

## 操作结果

部署和卸载响应包含 `status`、各节点结果及 `failed`。HTTP 200 仍可能包含 `status: partial`，需检查这些字段确认操作是否完成。部分成功时，CLI 退出码为 1。部署后没有任何就绪副本时，返回 HTTP 507。

通过 `GET /v1/aegaeon/runtime` 查看放置和路由状态。命令选项见 [CLI 参考](cli.md)，HTTP 字段见 [API 参考](api.md)。
