# 日志与监控

## 日志

Aegaeon 将 INFO 日志写入标准输出和 `AEGAEON_LOG_FILE` 指定的文件，默认文件为 `output.log`。也可在启动前设置绝对路径：

```bash
mkdir -p logs
export AEGAEON_LOG_FILE="$PWD/logs/server.log"
```

该路径会传给 Ray actor，因此每个节点上都必须存在其父目录。Ray actor 的标准输出和错误输出也可在对应 Ray session 的日志目录中查看。

## 运行时接口

| 接口 | 内容 |
|---|---|
| `/health` | API 存活状态 |
| `/v1/models` | 服务的已部署模型列表 |
| `/v1/aegaeon/runtime` | 节点、Engine、副本、路由、缓存及 Decode 负载 |
| `/v1/aegaeon/events` | 通过 SSE 提供 Engine 状态快照和模型切换事件 |
| `/v1/aegaeon/metrics` | 各模型的请求速率、生成 token 速率、计数及限流设置 |

```bash
curl -sS http://127.0.0.1:8000/v1/aegaeon/runtime
curl -N http://127.0.0.1:8000/v1/aegaeon/events
```

通过运行时快照中的副本状态 `ready` 确认模型可用性。请求速率和生成 token 速率是各模型在 60 秒滑动窗口内的平均值。启动 API 前设置 `AEGAEON_MAX_REQUESTS_PER_SECOND` 和 `AEGAEON_MAX_TOKENS_PER_SECOND` 可限制对应速率；超限返回 HTTP 429。
