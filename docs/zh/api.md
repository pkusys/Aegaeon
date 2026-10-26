# HTTP API

Aegaeon 为已部署模型提供 OpenAI 风格的 API，默认监听 8000 端口。运行中服务的 `/docs`、`/redoc` 和 `/openapi.json` 提供请求结构和默认值。鉴权由部署侧网关配置。

## 接口清单

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | /health | 固定健康响应；不深入检查 actor 状态 |
| GET | /v1/models | 已部署模型列表 |
| GET | /v1/models/{model_name:path} | 已部署模型元信息；路径转换器接受名称中的 / |
| POST | /v1/models/deploy | 部署 CPU 权重缓存副本，见[模型管理](model-management.md) |
| POST | /v1/models/undeploy | 卸载指定或全部副本，见[模型管理](model-management.md) |
| POST | /v1/chat/completions | Chat，非流式或 SSE |
| POST | /v1/completions | Legacy Completion，非流式或 SSE |
| POST | /tokenize | 将纯文本 prompt 编码为 token ID |
| POST | /detokenize | 将 token ID 解码为文本 |
| GET | /v1/aegaeon/runtime | 集群运行状态快照 |
| GET | /v1/aegaeon/events | 运行状态 SSE |
| GET | /v1/aegaeon/metrics | 60 秒窗口访问统计与限流设置 |

## 错误

| 状态码 | 含义 |
|---|---|
| 400 | 模型或参数无效，或请求所需的 KV 缓存容量不足 |
| 422 | 请求结构无效，包括不支持的生成字段 |
| 429 | 访问限流 |
| 503 | 没有 READY 副本 |
| 507 | 部署失败，且没有剩余的 READY 副本 |
| 500 | 服务端错误，详见服务日志 |

流式输出开始后，KV 缓存容量不足的错误通过含有 `error` 对象的 SSE 数据返回，随后发送 `data: [DONE]`。

运行状态接口见[日志与监控](observability.md)。
