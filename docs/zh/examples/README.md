# 配置示例

这些示例涵盖常见的服务拓扑。模型路径、GPU 分配、profile 别名和内存预算取决于部署环境。

| 文件 | 拓扑 |
|---|---|
| simple.yaml | 单节点、1 Simple Engine、TP=1 |
| pd.yaml | 单节点、1P+1D、TP=1 |
| multi-node.yaml | 两个物理节点，每节点 1 Simple Engine |
| dynamic.yaml | 空服务启动后显式部署 |
| work-stealing.yaml | 单节点、1P+2D |
| foundry-save.yaml / foundry-load.yaml | P/D Decode、batch=1、单个 4096 桶；需要 Foundry 和 hook；SAVE 创建存档，LOAD 要求完整存档 |
| chat-request.json | 最小非流式贪心请求 |

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @docs/zh/examples/chat-request.json
```

并发服务需要分配互不重叠的 GPU，并使用独立的共享内存目录。SAVE 与 LOAD 存档需要匹配的布局、模型和依赖。
