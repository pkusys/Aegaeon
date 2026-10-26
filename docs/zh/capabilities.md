# 功能与兼容性

## 支持范围

| 功能 | 支持情况 |
|---|---|
| 多模型推理 | Token 级调度与模型切换 |
| 部署 | 单节点或多节点的 Simple、Prefill/Decode 模式 |
| 模型管理 | 动态部署、卸载、多副本与请求路由 |
| HTTP | OpenAI 风格 Chat/Completion、SSE、分词及运行监控 |
| 生成 | 单候选贪心解码 |
| 函数工具 | 使用模型对应的 vLLM parser，客户端执行函数 |
| Decode Work Stealing | 可选的节点内 batch 均衡 |
| CUDA Graph | P/D Decode Worker 的 Foundry 保存与加载 |
| 多模态输入 | 当前仅提供文本输入与文本生成 |
| 量化、随机采样、beam search、logprobs | 当前配置范围之外 |

仅支持 `tensor_parallel_size=1`，Pipeline Parallel 固定为 1。多 GPU 通过增加 Engine 扩展。

用法见[模型准备](models.md)、[生成参数](generation.md)和 [HTTP API](api.md)。
