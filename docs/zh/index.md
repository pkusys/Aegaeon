# 欢迎使用 Aegaeon

<div class="home-hero">
<div class="hero-wordmark"><svg viewBox="0 0 72 80" aria-hidden="true"><path d="M36 3 69 71H54L36 33 18 71H3Z" fill="#5368d6"/><path d="M27 56h18l7 15H20Z" fill="#91a1ef"/><path d="M36 3 45 22 36 33 27 22Z" fill="#bdc8ff"/></svg><span>Aegaeon</span></div>
<p class="hero-tagline">以 Token 级自动扩缩容驱动高效的模型即服务。</p>
<div class="hero-actions"><a class="primary-action" href="quickstart.md">快速开始 <span aria-hidden="true">→</span></a><a class="secondary-action" href="architecture.md">系统架构 <span aria-hidden="true">↗</span></a></div>
</div>

Aegaeon 是一个高速推理系统，可在共享 GPU 上部署多个 AI 模型，在本地和云环境中提供模型即服务（Model-as-a-Service，MaaS）。

## 关键功能

- **Token 级自动扩缩容** — 在 Token 步骤之间切换模型，无需等待整个请求完成。
- **SLO 感知调度** — 根据 TTFT 和 TPOT 目标协调 Prefill 与 Decode 工作。
- **快速模型切换** — 结合主机侧权重缓存、预取、Rust 快速加载器和显式 KV-cache 搬移。
- **多模型运行时** — 支持动态模型部署、多副本和负载感知的请求路由。
- **vLLM 后端** — 继承 vLLM 对 Dense、Hybrid 和 MoE 架构的模型及算子支持。
- **服务与可观测性** — 提供 OpenAI 风格的 Chat 和 Completion API、SSE 流式输出及运行状态遥测。

更多技术细节见我们的 [SOSP '25 论文](https://doi.org/10.1145/3731569.3764815)。

## 文档

| 指南 | 内容 |
|---|---|
| [快速开始](quickstart.md) | 安装、配置与发送请求 |
| [部署](deployment.md) | 单节点与多节点服务 |
| [系统架构](architecture.md) | 请求流程与运行组件 |
| [基准测试](benchmarking.md) | 指标、Profiling 和性能工具 |
| [HTTP API](api.md) | 端点、schema 与响应 |
| [CLI](cli.md) | 启动、部署与查询 |

配置范围见[功能与兼容性](capabilities.md)。
