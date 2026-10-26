# 系统架构

```mermaid
flowchart TD
    Client[应用客户端] --> API[FastAPI 接口]
    API --> Service[LLMService 路由与副本]
    Service --> Ctrl[选中节点的 Controller]
    Ctrl --> PD[PrefillDispatcher]
    PD --> PE[PrefillEngine 与 Workers]
    PE -->|首个 token| Out[StepOutput 与响应]
    PE --> DD[DecodeDispatcher]
    DD --> DE[DecodeEngine 与 Workers]
    DE --> Out
    Ctrl --> Simple[另一种模式：SimpleDispatcher / Engine]
    Simple --> Out
    Ctrl --- BM[节点级 BlockManager]
    BM --- CPU[节点共享 CPU KV Cache]
    PE --- PKV[Prefill 各 Worker 的 GPU KV 缓存]
    DE --- DKV[Decode 各 Worker 的 GPU KV 缓存]
    Simple --- SKV[Simple 各 Worker 的 GPU KV 缓存]
    BM -. 协调 .-> PKV
    BM -. 协调 .-> DKV
    BM -. 协调 .-> SKV
    Ctrl --- QC[QuickCache CPU 权重]
    QC --> PE
    QC --> DE
    QC --> Simple
```

Simple 与 P/D 是两种可选执行模式。配置中的每个节点都有一个 Controller 和共享 CPU 缓存；同一台物理机可以运行多个这样的节点。各 Worker 持有独立的 GPU KV 缓存，由所属节点的 BlockManager 协调。

## 组件

- **API / LLMService：** 请求格式处理、副本放置与路由。
- **Controller / Schedulers：** 节点内分发、batch 和模型配额调度。
- **StageEngine / Worker：** 使用 vLLM 模型与内核执行 GPU 计算。
- **QuickCache / QuickLoader：** 共享 CPU 模型权重，并将其加载到 GPU。
- **BlockManager：** 请求 KV 状态及 CPU/GPU 搬运。

请求被路由到模型副本处于 READY 状态的节点。P/D 模式下，Prefill 产生首个 token，未结束的请求继续由 Decode 引擎处理；Simple 模式下，同一个引擎处理两个阶段。节点的 CPU KV 缓存在模型切换期间保留请求状态。详见[调度](scheduling.md)和[内存管理](memory.md)。
