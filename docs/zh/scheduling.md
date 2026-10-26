# 调度与 SLO

Aegaeon 按模型组织请求，在 batch 的执行轮次之间切换模型。Decode 结合[性能 profile](profiling.md)分配时间配额，平衡切换开销与 token 时延。

Simple 模式在同一个引擎中调度 Prefill 和 Decode。P/D 模式使用独立的引擎池：Prefill 按到达顺序和队列负载分发，Decode 按时间配额调度各模型的 batch。

## SLO 设置

TTFT 和 TPOT 目标以秒为单位，在调度模块导入时读取。API 进程和所有 Ray 节点环境都需要在启动前设置相同的值：

```bash
export AEGAEON_TTFT_SLO=3
export AEGAEON_TPOT_SLO=0.03
```

TTFT 默认 10 秒，TPOT 默认 0.1 秒。

## 策略作用域

| 策略 | 作用域 |
|---|---|
| `model_placement_policy` | 部署时选择副本节点 |
| `request_routing_policy` | 为新请求选择 READY 节点 |
| Prefill/Decode 调度 | 在节点内选择 batch 和模型配额 |
| [Decode Work Stealing](work-stealing.md) | 在节点内均衡等待中的 Decode batch |

指标定义和工具见[基准测试](benchmarking.md)。
