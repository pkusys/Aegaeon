# Decode Work Stealing

Work Stealing 在同一个 Controller 内均衡 Decode batch，默认关闭，需要至少两个 Decode 引擎才能转移任务。

## 配置

```yaml
server:
  num_engines: 0
  num_prefill_engines: 1
  num_decode_engines: 2
  decode_work_stealing: true
  decode_work_stealing_min_batch_gap: 2
```

## 转移规则

一轮调度结束后，供出方可以向队列为空的引擎转移一个完整的等待 batch，但不能转移队首 batch。候选 batch 没有正在执行的迭代或待合入的请求，KV 状态位于 CPU。

转移时，两端 batch 数量差必须达到配置值，且该值至少为 2。系统优先选择已驻留相同模型的接收方，以减少模型切换。

接收方从节点共享的 CPU KV 缓存恢复 batch。转移限于同一节点内，并沿用现有的 Decode 配额策略。

## 监控

`GET /v1/aegaeon/runtime` 中，`work_stealing` 提供转移计数，`decode_load` 提供各引擎的工作量和队列指标。接口用法见[日志与监控](observability.md)。
