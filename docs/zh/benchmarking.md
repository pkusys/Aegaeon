# 基准测试

## 指标

| 指标 | 含义 |
|---|---|
| TTFT | 请求到达到首个 token 的时间 |
| TPOT | 首个输出 token 之后的平均 token 间隔 |
| E2E | 请求到达到完成的时间 |
| 吞吐量 | 每秒完成的请求数或生成的 token 数 |
| CV | 总体标准差除以均值 |

Aegaeon 的 `qos` 统计采用基于 TTFT 和 TPOT 目标的累计时间比值。下方流式示例测量到首个非空文本 chunk 的时间。

## 工具

API 示例向已运行的服务发送请求。在仓库根目录执行：

```bash
python benchmark/demo_api.py --help
python benchmark/demo_api.py --host 127.0.0.1 --port 8000 \
  --model-config /srv/aegaeon/config.yaml --model Qwen/Qwen3-4B
```

| 工具 | 用途 |
|---|---|
| `benchmark/work_stealing_gpu_smoke.py` | 运行本地 1P+2D Work Stealing 负载 |
| `benchmark/compare_work_stealing_results.py` | 比较已保存的 Work Stealing 结果 |
| `benchmark/model_placement_gpu_smoke.py` | 运行本地放置与路由负载 |
| `benchmark/replay_workload.py` | 按 CSV 到达时间向已有 HTTP 服务回放请求 |
| `tools/benchmark_foundry_three_model_latency.py` | 从已有运行记录重放三模型 CUDA Graph 实验 |
| `sim/` | 纯 CPU 调度仿真，见 `sim/README.md` |

各工具通过 `--help` 提供参数说明。使用匹配的[性能数据](profiling.md)，并区分启动、模型切换和稳态测量。

使用服务上已部署的模型别名，回放仓库内的负载：

```bash
python benchmark/replay_workload.py --csv workloads/synth-2-0.1-1.0-1.0.csv \
  --host 127.0.0.1 --port 8000 --models qwen2_5_7b qwen2_1_5b
```

三模型 Foundry 延迟工具要求显式传入 `--source-run /path/to/validated-run`。
该目录须包含验证通过且带有 `archive_root` 的 `report.json`，以及配置了
Qwen3-4B、Qwen3.5-9B 和 Gemma-3-4B 的 `multi-model-load.yaml`。
正式测量前可用 `--preflight-only` 检查来源归档和本地模型路径。

## 调优

TTFT 较高可能源于排队、加载或 Prefill 容量不足。Decode 间隔较长可能源于模型竞争、数据搬运或配额设置。先检查这些原因，再调整 Engine 数量和内存预算。[Work Stealing](work-stealing.md) 用于改善 Decode 负载不均，[CUDA Graph](cuda-graphs.md) 用于减少提交开销。
