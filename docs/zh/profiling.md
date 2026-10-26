# 性能数据采集

Aegaeon 使用模型与设备的时延数据估计 Prefill 和 Decode 时间。`AEGAEON_PROFILE_ROOT` 优先指定数据根目录；未设置时，优先使用包内附带的数据，否则使用仓库的 `profiles/` 目录。文件布局如下：

```text
profiles/Qwen/Qwen3-4B/A100-PCIE-40GB/i32b1.json
```

## 批量采集

在仓库根目录执行：

```bash
export AEGAEON_PROFILE_ROOT="$PWD/profiles"
CUDA_VISIBLE_DEVICES=0 python profiles/do_profile.py --model-config /srv/aegaeon/config.yaml --tp 1
```

批量采集器仅支持 `--tp 1`，采用 eager bfloat16 执行，传入的上下文长度上限为 8192。结果写入 `AEGAEON_PROFILE_ROOT` 下，日志写入 `logs/` 下。也可通过单点采集命令选择其他执行设置。

## 单点采集

需要自定义输入长度或上下文限制时：

```bash
export AEGAEON_PROFILE_ROOT="$PWD/profiles"
mkdir -p profiles/Qwen/Qwen3-4B/A100-PCIE-40GB
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark_latency.py \
  --model /srv/models/Qwen3-4B --dtype bfloat16 --enforce-eager \
  --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --input-len 32 --output-len 10 --batch-size 1 \
  --output-json profiles/Qwen/Qwen3-4B/A100-PCIE-40GB/i32b1.json
```

`AEGAEON_PROFILE_ROOT` 指向采集结果所在目录时，服务会使用这些 profile。准确的时延估计依赖于覆盖预期输入长度和 batch size 的测量数据。模型、硬件或执行设置发生变化后，需要重新测量。
