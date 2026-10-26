# Profiling

Aegaeon uses model/device latency profiles to estimate Prefill and Decode time. `AEGAEON_PROFILE_ROOT` overrides the profile root; otherwise Aegaeon uses bundled profiles when present, then the checkout's `profiles/` directory. Files follow this layout:

```text
profiles/Qwen/Qwen3-4B/A100-PCIE-40GB/i32b1.json
```

## Batch collection

Run from the repository root:

```bash
export AEGAEON_PROFILE_ROOT="$PWD/profiles"
CUDA_VISIBLE_DEVICES=0 python profiles/do_profile.py --model-config /srv/aegaeon/config.yaml --tp 1
```

The batch collector supports only `--tp 1` and uses eager bfloat16 execution with a requested context limit of 8192. It writes results below `AEGAEON_PROFILE_ROOT` and logs below `logs/`. You can also use the single-point command to choose other execution settings.

## Single-point collection

For custom input lengths or context limits:

```bash
export AEGAEON_PROFILE_ROOT="$PWD/profiles"
mkdir -p profiles/Qwen/Qwen3-4B/A100-PCIE-40GB
CUDA_VISIBLE_DEVICES=0 python benchmark/benchmark_latency.py \
  --model /srv/models/Qwen3-4B --dtype bfloat16 --enforce-eager \
  --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --input-len 32 --output-len 10 --batch-size 1 \
  --output-json profiles/Qwen/Qwen3-4B/A100-PCIE-40GB/i32b1.json
```

The service uses the collected profiles when `AEGAEON_PROFILE_ROOT` points to their directory. Accurate latency estimates require measurements covering the expected input lengths and batch sizes. Changes to the model, hardware, or execution settings require new measurements.
