# Model Swapping Simulator

A CPU-only [SimPy](https://simpy.readthedocs.io/en/latest) simulator for model placement, switching, and scheduling policies.

## Run

From the repository root in a Python 3.10+ environment:

```bash
python -m pip install aenum typing_extensions simpy==4.1.1 numpy scikit-learn matplotlib
python -m sim.simulate --num-models 8 --arrival-rate 0.1 --devices a100 \
  --profile-root /path/to/execution --job-timeout 300
```

Profiles use `<profile-root>/<model>/<device>/i<input_length>b<batch_size>.json`. Each file supplies `avg_prefill_latency` and `avg_decode_latency` in seconds. The bundled eight-model workload uses `qwen2.5-7b/a100` and `yi-9b/a100` profiles.

## Outputs and options

Results default to `plots/json/`. Set `--output-dir` and `--name` to choose another location; `--job-timeout` limits each simulation job. Run `python -m sim.simulate --help` for all options.

The estimator fits Prefill and Decode latency from input length, batch size, and model dimensions, following [DistServe](https://arxiv.org/abs/2401.09670). Supply profiles covering the workload's lengths and batches.
