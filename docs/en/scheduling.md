# Scheduling and SLOs

Aegaeon groups requests by model and switches models between batch turns. Decode time quotas use [performance profiles](profiling.md) to balance switching overhead against token latency.

Simple mode schedules Prefill and Decode in one engine. P/D mode uses separate engine pools: Prefill dispatches by arrival order and queue load, while Decode schedules model batches with time quotas.

## SLO settings

TTFT and TPOT targets are expressed in seconds and read when the scheduling modules are imported. The API process and all Ray node environments need the same values before startup:

```bash
export AEGAEON_TTFT_SLO=3
export AEGAEON_TPOT_SLO=0.03
```

Defaults are 10 seconds for TTFT and 0.1 seconds for TPOT.

## Policy scopes

| Policy | Scope |
|---|---|
| `model_placement_policy` | Selects replica nodes at deployment |
| `request_routing_policy` | Selects a READY node for a new request |
| Prefill/Decode scheduling | Selects batches and model quotas within a node |
| [Decode Work Stealing](work-stealing.md) | Balances waiting Decode batches within a node |

For metric definitions and tools, see [Benchmarking](benchmarking.md).
