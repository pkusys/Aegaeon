# Decode Work Stealing

Work Stealing balances Decode batches within one Controller. It is disabled by default and needs at least two Decode engines to transfer work.

## Configuration

```yaml
server:
  num_engines: 0
  num_prefill_engines: 1
  num_decode_engines: 2
  decode_work_stealing: true
  decode_work_stealing_min_batch_gap: 2
```

## Transfer rules

After a scheduling round, a donor can transfer one complete waiting batch, excluding its queue head, to an engine with an empty queue. The batch has no active iteration or pending admissions, and its KV state is on CPU.

The transfer requires the configured batch-count gap, which is at least 2. A receiver whose resident model matches the batch is preferred to reduce switching.

The receiver resumes the batch from the node's shared CPU KV cache. Transfers remain within that node and use the existing Decode quota policy.

## Monitoring

`GET /v1/aegaeon/runtime` exposes `work_stealing` transfer counters and `decode_load` per-engine work and queue metrics. See [Monitoring](observability.md) for runtime access.
