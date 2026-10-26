# Logs and Monitoring

## Logs

Aegaeon writes INFO logs to stdout and to `AEGAEON_LOG_FILE`, which defaults to `output.log`. You can also set an absolute log path before startup:

```bash
mkdir -p logs
export AEGAEON_LOG_FILE="$PWD/logs/server.log"
```

The path is forwarded to Ray actors, so its parent directory must exist on every node. Ray actor stdout and stderr are also available in the Ray session log directory.

## Runtime interfaces

| Endpoint | Contents |
|---|---|
| `/health` | API liveness |
| `/v1/models` | The service's deployed model list |
| `/v1/aegaeon/runtime` | Nodes, engines, replicas, routing, caches, and Decode load |
| `/v1/aegaeon/events` | Engine-state snapshots and model-switch events over SSE |
| `/v1/aegaeon/metrics` | Per-model request and generated-token rates, counts, and configured limits |

```bash
curl -sS http://127.0.0.1:8000/v1/aegaeon/runtime
curl -N http://127.0.0.1:8000/v1/aegaeon/events
```

Use replica state `ready` in the runtime snapshot to confirm model availability. Request rates and generated-token rates are per-model averages over a sliding 60-second window. Set `AEGAEON_MAX_REQUESTS_PER_SECOND` and `AEGAEON_MAX_TOKENS_PER_SECOND` before API startup to limit those rates; exceeding a limit returns HTTP 429.
