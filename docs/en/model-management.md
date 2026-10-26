# Model Management

Aegaeon stores model replicas in each node's CPU weight cache and routes requests among ready replicas. Each request stays on one configured node for its Prefill and Decode lifecycle.

## Replica states

| State | Meaning |
|---|---|
| `absent` | No replica; omitted from placement snapshots |
| `loading` | Deployment in progress |
| `ready` | Accepts new requests |
| `draining` | Rejects new requests; retry unloading after active requests finish |
| `failed` | Deployment failed |

## Deployment and routing

```bash
aegaeon deploy Qwen/Qwen3-4B --replicas 2
aegaeon undeploy Qwen/Qwen3-4B --nodes node_1
```

`--replicas` (`replica_count` in HTTP requests) sets a minimum ready-replica target. Use undeploy to reduce the replica count. `--nodes` (`node_ids`) selects nodes directly and takes precedence over the replica target.

`server.model_placement_policy: least-loaded` selects nodes using cache capacity, eviction cost, and request load. `server.request_routing_policy: least-outstanding` selects a ready node with the fewest outstanding requests. Both accept `round-robin` as an alternative.

## Operation results

Deploy and undeploy return `status`, per-node results, and `failed`. HTTP 200 can contain `status: partial`; check these fields before treating the operation as complete. The CLI exits with status 1 for partial operations. A deployment that leaves no ready replica returns HTTP 507.

Inspect placements and routing with `GET /v1/aegaeon/runtime`. See the [CLI reference](cli.md) for command options and the [API reference](api.md) for HTTP fields.
