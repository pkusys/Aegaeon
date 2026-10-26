# Deployment

Each configured Aegaeon node runs one Controller and shares its CPU caches among that node's engines. Currently supported configuration assumes TP=1, so each engine uses one GPU. Standard deployments use one Aegaeon node per physical Ray node.

A multi-GPU host normally uses `nnodes: 1`. Python integrations can isolate multiple logical nodes on one host with separate `NodeConfig.cuda_visible_devices` and `NodeConfig.shm_dir` settings.

## Execution modes

| Mode | Server settings |
|---|---|
| Simple | `num_engines > 0`; both P/D counts are 0 |
| P/D | `num_engines: 0`; at least one Prefill and one Decode Engine |

Use the [configuration examples](examples/README.md) for single-GPU, P/D, or multi-node layouts. Make the same model files and matching profiles available on every node.

## Ray cluster

For a single host, pass `--ray-address local`. For two hosts, start Ray on each:

```bash
# Head node
ray start --head --port=6789 --resources='{"node_0":1}'
```

```bash
# Worker node
ray start --address=HEAD_IP:6789 --resources='{"node_1":1}'
```

Then start the API from the head node with `server.nnodes: 2` in its configuration:

```bash
aegaeon start --config /srv/aegaeon/config.yaml --ray-address HEAD_IP:6789
```

## Resources

Each node needs one visible GPU per configured engine. Its Ray CPU budget must cover `server.worker_num_cpus` logical CPUs per Worker, plus capacity for the Controller. `ray status` reports the cluster's CPU and GPU resources.

To override the default [SLO settings](scheduling.md#slo-settings), set the variables in the API and all Ray node environments before startup. Independent services need separate Ray instances and shared-memory directories. Stopping the foreground service disconnects its Ray driver; manage an existing Ray cluster separately.
