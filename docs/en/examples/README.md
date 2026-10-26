# Configuration Examples

These examples cover common service topologies. Model paths, GPU assignments, profile aliases, and memory budgets depend on the deployment environment.

| File | Topology |
|---|---|
| simple.yaml | One node, 1 Simple Engine, TP=1 |
| pd.yaml | One node, 1P+1D, TP=1 |
| multi-node.yaml | Two physical nodes, 1 Simple Engine per node |
| dynamic.yaml | Empty startup followed by explicit deployment |
| work-stealing.yaml | One node, 1P+2D |
| foundry-save.yaml / foundry-load.yaml | P/D Decode, batch=1, one 4096 bucket; requires Foundry and the hook; SAVE creates archives, LOAD requires complete archives |
| chat-request.json | Minimal nonstreaming greedy request |

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @docs/en/examples/chat-request.json
```

Concurrent services need disjoint GPU assignments and separate shared-memory directories. SAVE and LOAD archives require matching layouts, models, and dependencies.
