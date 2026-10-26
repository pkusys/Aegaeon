# HTTP API

Aegaeon exposes an OpenAI-style API for deployed models and listens on port 8000 by default. The running service publishes request schemas and defaults at `/docs`, `/redoc`, and `/openapi.json`. Authentication is configured at your gateway.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Fixed health response; no deep actor checks |
| GET | /v1/models | Deployed model list |
| GET | /v1/models/{model_name:path} | Deployed model metadata; the path converter accepts names containing / |
| POST | /v1/models/deploy | Deploy CPU weight cache replicas. See [Model Management](model-management.md) |
| POST | /v1/models/undeploy | Unload selected or all replicas. See [Model Management](model-management.md) |
| POST | /v1/chat/completions | Chat, nonstreaming or SSE |
| POST | /v1/completions | Legacy Completion, nonstreaming or SSE |
| POST | /tokenize | Tokenize a plain prompt |
| POST | /detokenize | Decode token IDs |
| GET | /v1/aegaeon/runtime | Cluster runtime snapshot |
| GET | /v1/aegaeon/events | Runtime SSE |
| GET | /v1/aegaeon/metrics | 60-second access metrics and rate-limit settings |

## Errors

| Status | Meaning |
|---|---|
| 400 | Invalid model or parameter, or insufficient KV-cache capacity for a request |
| 422 | Invalid request schema, including unsupported generation fields |
| 429 | Access rate limit |
| 503 | No READY replica |
| 507 | Deployment failed and no READY replica remains |
| 500 | Server error; inspect service logs |

A KV-cache capacity failure after streaming starts is returned as SSE data containing an `error` object, followed by `data: [DONE]`.

Runtime interfaces are described in [Monitoring](observability.md).
