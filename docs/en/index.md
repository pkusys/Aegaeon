# Welcome to Aegaeon

<div class="home-hero">
<div class="hero-wordmark"><svg viewBox="0 0 72 80" aria-hidden="true"><path d="M36 3 69 71H54L36 33 18 71H3Z" fill="#5368d6"/><path d="M27 56h18l7 15H20Z" fill="#91a1ef"/><path d="M36 3 45 22 36 33 27 22Z" fill="#bdc8ff"/></svg><span>Aegaeon</span></div>
<p class="hero-tagline">Efficient Model-as-a-Service, powered by token-level autoscaling.</p>
<div class="hero-actions"><a class="primary-action" href="quickstart.md">Get Started <span aria-hidden="true">→</span></a><a class="secondary-action" href="architecture.md">Architecture <span aria-hidden="true">↗</span></a></div>
</div>

Aegaeon is a blazingly fast inference system for deploying multiple AI models on shared GPUs, enabling Model-as-a-Service (MaaS) both in local and cloud environments.

## Key Features

- **Token-level autoscaling** — switch models between token steps instead of waiting for whole requests to finish.
- **SLO-aware scheduling** — coordinates Prefill and Decode work against TTFT and TPOT objectives.
- **Fast model switching** — combines host-side weight caching, prefetching, a Rust quick loader, and explicit KV-cache movement.
- **Multi-model runtime** — supports dynamic model deployment, multiple replicas, and load-aware request routing.
- **vLLM backend** — inherits model and kernel support across dense, hybrid, and MoE architectures from vLLM.
- **Serving and observability** — provides OpenAI-style Chat and Completion APIs, SSE streaming, and runtime state telemetry.

See our [SOSP '25 paper](https://doi.org/10.1145/3731569.3764815) for more technical details.

## Documentation

| Guide | Contents |
|---|---|
| [Quickstart](quickstart.md) | Install, configure, and send a request |
| [Deployment](deployment.md) | Single-node and multi-node serving |
| [Architecture](architecture.md) | Request flow and runtime components |
| [Benchmarking](benchmarking.md) | Metrics, profiling, and performance tools |
| [HTTP API](api.md) | Endpoints, schemas, and responses |
| [CLI](cli.md) | Start, deploy, and query |

See [Features and Compatibility](capabilities.md) for the supported configuration.
