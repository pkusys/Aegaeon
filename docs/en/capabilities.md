# Features and Compatibility

## Supported scope

| Capability | Support |
|---|---|
| Multi-model inference | Token-level scheduling and model switching |
| Deployment | Simple or Prefill/Decode mode across one or more nodes |
| Model management | Dynamic deployment, unloading, replicas, and request routing |
| Serving | OpenAI-style Chat/Completion, tokenization, and runtime monitoring |
| Function tools | Model-specific vLLM parsers; client-side execution |
| CUDA Graphs | Foundry save/load on P/D Decode Workers |

**Coming Next**:
- Multimodal input
- Quantization
- Advanced sampling / logprobs options
- Tensor Parallelism (TP) and Pipeline Parallelism (PP)

See [Model Preparation](models.md), [Generation Parameters](generation.md), and the [HTTP API](api.md) for detailed usage.
