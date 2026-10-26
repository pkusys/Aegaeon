# Model Preparation

A model checkpoint contains the model configuration, tokenizer assets, and all safetensors weight shards. Chat requests also need the tokenizer's chat template. In multi-node deployments, the checkpoint must be available at the same absolute path on every node.

## Model download

```bash
hf download Qwen/Qwen3-4B --local-dir /srv/models/Qwen3-4B
```

## Model registration

```yaml
models:
- name: Qwen/Qwen3-4B
  path: /srv/models/Qwen3-4B
  profile: Qwen/Qwen3-4B
  max_model_len: 4096
```

- `name` identifies the model in client requests.
- `path` locates the checkpoint.
- `profile` selects performance data; see [Profiling](profiling.md).
- `max_model_len` includes both prompt and generated tokens.

Usually omit `params` so Aegaeon reads parameter counts from the weight metadata. Explicit values use units of `2^30` parameters.

Registration defines a model; it does not by itself create a `ready` replica. Select models for startup caching or [deploy them dynamically](model-management.md). The [configuration reference](configuration.md) lists all model fields.
