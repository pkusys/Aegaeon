# Quickstart

This example runs Qwen3-4B on one GPU in Simple mode. It assumes [installation](installation.md) is complete and uses the provided A100 40 GB profile.

## Model and configuration

The [single-GPU configuration](examples/simple.yaml) defines the model, its cache budgets, and one Simple Engine. Save it as `aegaeon.yaml` in your working directory. A source checkout contains the same file at `docs/en/examples/simple.yaml`.

The `/srv/models` path in the example is a default convention. Replace it with your own absolute model path as shown below; see [filesystem paths](configuration.md#filesystem-paths) for the archive and dataset conventions.

The following commands download the model and update its path in that configuration:

```bash
CONFIG_FILE="$PWD/aegaeon.yaml"
MODEL_DIR="$HOME/aegaeon-models/Qwen3-4B"
hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
sed -i "s|/srv/models/Qwen3-4B|$MODEL_DIR|" "$CONFIG_FILE"
```

Another GPU or model needs matching device entries, performance profiles, and memory budgets. These fields are described in the [Configuration Reference](configuration.md).

## Service startup

```bash
export CUDA_VISIBLE_DEVICES=0
aegaeon start --config "$CONFIG_FILE" --host 127.0.0.1 --port 8000 --ray-address local
```

Startup loads the models listed in `startup_models`. The service is ready when the log reports `Application startup complete`.

## First request

In another terminal, the configured model name identifies the request’s destination:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Explain KV Cache in one sentence."}],"temperature":0,"max_tokens":64}'
```

The response includes `choices[0].message.content`, `finish_reason`, and `usage`. [Online Inference](serving.md) covers client integration and streaming.
