# CUDA Graphs

The optional CUDA Graph integration uses the repository's pinned [Foundry fork](https://github.com/asceticism-pku/foundry) to capture and restore Decode execution with less CPU launch overhead. It is disabled by default and supports P/D Decode Workers with TP=1. Prefill and Simple Workers use eager execution.

## Foundry installation

Foundry is installed separately in the same environment as Aegaeon. From the repository root:

```bash
git submodule update --init --recursive foundry
conda install -c conda-forge 'boost-cpp>=1.83' 'boost>=1.83' 'cmake>=4.0' ninja
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation -e ./foundry
export AEGAEON_FOUNDRY_HOOK_PATH="$(python -c 'from pathlib import Path; import foundry.ops; print(Path(foundry.ops.__file__).resolve().parent / "libcuda_hook.so")')"
```

Each Decode node needs Foundry and valid absolute hook/library paths before its Workers start. For library loading errors, see [Troubleshooting](troubleshooting.md).

## Graph configuration

```yaml
server:
  num_engines: 0
  num_prefill_engines: 1
  num_decode_engines: 1
  tensor_parallel_size: 1
  cuda_graph:
    mode: 'save'
    archive_dir: /srv/aegaeon-archives/qwen3
    batch_sizes: [1]
    seq_len_buckets: [4096]
```

SAVE and LOAD currently require one maximum sequence-length bucket. A Decode batch replays a graph only if its size exactly matches a configured batch size and its sequence length fits within that bucket. Other batches use eager execution. LOAD requires complete archives with a compatible memory layout; archive or layout validation errors stop loading.

## Archive creation and reuse

SAVE captures the configured graphs when a model is loaded onto a Decode Worker; LOAD restores them from the archive. The [SAVE and LOAD examples](examples/README.md) provide complete configurations. After adjusting model paths and budgets, start SAVE:

```bash
aegaeon start --config docs/en/examples/foundry-save.yaml --ray-address local
```

After `Application startup complete`, send a request from another terminal to load the model onto the Decode Worker and capture its graphs:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @docs/en/examples/chat-request.json
```

For multiple models, send a request for each configured model. Wait for the request to finish and the Worker log `captured ... Foundry decode graphs during model load`. Then stop the service and start LOAD with the same model, memory layout, and archive directory:

```bash
aegaeon start --config docs/en/examples/foundry-load.yaml --ray-address local
```

`Foundry decode replay active` identifies graph execution in Worker logs. Changes to model weights, graph sizes, memory layout, or runtime dependencies require fresh archives.
