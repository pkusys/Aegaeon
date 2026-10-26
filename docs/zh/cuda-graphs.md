# CUDA Graph

可选的 CUDA Graph 集成使用仓库锁定的 [Foundry fork](https://github.com/asceticism-pku/foundry)捕获和恢复 Decode 执行，以减少 CPU 提交开销。该功能默认关闭，支持 P/D 模式下 TP=1 的 Decode Worker。Prefill 和 Simple Worker 使用 eager 执行。

## Foundry 安装

Foundry 需要单独安装在 Aegaeon 所在的环境中。从仓库根目录执行：

```bash
git submodule update --init --recursive foundry
conda install -c conda-forge 'boost-cpp>=1.83' 'boost>=1.83' 'cmake>=4.0' ninja
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation -e ./foundry
export AEGAEON_FOUNDRY_HOOK_PATH="$(python -c 'from pathlib import Path; import foundry.ops; print(Path(foundry.ops.__file__).resolve().parent / "libcuda_hook.so")')"
```

每个 Decode 节点都需在 Worker 启动前安装 Foundry，并设置有效的 hook 和动态库绝对路径。动态库加载错误见[故障排查](troubleshooting.md)。

## 图配置

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

SAVE 和 LOAD 目前都要求只配置一个最大序列长度桶。只有 batch size 与配置值精确匹配，且序列长度不超过桶上限的 Decode batch 才会回放 graph，其余 batch 使用 eager 执行。LOAD 需要完整存档和兼容的内存布局；存档或布局校验错误会中止加载。

## 存档创建与复用

SAVE 在模型加载到 Decode Worker 时捕获配置的 graph，LOAD 从存档恢复。[SAVE 和 LOAD 示例](examples/README.md)提供完整配置。调整模型路径和预算后启动 SAVE：

```bash
aegaeon start --config docs/zh/examples/foundry-save.yaml --ray-address local
```

出现 `Application startup complete` 后，在另一终端发送请求，将模型加载到 Decode Worker 并捕获 graph：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @docs/zh/examples/chat-request.json
```

多模型配置需为每个配置的模型发送请求。等待请求完成，并确认 Worker 日志出现 `captured ... Foundry decode graphs during model load`，再停止服务，以相同的模型、内存布局和存档目录启动 LOAD：

```bash
aegaeon start --config docs/zh/examples/foundry-load.yaml --ray-address local
```

Worker 日志中的 `Foundry decode replay active` 表示正在执行 graph。更改权重、graph 尺寸、内存布局或运行时依赖后，需要重新生成存档。
