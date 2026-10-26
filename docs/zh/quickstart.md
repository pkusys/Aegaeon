# 快速开始

本例在单张 GPU 上以 Simple 模式运行 Qwen3-4B，使用提供的 A100 40 GB profile，并假定已完成[安装](installation.md)。

## 模型与配置

[单 GPU 配置](examples/simple.yaml)定义了模型、缓存预算和一个 Simple Engine。将它保存为工作目录下的 `aegaeon.yaml`。源码仓库中的同一文件位于 `docs/zh/examples/simple.yaml`。

示例采用 `/srv/models` 作为默认路径。请按下面的方法将其替换为实际使用的模型绝对路径；存档和数据集的路径约定见[文件系统路径](configuration.md#文件系统路径)。

以下命令下载模型，并更新配置中的模型路径：

```bash
CONFIG_FILE="$PWD/aegaeon.yaml"
MODEL_DIR="$HOME/aegaeon-models/Qwen3-4B"
hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
sed -i "s|/srv/models/Qwen3-4B|$MODEL_DIR|" "$CONFIG_FILE"
```

其他 GPU 或模型需要匹配的设备条目、性能 profile 和内存预算。这些字段见[配置参考](configuration.md)。

## 服务启动

```bash
export CUDA_VISIBLE_DEVICES=0
aegaeon start --config "$CONFIG_FILE" --host 127.0.0.1 --port 8000 --ray-address local
```

启动时会加载 `startup_models` 中列出的模型。日志显示 `Application startup complete` 后，服务就绪。

## 首次请求

在另一个终端中，使用配置里的模型名称指定请求目标：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"用一句话解释 KV Cache。"}],"temperature":0,"max_tokens":64}'
```

响应包含 `choices[0].message.content`、`finish_reason` 和 `usage`。客户端接入与流式输出见[在线推理](serving.md)。
