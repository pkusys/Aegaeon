# 模型准备

模型 checkpoint 包含模型配置、tokenizer 文件及所有 safetensors 权重分片。聊天请求还需要 tokenizer 的聊天模板。多节点部署时，各节点必须能通过相同的绝对路径访问 checkpoint。

## 模型下载

```bash
hf download Qwen/Qwen3-4B --local-dir /srv/models/Qwen3-4B
```

## 模型注册

```yaml
models:
- name: Qwen/Qwen3-4B
  path: /srv/models/Qwen3-4B
  profile: Qwen/Qwen3-4B
  max_model_len: 4096
```

- `name`：客户端请求中的模型标识。
- `path`：checkpoint 路径。
- `profile`：选择性能数据，见[性能数据采集](profiling.md)。
- `max_model_len`：包含输入和生成 token 的总长度。

通常省略 `params`，由 Aegaeon 读取权重元数据中的参数量；显式填写时以 `2^30` 个参数为单位。

注册用于定义模型，本身不会创建 `ready` 副本。还需选择启动时缓存的模型，或[动态部署](model-management.md)。完整模型字段见[配置参考](configuration.md)。
