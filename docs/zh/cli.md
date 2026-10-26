# CLI 参考

使用 `aegaeon` 或 `python -m aegaeon`。

## doctor

`doctor` 检查当前 Python 环境和本机运行条件。提供配置文件后，还会检查模型路径、性能 profile、拓扑和内存预算。

```bash
aegaeon doctor
CUDA_VISIBLE_DEVICES=0 aegaeon doctor --check-cuda
```

| 参数 | 解析器默认值 | 说明 |
|---|---|---|
| --config | None | 用于本地模型、性能 profile、拓扑和内存检查的 YAML 文件 |
| --models | None | 逗号分隔的启动模型列表；需要 --config |
| --tensor-parallel-size | None | 非零值覆盖 YAML，0 沿用 YAML；需要 --config，最终值必须为 1 |
| --strict | false | 警告和失败均返回非零退出码 |
| --check-cuda | false | 执行小型 PyTorch/CuPy CUDA 运算，导入 vLLM 原生扩展，检查可见 GPU，并测试单页主机内存注册 |

## start

```bash
aegaeon start --config /srv/aegaeon/config.yaml --host 127.0.0.1 --port 8000 --ray-address 127.0.0.1:6789
```

| 参数 | 解析器默认值 | 说明 |
|---|---|---|
| --config | 必填 | YAML 文件 |
| --host | 0.0.0.0 | 监听地址 |
| --port | 8000 | 监听端口 |
| --models | None | 逗号分隔的启动模型列表；覆盖 startup_models；空字符串表示启动时不加载模型 |
| --num-prefill-engines | 1 | 正常路径以 YAML ServerConfig 为准 |
| --num-decode-engines | 1 | 同上 |
| --num-engines | 0 | 同上 |
| --tensor-parallel-size | None | 显式非零值覆盖 YAML；0 沿用 YAML |
| --model-cache-size-gb | 20 | CPU 权重缓存容量回退值，单位 GiB |
| --cpu-num-slabs | 15 | slab 数量回退值 |
| --cpu-slab-size-gb | 1 | slab 大小回退值，单位 GiB |
| --ray-address | None | 指定集群，优先于环境变量 |

拓扑和内存预算在 [YAML 配置](configuration.md)中设置。

## deploy

```bash
aegaeon deploy Qwen/Qwen3-4B --path /srv/models/Qwen3-4B --replicas 2
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| model | 位置参数，必填 | 服务中的模型名字 |
| --path | None | 本地权重目录；未指定时先查注册配置，再尝试下载 |
| --host | localhost | API 服务主机 |
| --port | 8000 | API 端口 |
| --max-model-len | None | 覆盖上下文长度 |
| --replicas | 1 | 自动部署时的最小 READY 副本数 |
| --nodes | None | 显式节点 ID；绕过自动放置 |
| --tool-parser | None | 为该模型别名覆盖工具解析器，取值须为支持的 vLLM 解析器 |

部分成功时返回非零退出码，见[模型管理](model-management.md)。

## undeploy

```bash
aegaeon undeploy Qwen/Qwen3-4B --nodes node_1
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| model | 必填位置参数 | 服务中的模型名 |
| --host | localhost | API 主机 |
| --port | 8000 | API 端口 |
| --nodes | 全部已知副本 | 只卸载列出的节点 |

## 查询

各命令的帮助列出参数及解析器默认值。

```bash
aegaeon --help
aegaeon doctor --help
aegaeon start --help
aegaeon deploy --help
aegaeon undeploy --help
```

启动资源以 YAML 字段及其默认值为准，优先于旧的资源回退选项。`--models` 和非零 `--tensor-parallel-size` 覆盖对应 YAML 字段；Ray 地址依次取 `--ray-address`、`AEGAEON_RAY_ADDRESS`、`RAY_ADDRESS`、`auto`。
