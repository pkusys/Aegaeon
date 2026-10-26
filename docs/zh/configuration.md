# 配置参考

YAML 包含三部分：`server` 定义 Engine 拓扑和内存预算，`models` 注册模型路径和性能 profile，`devices` 描述 GPU 容量与带宽。监听地址、端口和 Ray 地址通过 CLI 传入。完整配置见[示例](examples/README.md)。

## 文件系统路径

示例和部分随附工具默认采用 `/srv` 目录布局：`/srv/models` 存放模型权重，`/srv/aegaeon-archives` 存放 CUDA Graph 存档，`/srv/datasets` 存放数据集。

运行示例前，请将路径替换为本机实际使用的绝对路径。将各个 `models[].path` 设为对应的 checkpoint 目录；使用 CUDA Graph save/load 时，将 `server.cuda_graph.archive_dir` 设为所需的存档位置。对于工具，优先使用其路径参数或文档说明的环境变量；若未提供这些选项，则修改工具中的默认路径。确保服务能够读取输入并写入输出。多节点部署时，所有使用这些路径的节点都必须能够访问相应文件。[快速开始](quickstart.md)演示了如何替换配置中的路径。

## ServerConfig 默认值

| 字段 | 默认值 | 含义 / 注意事项 |
|---|---:|---|
| cpu_num_slabs | 128 | 每节点 CPU KV slab 数量 |
| cpu_slab_size_bytes | 1073741824 | 每 slab 1 GiB，单位 bytes |
| memory_utilization | 0.775 | 显存容量估算比例；CUDA allocator 另行管理实际分配 |
| worker_num_cpus | 16 | 每 Worker Ray CPU 资源请求 |
| nnodes | 1 | 物理 Ray 节点数 |
| num_prefill_engines | 1 | 每节点 Prefill Engine 数 |
| num_decode_engines | 1 | 每节点 Decode Engine 数 |
| num_engines | 0 | 每节点 Simple Engine 数；与 P/D 引擎互斥 |
| tensor_parallel_size | 1 | 每 Engine 一个 Worker；当前只支持值 1 |
| model_cache_size_gb | 0 | CPU 权重缓存 GiB；0 进入自动估算路径 |
| dummy_weights | false | 测试占位权重；仅用于控制路径测试 |
| decode_work_stealing | false | 节点内 Decode 批次转移 |
| decode_work_stealing_min_batch_gap | 2 | donor/receiver 批次数差阈值，至少 2 |
| model_placement_policy | least-loaded | 或 round-robin |
| request_routing_policy | least-outstanding | 或 round-robin |
| startup_models | null | null=全部；[]=空服务 |
| cuda_graph | CudaGraphConfig() | 默认关闭；详见下表 |

CPU KV 默认每节点 128 GiB，slab 和权重缓存需匹配可用主机内存。Simple 模式使用 `num_engines > 0`，Prefill 与 Decode Engine 数量均为 0；P/D 模式使用 `num_engines: 0`，两个阶段至少各有一个 Engine。

`startup_models: null` 加载所有配置模型，`[]` 启动空服务，列表选择指定模型。CLI 的 `--models` 优先。

## models 条目

| 字段 | 默认 | 说明 |
|---|---|---|
| name | 必填 | 对外模型名字；profile 文件夹可使用独立名称 |
| path | 空字符串 | 本地权重目录；省略时 `ModelSpec.path()` 触发下载 |
| params | null | 输入数字乘以 2^30 得到参数量；null 表示从本地权重元数据推断 |
| profile | 空字符串 | 性能目录别名；省略时由模型名称最后一段规范化生成 |
| max_model_len | null | 覆盖上下文总长度；仍受模型能力限制 |
| id | 省略 | 所有模型都省略时按顺序生成；显式设置时为每个模型都填写 ID |
| tool_parser | null | 支持的 vLLM 工具解析器名称；覆盖根据模型名称或路径选择的解析器 |

## devices 条目

| 字段 | 说明 |
|---|---|
| name | 必填，设备逻辑名；应与检测/别名匹配 |
| memory | 必填，乘以 2^30 得到容量 bytes |
| bandwidth | 可选，乘以 2^30 得到 bytes/s；省略会尝试 NVML 检测 |
| profile | 通过 DeviceSpec.extra 保存，指定性能目录别名 |
| 其他字段 | 保存到 DeviceSpec.extra；只有实现中显式读取的字段才会生效 |

## cuda_graph 默认与约束

| 字段 | 配置类默认值 | 约束 |
|---|---|---|
| mode | 'off' | 'off' / save / load；各取值均使用引号 |
| archive_dir | '' | save/load 必填 |
| batch_sizes | [1,2,4,8] | 正整数、非空；会排序去重 |
| seq_len_buckets | [128,256,512,1024,2048] | 配置类接受多个；**当前 Worker save/load 必须只填一个最大长度桶** |
| strict | true | 控制初始化 scratch 等严格检查；ABI/layout/binding 校验始终执行 |
| base_address | 0x400000000000 | 正整数、2 MiB 对齐；可写带引号十六进制字符串 |
| region_size_bytes | 274877906944 | 256 GiB 虚拟地址范围；正整数、2 MiB 对齐 |
| scratch_size_bytes | 4294967296 | 4 GiB；正整数、2 MiB 对齐，且小于 region_size_bytes |

CUDA Graph save/load 默认关闭；`mode` 为 `'save'` 或 `'load'` 时，在 P/D Decode Worker 上运行。详见 [CUDA Graph](cuda-graphs.md)。
