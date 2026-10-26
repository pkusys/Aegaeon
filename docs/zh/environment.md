# 环境变量

环境变量配置进程级默认值和动态库路径。变量在启动或模块导入时读取，修改后重启生效。Worker 使用的设置也需要在对应 Ray 节点上可用。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `AEGAEON_WEB_DIR` | 未设置 | 可选的外部 UI 资源目录 |
| `AEGAEON_LOG_FILE` | `output.log` | 服务日志路径 |
| `AEGAEON_TTFT_SLO` | `10` | TTFT 目标，秒 |
| `AEGAEON_TPOT_SLO` | `0.1` | TPOT 目标，秒 |
| `AEGAEON_DEFAULT_MODEL_REPO` | `modelscope` | 模型来源：modelscope 或 huggingface |
| `AEGAEON_PROFILE_ROOT` | 内置 profile，其次为仓库 `profiles/` | 模型与设备的性能 profile |
| `AEGAEON_SHM_DIR` | `/dev/shm` | 共享内存缓存目录 |
| `AEGAEON_RAY_ADDRESS` | `RAY_ADDRESS` / `auto` | Ray 集群地址，CLI 优先 |
| `AEGAEON_RAY_TEMP_DIR` | 未设置 | 本地 Ray 临时目录 |
| `AEGAEON_ARENA_FRACTION` | `0.875` | QuickLoader GPU arena 比例 |
| `CUDA_VISIBLE_DEVICES` | 进程环境 | 可见 GPU 集合 |
| `AEGAEON_MAX_REQUESTS_PER_SECOND` | 未设置 | 每模型请求数/秒，按 60 秒平均，包含待接入的当前请求 |
| `AEGAEON_MAX_TOKENS_PER_SECOND` | 未设置 | 每模型已生成 token 数/秒，按 60 秒平均，在请求接入时检查 |
| `AEGAEON_FOUNDRY_HOOK_PATH` | 未设置 | `libcuda_hook.so` 绝对路径 |
| `AEGAEON_CUDA_COMPAT_PATH` | 自动查找 | CUDA compatibility 动态库目录 |
| `LD_LIBRARY_PATH` | 进程环境 | 动态库搜索路径 |
| `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` | 进程环境 | 下载与网络代理 |

图设置使用 [YAML 配置](configuration.md)，启动命令见 [CLI 参考](cli.md)。
