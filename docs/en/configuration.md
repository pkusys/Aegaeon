# Configuration Reference

YAML has three sections: `server` defines engine topology and memory budgets, `models` registers model paths and profiles, and `devices` describes GPU capacity and bandwidth. The CLI supplies the bind address, port, and Ray address. See the [examples](examples/README.md).

## Filesystem paths

The examples and several bundled tools assume a `/srv` directory layout by default: `/srv/models` for model weights, `/srv/aegaeon-archives` for CUDA Graph archives, and `/srv/datasets` for datasets.

Before running an example, replace its paths with absolute paths on your machine. Set each `models[].path` to the corresponding checkpoint directory and, when using CUDA Graph save/load, set `server.cuda_graph.archive_dir` to your archive location. For tools, use their path arguments or documented environment variables where available; otherwise adjust the tool's path defaults. Ensure the service can read the inputs and write the outputs. In multi-node deployments, the configured paths must be accessible on every node that uses them.

## ServerConfig defaults

| Field | Default | Meaning / constraints |
|---|---:|---|
| cpu_num_slabs | 128 | CPU KV slabs per node |
| cpu_slab_size_bytes | 1073741824 | 1 GiB per slab, in bytes |
| memory_utilization | 0.775 | GPU capacity estimate; the CUDA allocator manages actual allocations separately |
| worker_num_cpus | 16 | Ray CPUs requested per Worker |
| nnodes | 1 | Physical Ray node count |
| num_prefill_engines | 1 | Prefill Engines per node |
| num_decode_engines | 1 | Decode Engines per node |
| num_engines | 0 | Simple Engines per node; mutually exclusive with P/D |
| tensor_parallel_size | 1 | One Worker per Engine; 1 is the only supported value |
| model_cache_size_gb | 0 | CPU weight cache in GiB; 0 selects automatic sizing |
| dummy_weights | false | Placeholder weights for control-path tests only |
| decode_work_stealing | false | Node-local Decode batch transfers |
| decode_work_stealing_min_batch_gap | 2 | Minimum donor/receiver batch-count gap, at least 2 |
| model_placement_policy | least-loaded | Also accepts round-robin |
| request_routing_policy | least-outstanding | Also accepts round-robin |
| startup_models | null | null=all models; []=empty service |
| cuda_graph | CudaGraphConfig() | Disabled by default; see below |

CPU KV defaults to 128 GiB per node; size slabs and the weight cache for available host memory. Simple mode uses `num_engines > 0` with both P/D counts set to 0. P/D mode uses `num_engines: 0` with at least one engine of each stage.

`startup_models: null` loads all configured models; `[]` starts empty; a list selects models. CLI `--models` overrides this selection.

## Model entries

| Field | Default | Meaning |
|---|---|---|
| name | Required | Client-facing model name; need not match the profile folder |
| path | Empty string | Local weights; when unset, ModelSpec.path() triggers download |
| params | null | Input multiplied by 2^30 gives the parameter count; null infers it from local weight metadata |
| profile | Empty string | Performance directory alias; otherwise normalized from the model name's last segment |
| max_model_len | null | Total context length override, still subject to model capability |
| id | Unset | Generated sequentially when all are omitted; when explicit, supply an ID for every model |
| tool_parser | null | A supported vLLM parser name; overrides selection from the model name/path |

## Device entries

| Field | Meaning |
|---|---|
| name | Required logical device name matching detection/aliases |
| memory | Required; multiplied by 2^30 to obtain bytes |
| bandwidth | Optional; multiplied by 2^30 to obtain bytes/s; omission attempts NVML detection |
| profile | Retained in DeviceSpec.extra as a performance directory alias |
| Other fields | Retained in DeviceSpec.extra; only fields read by an implementation have an effect |

## CUDA graph defaults and constraints

| Field | Class default | Constraints |
|---|---|---|
| mode | 'off' | 'off' / save / load; quote each value |
| archive_dir | '' | Required for save/load |
| batch_sizes | [1,2,4,8] | Nonempty positive integers, sorted and deduplicated |
| seq_len_buckets | [128,256,512,1024,2048] | Class accepts multiple; **Worker save/load currently requires exactly one maximum-length bucket** |
| strict | true | Controls strict checks such as initialization scratch bounds; ABI/layout/binding checks always run |
| base_address | 0x400000000000 | Positive and 2 MiB aligned; accepts a quoted hexadecimal string |
| region_size_bytes | 274877906944 | 256 GiB virtual address range; positive and 2 MiB aligned |
| scratch_size_bytes | 4294967296 | 4 GiB; positive, 2 MiB aligned, and smaller than region_size_bytes |

CUDA Graph save/load is opt-in and runs on P/D Decode Workers when `mode` is `'save'` or `'load'`. See [CUDA Graphs](cuda-graphs.md).
