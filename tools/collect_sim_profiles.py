#!/usr/bin/env python3
"""Collect measured vLLM profiles for the simulator (stdlib-only --help)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


MODEL_KEYS = ("qwen2.5-7b", "qwen2.5-1.5b")
PERCENTILES = (10, 25, 50, 75, 90)
GPU_FIELDS = (
    "index", "uuid", "name", "driver_version", "memory.total",
    "clocks.current.graphics", "clocks.current.memory", "pstate",
)
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_USE_V1",
    "VLLM_ATTENTION_BACKEND", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
)
PROTOCOL = {
    "units": "seconds",
    "request_selection": "first returned request in each completed batch",
    "prefill": "first_token_ts - scheduled_ts",
    "decode": "(last_token_ts - first_token_ts) / (generated_tokens - 1)",
    "batch_validation": "all requests finished with exactly output_len tokens and valid metrics",
    "percentiles": "linear interpolation at (n - 1) * percentile / 100",
    "warmup_in_statistics": False,
    "swap": "not measured; simulator swap parameters are separate assumptions",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cases(output_len=10):
    """Same fit limits as benchmark/profile_latency.sh: 33 default points."""
    return [
        (length, batch)
        for length in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        for batch in (1, 2, 4, 8)
        if length * batch <= 8192 and (length + output_len) * batch < 10240
    ]


def inspect_gpu(device_key, *, environ=None, runner=subprocess.run):
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if len(ids) != 1 or ids[0] == "-1":
            raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or UUID")
    else:
        ids = []
    command = ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"]
    if ids:
        command.append("--id=" + ids[0])
    try:
        result = runner(command, check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot inspect GPU with nvidia-smi") from error
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
        raise ValueError("Exactly one visible GPU is required; set CUDA_VISIBLE_DEVICES")
    gpu = dict(zip(GPU_FIELDS, (item.strip() for item in rows[0])))
    try:
        memory_mib = float(gpu["memory.total"])
    except ValueError as error:
        raise ValueError("Invalid GPU memory reported by nvidia-smi") from error
    if device_key != "a100_40gb" or "A100" not in gpu["name"] or not 39000 <= memory_mib <= 42000:
        raise ValueError(f"device-key {device_key!r} does not match {gpu['name']} ({memory_mib} MiB)")
    # Pin the queried physical GPU so CUDA and nvidia-smi identify the same device.
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    gpu["memory_unit"] = "MiB"
    gpu["clock_unit"] = "MHz"
    return gpu


def checkpoint_metadata(model, model_key):
    root = Path(model).resolve()
    if not root.is_dir():
        raise ValueError("--model must name a local checkpoint directory")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hidden = {"qwen2.5-7b": 3584, "qwen2.5-1.5b": 1536}[model_key]
    if config.get("model_type") != "qwen2" or config.get("hidden_size") != expected_hidden:
        raise ValueError(f"Checkpoint config does not match {model_key}")
    weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("Local checkpoint contains no safetensors or pytorch_model weights")
    identity_files = {config_path, *weights}
    for pattern in ("*.index.json", "tokenizer*.json", "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"):
        identity_files.update(root.glob(pattern))
    files = {}
    for path in sorted(identity_files):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for index_path in root.glob("*.index.json"):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for filename in set(index.get("weight_map", {}).values()):
            if filename not in files:
                raise ValueError(f"Missing checkpoint shard referenced by {index_path.name}: {filename}")
    return {"directory_name": root.name, "config": config, "files": files, "weights_hashing": "complete SHA256 of every weight file"}


def package_versions():
    versions = {}
    for name in ("vllm", "torch", "transformers", "numpy", "safetensors", "aegaeon"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_identity(root):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(status.strip())}


def percentile(values, percentage):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(prefill, decode):
    if not prefill or len(prefill) != len(decode):
        raise ValueError("A profile requires paired nonempty observations")
    return {
        "avg_prefill_latency": statistics.mean(prefill),
        "avg_decode_latency": statistics.mean(decode),
        "prefill_latencies": prefill,
        "decode_latencies": decode,
        "prefill_percentiles": {str(p): percentile(prefill, p) for p in PERCENTILES},
        "decode_percentiles": {str(p): percentile(decode, p) for p in PERCENTILES},
    }


def validate_outputs(outputs, batch_size, output_len):
    if len(outputs) != batch_size:
        raise RuntimeError(f"Expected {batch_size} completed requests, got {len(outputs)}")
    timings = []
    for index, request in enumerate(outputs):
        if not request.finished:
            raise RuntimeError(f"Request {index} is unfinished")
        if len(request.outputs) != 1 or len(request.outputs[0].token_ids) != output_len:
            raise RuntimeError(f"Request {index} must generate exactly {output_len} tokens")
        metrics = request.metrics
        if metrics is None:
            raise RuntimeError(f"Request {index} has no timing metrics")
        stamps = [metrics.scheduled_ts, metrics.first_token_ts, metrics.last_token_ts]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in stamps):
            raise RuntimeError(f"Request {index} has invalid timing metrics")
        prefill = stamps[1] - stamps[0]
        decode = (stamps[2] - stamps[1]) / (output_len - 1)
        if not all(math.isfinite(value) and value > 0 for value in (prefill, decode)):
            raise RuntimeError(f"Request {index} has nonpositive latency")
        timings.append((prefill, decode))
    return timings[0]


def engine_arguments(args):
    return {
        "model": str(Path(args.model).resolve()), "trust_remote_code": False,
        "tensor_parallel_size": 1, "dtype": "bfloat16", "kv_cache_dtype": "auto",
        "enforce_eager": True, "async_scheduling": False,
        "enable_prefix_caching": False, "enable_chunked_prefill": False,
        "max_model_len": 8192, "max_num_batched_tokens": 8192, "max_num_seqs": 8,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": False, "seed": args.seed,
    }


def shutdown_engine(llm, baseline_children):
    """Close vLLM and reap only multiprocessing children created by this collector."""
    try:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None:
            core.shutdown(timeout=60)
        elif hasattr(llm, "shutdown"):
            llm.shutdown()
    finally:
        for child in multiprocessing.active_children():
            if child.pid in baseline_children:
                continue
            child.join(timeout=3)
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)
            if child.is_alive():
                child.kill()
                child.join(timeout=3)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def collect(args, *, llm_factory=None, sampling_factory=None, gpu_probe=inspect_gpu):
    if args.warmup < 0 or args.iterations < 1 or args.output_len < 2:
        raise ValueError("warmup >= 0, iterations >= 1 and output-len >= 2 are required")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    grid = cases(args.output_len)
    if not grid:
        raise ValueError("No profile points fit the token limits")
    destination = Path(args.output_root).resolve() / args.model_key / args.device_key
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    gpu = gpu_probe(args.device_key)
    checkpoint = checkpoint_metadata(args.model, args.model_key)
    if llm_factory is None or sampling_factory is None:
        from vllm import LLM, SamplingParams
        llm_factory = LLM if llm_factory is None else llm_factory
        sampling_factory = SamplingParams if sampling_factory is None else sampling_factory
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.device_key + "-collect-", dir=destination.parent))
    llm = None
    baseline_children = {child.pid for child in multiprocessing.active_children()}
    started = utc_now()
    llm_args = engine_arguments(args)
    try:
        llm = llm_factory(**llm_args)
        sampling_args = {"temperature": 0.0, "top_p": 1.0, "ignore_eos": True, "max_tokens": args.output_len, "seed": args.seed}
        sampling = sampling_factory(**sampling_args)
        rng = random.Random(args.seed)
        point_hashes = {}
        for number, (input_len, batch_size) in enumerate(grid, 1):
            prompts = [{"prompt_token_ids": [rng.randrange(min(10000, checkpoint["config"]["vocab_size"])) for _ in range(input_len)]} for _ in range(batch_size)]
            prompt_digest = hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
            prefill, decode = [], []
            for iteration in range(args.warmup + args.iterations):
                outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
                prefill_time, decode_time = validate_outputs(outputs, batch_size, args.output_len)
                if iteration >= args.warmup:
                    prefill.append(prefill_time)
                    decode.append(decode_time)
            filename = f"i{input_len}b{batch_size}.json"
            data = {
                "_synthetic": False, "source": "measured", "manifest": "metadata.json",
                "model_key": args.model_key, "device_key": args.device_key,
                "input_len": input_len, "batch_size": batch_size, "output_len": args.output_len,
                "seed": args.seed, "prompt_sha256": prompt_digest, "units": "seconds",
                **aggregate(prefill, decode),
            }
            write_json(staging / filename, data)
            point_hashes[filename] = sha256_file(staging / filename)
            print(f"[{number}/{len(grid)}] {filename}: prefill={data['avg_prefill_latency']:.6f}s decode={data['avg_decode_latency']:.6f}s", flush=True)
        shutdown_engine(llm, baseline_children)
        llm = None
        metadata = {
            "schema_version": 1, "source": "measured", "_synthetic": False, "complete": True,
            "model_key": args.model_key, "device_key": args.device_key,
            "collection_started_utc": started, "collection_finished_utc": utc_now(),
            "checkpoint": checkpoint, "gpu": gpu, "packages": package_versions(),
            "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
            "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
            "git": git_identity(Path(__file__).resolve().parents[1]),
            "collector_sha256": sha256_file(__file__),
            "parameters": {"seed": args.seed, "warmup": args.warmup, "iterations": args.iterations, "output_len": args.output_len,
                           "engine": {key: value for key, value in llm_args.items() if key != "model"}, "sampling": sampling_args,
                           "input_lengths": sorted({length for length, _ in grid}), "batch_sizes": [1, 2, 4, 8],
                           "fit_limits": {"max_prefill_tokens_inclusive": 8192, "max_total_tokens_exclusive": 10240}},
            "timing_protocol": PROTOCOL, "point_count": len(point_hashes), "point_sha256": point_hashes,
        }
        write_json(staging / "metadata.json", metadata)
        # A populated dataset cannot be replaced by POSIX rename; recheck for races.
        if destination.exists():
            raise FileExistsError(f"Dataset appeared during collection: {destination}")
        staging.rename(destination)
        return destination
    finally:
        if llm is not None:
            shutdown_engine(llm, baseline_children)
        if staging.exists():
            shutil.rmtree(staging)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="local Qwen2.5 Instruct checkpoint directory (no download)")
    result.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    result.add_argument("--device-key", choices=("a100_40gb",), default="a100_40gb")
    result.add_argument("--output-root", type=Path, default=Path("work/collected-sim-profiles"))
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--warmup", type=int, default=4)
    result.add_argument("--iterations", type=int, default=12)
    result.add_argument("--output-len", type=int, default=10)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        destination = collect(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"Profile collection failed: {error}", file=sys.stderr)
        return 1
    print(f"Published complete measured dataset: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
