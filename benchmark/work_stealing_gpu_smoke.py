"""Manual 1P+2D GPU smoke test for decode work stealing.

This script is intentionally kept out of the pytest suite.  It uses two local
checkpoints and long prompts to form non-mergeable Decode batches, submits an
asymmetric workload in a deterministic order, and verifies both output
completion and the runtime work-stealing counter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import shutil
import statistics
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-paths",
        type=Path,
        nargs=2,
        required=True,
        metavar=("MODEL_A", "MODEL_B"),
    )
    parser.add_argument(
        "--profiles",
        nargs=2,
        default=["qwen2_5_7b", "qwen2_5_7b"],
        metavar=("PROFILE_A", "PROFILE_B"),
    )
    parser.add_argument("--device-profile", default="A100")
    parser.add_argument("--long-tokens", type=int, default=512)
    parser.add_argument("--short-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument(
        "--work-stealing",
        choices=("on", "off"),
        default="on",
        help="Enable work stealing or run the original dispatcher baseline.",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional path for the structured JSON result.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, path: Path) -> list[str]:
    import yaml

    work_stealing = args.work_stealing == "on"
    model_names = ["ws-smoke-a", "ws-smoke-b"]
    parameter_counts = [0.6, 7.1]
    config = {
        "server": {
            "cpu_num_slabs": 8,
            "cpu_slab_size_bytes": 1024**3,
            "memory_utilization": 0.21,
            "worker_num_cpus": 4,
            "nnodes": 1,
            "num_prefill_engines": 1,
            "num_decode_engines": 2,
            "num_engines": 0,
            "model_cache_size_gb": 18,
            "decode_work_stealing": work_stealing,
            "decode_work_stealing_min_batch_gap": 2,
        },
        "models": [
            {
                "id": index,
                "name": name,
                "path": str(args.model_paths[index].resolve()),
                "params": parameter_counts[index],
                "profile": args.profiles[index],
            }
            for index, name in enumerate(model_names)
        ],
        "devices": [
            {
                "name": "A100",
                "memory": 80,
                "bandwidth": 32,
                "profile": args.device_profile,
            }
        ],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return model_names


def serve_preordered(service, requests) -> list:
    """Submit in a deterministic order, then collect all generated outputs."""
    import ray

    node = service.nodes[0]
    for request in requests:
        ray.get(node.submit.remote(request))

    async def collect(request) -> list:
        outputs = []
        while True:
            new_outputs, is_finished = await node.wait_for_step_outputs.remote(
                request.request_id, len(outputs)
            )
            outputs.extend(new_outputs)
            if is_finished:
                await node.release_request.remote(request.request_id)
                return outputs

    async def collect_all() -> list:
        return await asyncio.gather(*(collect(request) for request in requests))

    return asyncio.run(collect_all())


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((percentage / 100) * (len(ordered) - 1))
    return ordered[index]


def summarize_run(args: argparse.Namespace, requests, outputs, elapsed: float) -> dict:
    from aegaeon.utils import (
        TTFT_SLO,
        TPOT_SLO,
        compute_request_latencies,
        compute_request_metrics,
    )

    request_results = []
    for request, request_outputs in zip(requests, outputs):
        ttft, attainment, per_token = compute_request_metrics(
            request, request_outputs
        )
        latency_breakdown = compute_request_latencies(request, request_outputs)
        request_results.append(
            {
                "request_id": request.request_id,
                "model": request.model.name,
                "input_tokens": request.get_input_len(),
                "output_tokens": len(request_outputs),
                "ttft_seconds": ttft,
                "mean_tpot_seconds": (
                    statistics.fmean(per_token[1:]) if len(per_token) > 1 else None
                ),
                "e2e_seconds": request_outputs[-1].step_time - request.arrival_time,
                "slo_attainment": attainment,
                "per_token_seconds": per_token,
                **latency_breakdown,
            }
        )

    makespan = max(
        request_outputs[-1].step_time for request_outputs in outputs
    ) - min(request.arrival_time for request in requests)
    ttfts = [result["ttft_seconds"] for result in request_results]
    tpots = [
        result["mean_tpot_seconds"]
        for result in request_results
        if result["mean_tpot_seconds"] is not None
    ]
    attainments = [result["slo_attainment"] for result in request_results]
    total_output_tokens = sum(result["output_tokens"] for result in request_results)
    return {
        "setting": "work_stealing" if args.work_stealing == "on" else "baseline",
        "total_runtime_seconds": elapsed,
        "workload_makespan_seconds": makespan,
        "output_throughput_tokens_per_second": total_output_tokens / makespan,
        "request_count": len(request_results),
        "output_tokens": total_output_tokens,
        "ttft_slo_seconds": TTFT_SLO,
        "tpot_slo_seconds": TPOT_SLO,
        "mean_slo_attainment": statistics.fmean(attainments),
        "median_ttft_seconds": statistics.median(ttfts),
        "p95_ttft_seconds": percentile(ttfts, 95),
        "median_tpot_seconds": statistics.median(tpots),
        "p95_tpot_seconds": percentile(tpots, 95),
        "requests": request_results,
    }


def main() -> None:
    args = parse_args()
    missing_paths = [path for path in args.model_paths if not path.is_dir()]
    if missing_paths:
        raise FileNotFoundError(f"model directories do not exist: {missing_paths}")

    import psutil
    import ray

    config_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="aegaeon-ws-smoke-", suffix=".yaml", delete=False
    )
    config_file.close()
    config_path = Path(config_file.name)
    shm_dir = Path(tempfile.mkdtemp(prefix="aegaeon-ws-smoke-", dir="/dev/shm"))
    ray_temp_dir = tempfile.mkdtemp(prefix="aegaeon-ws-ray-")

    service = None
    started_at = time.monotonic()
    work_stealing = args.work_stealing == "on"

    def on_timeout(_signum, _frame) -> None:
        raise TimeoutError(
            f"smoke test exceeded configured timeout ({args.timeout_seconds}s)"
        )

    previous_alarm_handler = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    try:
        model_names = build_config(args, config_path)
        os.environ["AEGAEON_CONFIG_PATH"] = str(config_path)
        os.environ["AEGAEON_CONFIG_CONTENT"] = config_path.read_text(encoding="utf-8")
        os.environ["AEGAEON_SHM_DIR"] = str(shm_dir)

        from aegaeon.models import set_model_registry

        set_model_registry(str(config_path))
        from aegaeon.model_registry import get_registry
        from aegaeon import LLMService, NodeConfig, Request

        registry = get_registry()
        models = [registry.get_by_name(name) for name in model_names]
        assert all(model is not None for model in models)

        ray.init(
            num_cpus=psutil.cpu_count(),
            resources={"node_0": 1},
            _temp_dir=ray_temp_dir,
        )
        service = LLMService(
            [
                NodeConfig(
                    node_id="node_0",
                    num_prefill_engines=1,
                    num_decode_engines=2,
                    model_cache_size=18 * 1024**3,
                    cpu_num_slabs=8,
                    cpu_slab_size_bytes=1024**3,
                    cached_models=models,
                    config_path=str(config_path),
                    decode_work_stealing=work_stealing,
                    decode_work_stealing_min_batch_gap=2,
                )
            ]
        )

        # Model A first occupies D0.  At 21% memory utilization, the 16K Model B
        # requests remain separate Decode batches and are assigned alternately:
        # requests 1/3 to D1 and request 2 to D0.  Requests 1/3 are deliberately
        # short, leaving D1 idle while D0 still owns long Model A and B batches.
        request_models = [models[0], models[1], models[1], models[1]]
        requests = [
            Request(
                model=model,
                arrival_time=time.time(),
                request_id=index,
                prompt_token_ids=[1]
                * (args.prompt_tokens // 2 if index == 0 else args.prompt_tokens),
                decode_tokens=(
                    args.short_tokens if index in (1, 3) else args.long_tokens
                ),
                ignore_eos=True,
            )
            for index, model in enumerate(request_models)
        ]
        outputs = serve_preordered(service, requests)
        elapsed = time.monotonic() - started_at

        output_lengths = [
            len(output) if isinstance(output, list) else -1 for output in outputs
        ]
        expected_lengths = [request.decode_tokens for request in requests]
        if output_lengths != expected_lengths:
            raise AssertionError(
                f"incomplete outputs: expected={expected_lengths}, actual={output_lengths}, "
                f"raw_types={[type(output).__name__ for output in outputs]}"
            )

        snapshot = ray.get(service.nodes[0].runtime_snapshot.remote())
        stats = snapshot["work_stealing"]
        result = summarize_run(args, requests, outputs, elapsed)
        result["output_lengths"] = output_lengths
        result["work_stealing"] = stats
        result["decode_load"] = snapshot["decode_load"]
        print("WORK_STEALING_GPU_SMOKE_RESULT=" + json.dumps(result, sort_keys=True))
        if work_stealing and (stats is None or stats["successes"] < 1):
            raise AssertionError(f"work stealing did not trigger: {stats}")
        if not work_stealing and stats is not None and stats["successes"] != 0:
            raise AssertionError(f"disabled work stealing changed state: {stats}")
        if args.result_file is not None:
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            args.result_file.write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        if service is not None and ray.is_initialized():
            for node in service.nodes:
                ray.kill(node, no_restart=True)
        if ray.is_initialized():
            ray.shutdown()
        config_path.unlink(missing_ok=True)
        # The shared-memory files are owned by this process-specific directory.
        for child in shm_dir.iterdir():
            child.unlink(missing_ok=True)
        shm_dir.rmdir()
        shutil.rmtree(ray_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
