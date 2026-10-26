"""Manual multi-logical-node GPU smoke test for dynamic model placement.

Each logical node owns either one SimpleEngine GPU or a Prefill/Decode GPU pair
and an isolated shared-memory directory.  The test deploys one replica per
node, reserves and submits one request per replica through load-aware routing,
verifies that every node executes a request, and finally removes all replicas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import statistics
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--profile", default="qwen2_5_7b")
    parser.add_argument(
        "--placement-probe-model-path",
        type=Path,
        help="Enable placement A/B by deploying this second model during a busy request.",
    )
    parser.add_argument(
        "--placement-probe-model-name",
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--model-placement-policy",
        choices=("least-loaded", "round-robin"),
        default="least-loaded",
    )
    parser.add_argument(
        "--engine-mode",
        choices=("simple", "pd"),
        default="simple",
        help="Use one SimpleEngine GPU or one Prefill plus one Decode GPU per node.",
    )
    parser.add_argument(
        "--gpus",
        default="4,5,6,7",
        help="Comma-separated physical GPU IDs; PD mode groups consecutive pairs.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument(
        "--routing-scenario",
        choices=("balanced", "busy-avoidance"),
        default="balanced",
    )
    parser.add_argument(
        "--request-routing-policy",
        choices=("least-outstanding", "round-robin"),
        default="least-outstanding",
    )
    parser.add_argument(
        "--warm-replicas",
        action="store_true",
        help="Run one direct 1-token request on every replica before measurement.",
    )
    parser.add_argument("--busy-output-tokens", type=int, default=64)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of sequential long/short request pairs in busy-avoidance mode.",
    )
    parser.add_argument("--model-cache-gb", type=int, default=18)
    parser.add_argument("--cpu-cache-gb", type=int, default=4)
    parser.add_argument("--memory-utilization", type=float, default=0.21)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional path for the structured JSON result.",
    )
    return parser.parse_args()


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpus contains duplicate IDs: {gpu_ids}")
    if any(not gpu_id.isdigit() for gpu_id in gpu_ids):
        raise ValueError(f"--gpus only accepts numeric GPU IDs: {gpu_ids}")
    return gpu_ids


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    """Summarize repeated request latency using nearest-rank P95."""
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def emit_result(args: argparse.Namespace, result: dict) -> None:
    print("MODEL_PLACEMENT_GPU_SMOKE_RESULT=" + json.dumps(result, sort_keys=True))
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )


def group_node_gpus(gpu_ids: list[str], engine_mode: str) -> list[str]:
    gpus_per_node = 1 if engine_mode == "simple" else 2
    if len(gpu_ids) % gpus_per_node:
        raise ValueError(
            f"{engine_mode} mode requires a multiple of {gpus_per_node} GPUs"
        )
    return [
        ",".join(gpu_ids[index : index + gpus_per_node])
        for index in range(0, len(gpu_ids), gpus_per_node)
    ]


def build_config(args: argparse.Namespace, path: Path, node_count: int) -> None:
    import yaml

    config = {
        "server": {
            "cpu_num_slabs": args.cpu_cache_gb,
            "cpu_slab_size_bytes": 1024**3,
            "memory_utilization": args.memory_utilization,
            "worker_num_cpus": 4,
            "nnodes": node_count,
            "num_prefill_engines": 1 if args.engine_mode == "pd" else 0,
            "num_decode_engines": 1 if args.engine_mode == "pd" else 0,
            "num_engines": 1 if args.engine_mode == "simple" else 0,
            "model_cache_size_gb": args.model_cache_gb,
            "dummy_weights": False,
        },
        "models": [
            {
                "id": 0,
                "name": args.model_name,
                "path": str(args.model_path.resolve()),
                "params": 7.1,
                "profile": args.profile,
                "max_model_len": args.max_model_len,
            }
        ],
        "devices": [{"name": "A100", "memory": 80, "bandwidth": 32}],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.placement_probe_model_path is not None:
        args.placement_probe_model_path = args.placement_probe_model_path.resolve()
        if not args.placement_probe_model_path.is_dir():
            raise FileNotFoundError(args.placement_probe_model_path)
    gpu_ids = parse_gpu_ids(args.gpus)
    node_gpu_groups = group_node_gpus(gpu_ids, args.engine_mode)
    node_ids = [f"node_{index}" for index in range(len(node_gpu_groups))]

    import psutil
    import ray

    shm_dirs = [
        Path(
            tempfile.mkdtemp(
                prefix=f"aegaeon-placement-node-{index}-", dir="/dev/shm"
            )
        )
        for index in range(len(gpu_ids))
    ]
    ray_dir = Path(tempfile.mkdtemp(prefix="aegaeon-placement-ray-", dir="/tmp"))
    config_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="aegaeon-placement-", suffix=".yaml", delete=False
    )
    config_path = Path(config_file.name)
    config_file.close()
    started = time.monotonic()

    def on_timeout(_signum, _frame) -> None:
        raise TimeoutError(
            f"model-placement smoke exceeded {args.timeout_seconds} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    try:
        build_config(args, config_path, len(gpu_ids))
        os.environ["AEGAEON_CONFIG_PATH"] = str(config_path)
        os.environ["AEGAEON_CONFIG_CONTENT"] = config_path.read_text(
            encoding="utf-8"
        )

        from aegaeon.config import NodeConfig
        from aegaeon.llm import LLMService
        from aegaeon.model_placement import (
            create_model_placement_policy,
            create_request_routing_policy,
        )
        from aegaeon.model_registry import get_registry, ModelSpec
        from aegaeon.models import set_model_registry
        from aegaeon.request import Request
        from vllm.tokenizers import get_tokenizer

        set_model_registry(str(config_path))
        ray.init(
            num_cpus=psutil.cpu_count(),
            resources={node_id: 1 for node_id in node_ids},
            ignore_reinit_error=True,
            _temp_dir=str(ray_dir),
        )
        node_configs = [
            NodeConfig(
                node_id=node_id,
                num_prefill_engines=1 if args.engine_mode == "pd" else 0,
                num_decode_engines=1 if args.engine_mode == "pd" else 0,
                num_engines=1 if args.engine_mode == "simple" else 0,
                cpu_num_slabs=args.cpu_cache_gb,
                cpu_slab_size_bytes=1024**3,
                model_cache_size=args.model_cache_gb * 1024**3,
                cached_models=[],
                config_path=str(config_path),
                cuda_visible_devices=gpu_id,
                shm_dir=str(shm_dir),
            )
            for node_id, gpu_id, shm_dir in zip(
                node_ids, node_gpu_groups, shm_dirs
            )
        ]
        service = LLMService(
            node_configs,
            model_placement_policy=create_model_placement_policy(
                args.model_placement_policy
            ),
            request_routing_policy=create_request_routing_policy(
                args.request_routing_policy
            ),
        )
        model = get_registry().get_by_name(args.model_name)
        assert model is not None

        deploy_started = time.monotonic()
        deploy_result = asyncio.run(
            service.deploy_model_replicas(
                model,
                str(args.model_path),
                args.max_model_len,
                node_ids=(
                    node_ids
                    if args.placement_probe_model_path is not None
                    else None
                ),
                replica_count=len(node_ids),
            )
        )
        deploy_seconds = time.monotonic() - deploy_started
        expected_initial_nodes = tuple(node_ids)
        assert (
            service.model_placements.ready_nodes(args.model_name)
            == expected_initial_nodes
        )

        tokenizer = get_tokenizer(str(args.model_path), trust_remote_code=True)
        if args.placement_probe_model_path is not None:
            if len(node_ids) != 2:
                raise ValueError("placement probe requires exactly two logical nodes")

            probe_model = ModelSpec(
                name=args.placement_probe_model_name,
                model_path=str(args.placement_probe_model_path),
                max_model_len=args.max_model_len,
            )
            busy_request = Request(
                model=model,
                arrival_time=time.time(),
                request_id=0,
                prompt_token_ids=tokenizer.encode(
                    "Keep node zero busy during placement", add_special_tokens=True
                ),
                decode_tokens=args.busy_output_tokens,
                ignore_eos=True,
                max_model_len=args.max_model_len,
            )

            async def run_placement_probe():
                busy_request.arrival_time = time.time()
                busy_node = service.reserve_node(model, busy_request.request_id)
                assert busy_node == service.nodes_by_id[node_ids[0]]
                busy_output = busy_node.serve.remote(busy_request)
                try:
                    deadline = time.monotonic() + 60
                    while True:
                        snapshot = await busy_node.runtime_snapshot.remote(0)
                        if snapshot["request_load"]["total"] > 0:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "busy request did not become active within 60 seconds"
                            )
                        await asyncio.sleep(0.05)

                    probe_started = time.monotonic()
                    probe_deploy_result = await service.deploy_model_replicas(
                        probe_model,
                        str(args.placement_probe_model_path),
                        args.max_model_len,
                        replica_count=1,
                    )
                    probe_seconds = time.monotonic() - probe_started
                    output = await busy_output
                    return (
                        output,
                        probe_deploy_result,
                        probe_seconds,
                        snapshot["request_load"],
                    )
                finally:
                    await asyncio.gather(busy_output, return_exceptions=True)
                    service.release_node_reservation(busy_request.request_id)

            (
                busy_output,
                probe_deploy_result,
                probe_seconds,
                busy_controller_load,
            ) = asyncio.run(run_placement_probe())
            assert len(busy_output) == busy_request.decode_tokens
            expected_probe_node = (
                node_ids[0]
                if args.model_placement_policy == "round-robin"
                else node_ids[1]
            )
            assert probe_deploy_result["target_nodes"] == [expected_probe_node]
            assert service.model_placements.ready_nodes(probe_model.name) == (
                expected_probe_node,
            )

            probe_undeploy_result = asyncio.run(
                service.undeploy_model_replicas(probe_model.name)
            )
            undeploy_result = asyncio.run(
                service.undeploy_model_replicas(args.model_name)
            )
            result = {
                "status": "ok",
                "scenario": "placement-busy-avoidance",
                "engine_mode": args.engine_mode,
                "model_placement_policy": args.model_placement_policy,
                "logical_nodes": dict(zip(node_ids, node_gpu_groups)),
                "busy_model_nodes": list(expected_initial_nodes),
                "busy_controller_load": busy_controller_load,
                "probe_model": probe_model.name,
                "probe_selected_node": expected_probe_node,
                "placement_decision": service.last_model_placement_decision,
                "model_placement_stats": service.model_placement_stats_snapshot(),
                "deploy_seconds": deploy_seconds,
                "probe_deploy_seconds": probe_seconds,
                "busy_request_e2e_seconds": (
                    busy_output[-1].step_time - busy_request.arrival_time
                ),
                "deploy_result": deploy_result,
                "probe_deploy_result": probe_deploy_result,
                "probe_undeploy_result": probe_undeploy_result,
                "undeploy_result": undeploy_result,
                "total_seconds": time.monotonic() - started,
            }
            emit_result(args, result)
            return

        warmup_seconds = None
        if args.warm_replicas:
            warmup_requests = [
                Request(
                    model=model,
                    arrival_time=time.time(),
                    request_id=1_000_000 + position,
                    prompt_token_ids=tokenizer.encode(
                        f"Warm replica {node_id}", add_special_tokens=True
                    ),
                    decode_tokens=1,
                    ignore_eos=True,
                    max_model_len=args.max_model_len,
                )
                for position, node_id in enumerate(node_ids)
            ]

            async def warm_all_replicas():
                return await asyncio.gather(
                    *(
                        service.nodes_by_id[node_id].serve.remote(request)
                        for node_id, request in zip(node_ids, warmup_requests)
                    ),
                    return_exceptions=True,
                )

            warmup_started = time.monotonic()
            warmup_outputs = asyncio.run(warm_all_replicas())
            warmup_seconds = time.monotonic() - warmup_started
            for output in warmup_outputs:
                if isinstance(output, BaseException):
                    raise output
                assert len(output) == 1

        if args.repetitions < 1:
            raise ValueError("--repetitions must be at least 1")
        if args.routing_scenario == "balanced" and args.repetitions != 1:
            raise ValueError("--repetitions is only supported by busy-avoidance")
        request_ids = (
            list(range(len(node_ids)))
            if args.routing_scenario == "balanced"
            else [
                request_id
                for repetition in range(args.repetitions)
                for request_id in (4 * repetition, 4 * repetition + 2)
            ]
        )
        if args.routing_scenario == "busy-avoidance" and len(node_ids) != 2:
            raise ValueError("busy-avoidance currently requires exactly two nodes")
        requests = [
            Request(
                model=model,
                arrival_time=time.time(),
                request_id=request_id,
                prompt_token_ids=tokenizer.encode(
                    f"Hello from request {request_id}", add_special_tokens=True
                ),
                decode_tokens=(
                    args.busy_output_tokens
                    if args.routing_scenario == "busy-avoidance" and position % 2 == 0
                    else args.output_tokens
                ),
                ignore_eos=True,
                max_model_len=args.max_model_len,
            )
            for position, request_id in enumerate(request_ids)
        ]

        def node_id_for(actor) -> str:
            return next(
                node_id
                for node_id, node in service.nodes_by_id.items()
                if node == actor
            )

        selected_nodes = []
        busy_controller_load = None
        busy_controller_loads = []
        routing_samples = []
        inference_started = time.monotonic()
        if args.routing_scenario == "balanced":
            arrival_time = time.time()
            for request in requests:
                request.arrival_time = arrival_time
                selected_nodes.append(
                    node_id_for(service.reserve_node(model, request.request_id))
                )
            assert selected_nodes == node_ids
            routing_before_inference = service.request_routing_snapshot()
            assert routing_before_inference["total"] == len(node_ids)
            assert routing_before_inference["by_node"] == {
                node_id: 1 for node_id in node_ids
            }
            outputs = service.serve(requests)
        else:
            async def run_busy_avoidance():
                results = []
                for repetition in range(args.repetitions):
                    first, second = requests[2 * repetition : 2 * repetition + 2]
                    first.arrival_time = time.time()
                    first_node = service.reserve_node(model, first.request_id)
                    selected_nodes.append(node_id_for(first_node))
                    first_output = first_node.serve.remote(first)
                    remote_outputs = [first_output]
                    try:
                        deadline = time.monotonic() + 60
                        while True:
                            snapshot = await first_node.runtime_snapshot.remote(0)
                            if snapshot["request_load"]["total"] > 0:
                                break
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "busy request did not become active within 60 seconds"
                                )
                            await asyncio.sleep(0.05)

                        second.arrival_time = time.time()
                        second_node = service.reserve_node(model, second.request_id)
                        selected_nodes.append(node_id_for(second_node))
                        expected_pair = (
                            [node_ids[0], node_ids[0]]
                            if args.request_routing_policy == "round-robin"
                            else node_ids
                        )
                        assert selected_nodes[-2:] == expected_pair
                        routing_samples.append(service.request_routing_snapshot())
                        busy_controller_loads.append(snapshot["request_load"])
                        second_output = second_node.serve.remote(second)
                        remote_outputs.append(second_output)
                        results.extend(
                            await asyncio.gather(
                                *remote_outputs, return_exceptions=True
                            )
                        )
                    finally:
                        await asyncio.gather(*remote_outputs, return_exceptions=True)
                        service.release_node_reservation(first.request_id)
                        service.release_node_reservation(second.request_id)
                return results

            outputs = asyncio.run(run_busy_avoidance())
            routing_before_inference = routing_samples[0]
            busy_controller_load = busy_controller_loads[0]

        inference_seconds = time.monotonic() - inference_started
        for request, output in zip(requests, outputs):
            if isinstance(output, BaseException):
                raise output
            assert len(output) == request.decode_tokens
        request_e2e_seconds = [
            output[-1].step_time - request.arrival_time
            for request, output in zip(requests, outputs)
        ]
        request_latency_summary = None
        if args.routing_scenario == "busy-avoidance":
            request_latency_summary = {
                "long": summarize_latencies(request_e2e_seconds[::2]),
                "short": summarize_latencies(request_e2e_seconds[1::2]),
            }
        routing_after_inference = service.request_routing_snapshot()
        assert routing_after_inference["total"] == 0
        assert routing_after_inference["by_node"] == {
            node_id: 0 for node_id in node_ids
        }
        assignment_counts = {
            node_id: selected_nodes.count(node_id) for node_id in node_ids
        }
        assert routing_after_inference["cumulative_assignments"] == {
            "total": len(requests),
            "by_node": assignment_counts,
            "by_model": {
                args.model_name: {
                    "total": len(requests),
                    "by_node": assignment_counts,
                }
            },
        }

        snapshots = ray.get(
            [node.runtime_snapshot.remote(0) for node in service.nodes]
        )
        switched_nodes = sorted(
            snapshot["node_id"]
            for snapshot in snapshots
            if any(event["type"] == "model_switch" for event in snapshot["events"])
        )
        expected_switched_nodes = (
            node_ids if args.warm_replicas else sorted(set(selected_nodes))
        )
        assert switched_nodes == expected_switched_nodes
        for snapshot in snapshots:
            assert args.model_name in snapshot["model_cache"]["models"]
            assert snapshot["request_load"]["total"] == 0

        undeploy_result = asyncio.run(
            service.undeploy_model_replicas(args.model_name)
        )
        assert service.model_placements.nodes(args.model_name) == ()

        result = {
            "status": "ok",
            "engine_mode": args.engine_mode,
            "routing_scenario": args.routing_scenario,
            "request_routing_policy": args.request_routing_policy,
            "warm_replicas": args.warm_replicas,
            "repetitions": args.repetitions,
            "logical_nodes": dict(zip(node_ids, node_gpu_groups)),
            "selected_nodes": selected_nodes,
            "busy_controller_load": busy_controller_load,
            "busy_controller_loads": busy_controller_loads,
            "routing_samples": routing_samples,
            "routing_before_inference": routing_before_inference,
            "routing_after_inference": routing_after_inference,
            "switched_nodes": switched_nodes,
            "deploy_seconds": deploy_seconds,
            "warmup_seconds": warmup_seconds,
            "inference_seconds": inference_seconds,
            "request_e2e_seconds": request_e2e_seconds,
            "request_latency_summary": request_latency_summary,
            "output_tokens": [len(output) for output in outputs],
            "deploy_result": deploy_result,
            "undeploy_result": undeploy_result,
            "total_seconds": time.monotonic() - started,
        }
        emit_result(args, result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if ray.is_initialized():
            ray.shutdown()
        for path in shm_dirs:
            if path.is_dir() and path.name.startswith("aegaeon-placement-node-"):
                shutil.rmtree(path)
        if ray_dir.is_dir() and ray_dir.name.startswith("aegaeon-placement-ray-"):
            shutil.rmtree(ray_dir)
        if config_path.is_file() and config_path.name.startswith(
            "aegaeon-placement-"
        ):
            config_path.unlink()


if __name__ == "__main__":
    main()
