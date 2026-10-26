#!/usr/bin/env python3
"""Compare Aegaeon request and model-switch latency with Foundry OFF vs LOAD.

The benchmark starts two otherwise identical Aegaeon servers sequentially:

* ``cuda_graph.mode=off``: eager baseline without Foundry CUDA Graphs.
* ``cuda_graph.mode=load``: pre-recorded Foundry CUDA Graph archives.

For each mode it measures warm, active-model 512-token streaming requests and
A -> B -> A switch requests. A switch request emits one token and is followed
immediately by an active-model one-token control request; their latency
difference estimates model-switch overhead independently of decode time.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmark" / "foundry-vmm-qwen-family-load.yaml"
DEFAULT_HOOK = ROOT / "foundry" / "python" / "foundry" / "libcuda_hook.so"
DEFAULT_OUTPUT = ROOT / "logs" / "cuda-graph-latency-comparison.json"
MODEL_A = "Qwen/Qwen3-4B"
MODEL_B = "Qwen/Qwen3.5-9B"
PROMPT = "What are the differences between processes and threads?"


@dataclass
class Measurement:
    mode: str
    kind: str
    model: str
    pair_id: str | None
    e2e_s: float
    ttft_s: float | None
    stream_done_s: float
    curl_time_starttransfer_s: float | None
    curl_time_total_s: float | None
    completion_tokens: int

    @property
    def output_tokens_per_second(self) -> float:
        return self.completion_tokens / self.e2e_s if self.e2e_s > 0 else 0.0


@dataclass
class Deployment:
    mode: str
    model: str
    pair_id: str
    latency_s: float


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait_for_health(
    process: subprocess.Popen,
    base_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Aegaeon exited before becoming healthy (exit={return_code})"
            )
        try:
            response = _http_json("GET", f"{base_url}/health", timeout=5)
            if response.get("status") == "ok":
                return
        except (URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(
        f"Aegaeon did not become healthy within {timeout:.0f}s: {last_error}"
    )


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=90)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _tail(path: Path, line_count: int = 120) -> str:
    if not path.is_file():
        return "<log file was not created>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _load_config(path: Path, model_names: tuple[str, str]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config is not a YAML mapping: {path}")
    server = raw.get("server")
    if not isinstance(server, dict):
        raise RuntimeError("Config is missing server mapping")
    graph = server.get("cuda_graph")
    if not isinstance(graph, dict) or graph.get("mode") != "load":
        raise RuntimeError("Source config server.cuda_graph.mode must be 'load'")
    if server.get("startup_models") != []:
        raise RuntimeError("server.startup_models must be []")

    configured = {
        item.get("name"): item
        for item in raw.get("models", [])
        if isinstance(item, dict)
    }
    missing = [name for name in model_names if name not in configured]
    if missing:
        raise RuntimeError(f"Models missing from config: {missing}")

    archive_root = Path(str(graph.get("archive_dir", ""))).expanduser().resolve()
    if not archive_root.is_dir():
        raise RuntimeError(f"Archive catalog does not exist: {archive_root}")
    for model_name in model_names:
        from urllib.parse import quote

        key = quote(model_name, safe="").replace(".", "%2E")
        rank_dir = archive_root / "models" / key / "rank_0"
        for filename in (
            "fatbin_image_packed.img",
            "fatbin_entrypoint_packed.txt",
            "vmm_layout.json",
            "graph_manifest.json",
        ):
            required = rank_dir / filename
            if not required.is_file():
                raise RuntimeError(f"Missing Foundry archive file: {required}")
    return raw


def _write_mode_config(
    source: dict[str, Any],
    mode: str,
    path: Path,
) -> None:
    config = copy.deepcopy(source)
    config["server"]["cuda_graph"]["mode"] = mode
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _deploy(
    base_url: str,
    model: dict[str, Any],
    timeout: float,
) -> float:
    payload: dict[str, Any] = {
        "model": model["name"],
        "model_path": model["path"],
        "replica_count": 1,
    }
    if model.get("max_model_len") is not None:
        payload["max_model_len"] = model["max_model_len"]
    start = time.perf_counter()
    response = _http_json(
        "POST",
        f"{base_url}/v1/models/deploy",
        payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if response.get("status") not in (None, "ok"):
        raise RuntimeError(
            f"Unexpected deploy response for {model['name']}: {response}"
        )
    return elapsed


def _stream_request(
    base_url: str,
    mode: str,
    kind: str,
    model_name: str,
    max_tokens: int,
    timeout: float,
    pair_id: str | None = None,
) -> Measurement:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    marker = "__CURL_TIMING__"
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--no-buffer",
        "--noproxy",
        "localhost,127.0.0.1,::1",
        "--max-time",
        str(timeout),
        f"{base_url}/v1/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload, ensure_ascii=False),
        "--write-out",
        f"\n{marker}%{{time_starttransfer}}|%{{time_total}}|%{{http_code}}\n",
    ]

    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    first_token_at: float | None = None
    generation_done_at: float | None = None
    stream_done_at: float | None = None
    completion_tokens: int | None = None
    token_ids: list[int] | None = None
    curl_starttransfer: float | None = None
    curl_total: float | None = None
    http_status: int | None = None

    for raw_line in process.stdout:
        now = time.perf_counter()
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(marker):
            values = line[len(marker) :].split("|")
            if len(values) == 3:
                curl_starttransfer = float(values[0])
                curl_total = float(values[1])
                http_status = int(values[2])
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            stream_done_at = now
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            process.kill()
            raise RuntimeError(f"Invalid SSE JSON from {model_name}: {data}") from exc

        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            completion_tokens = int(usage["completion_tokens"])
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            if first_token_at is None:
                first_token_at = now
        if isinstance(choice.get("token_ids"), list):
            token_ids = choice["token_ids"]
        if choice.get("finish_reason") is not None:
            generation_done_at = now

    stderr = process.stderr.read()
    return_code = process.wait()
    exit_at = time.perf_counter()
    if return_code != 0:
        raise RuntimeError(
            f"curl failed for {model_name} ({kind}, exit={return_code}): {stderr}"
        )
    if http_status != 200:
        raise RuntimeError(
            f"HTTP {http_status} for {model_name} ({kind}); see the server log"
        )
    if generation_done_at is None:
        raise RuntimeError(
            f"No final generation chunk received for {model_name} ({kind})"
        )
    if stream_done_at is None:
        raise RuntimeError(f"No [DONE] marker received for {model_name} ({kind})")
    if completion_tokens is None:
        completion_tokens = len(token_ids or [])
    if completion_tokens <= 0:
        raise RuntimeError(f"No output tokens received for {model_name} ({kind})")

    return Measurement(
        mode=mode,
        kind=kind,
        model=model_name,
        pair_id=pair_id,
        e2e_s=generation_done_at - start,
        ttft_s=None if first_token_at is None else first_token_at - start,
        stream_done_s=stream_done_at - start,
        curl_time_starttransfer_s=curl_starttransfer,
        curl_time_total_s=curl_total or (exit_at - start),
        completion_tokens=completion_tokens,
    )


def _validate_mode_log(mode: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal_markers = ("[HOOK] ERROR", "CUDA error", "Traceback")
    found = [marker for marker in fatal_markers if marker in text]
    if found:
        raise RuntimeError(f"{mode} server log contains errors: {found}")
    if mode == "load":
        if "Loaded 36 unique binaries from 2 packed archives" not in text:
            raise RuntimeError("LOAD log is missing the multi-archive binary marker")
        if "loaded 64 Foundry decode graph shapes" not in text:
            raise RuntimeError("LOAD log is missing active-model graph loads")
    elif "loaded 64 Foundry decode graph shapes" in text:
        raise RuntimeError("OFF baseline unexpectedly loaded Foundry CUDA Graphs")


def _run_mode(
    args: argparse.Namespace,
    mode: str,
    config_path: Path,
    models: dict[str, dict[str, Any]],
) -> tuple[list[Measurement], list[Deployment], Path]:
    base_url = f"http://{args.host}:{args.port}"
    log_path = args.log_dir / f"cuda-graph-latency-{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AEGAEON_FOUNDRY_HOOK_PATH"] = str(args.hook)
    env["AEGAEON_CUDA_COMPAT_PATH"] = str(args.cuda_compat)
    python_paths = [str(ROOT), str(ROOT / "foundry" / "python")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    command = [
        sys.executable,
        "-m",
        "aegaeon",
        "start",
        "--config",
        str(config_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ray-address",
        args.ray_address,
    ]
    print(f"\n[{mode.upper()}] Starting server")
    print("  " + " ".join(command))
    print(f"  log: {log_path}")

    measurements: list[Measurement] = []
    deployments: list[Deployment] = []
    process: subprocess.Popen | None = None
    failure: BaseException | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            _wait_for_health(process, base_url, args.startup_timeout)
            print(f"[{mode.upper()}] Server healthy")

            initial_deploy = _deploy(
                base_url, models[args.model_a], args.request_timeout
            )
            deployments.append(
                Deployment(mode, args.model_a, "initial-a", initial_deploy)
            )
            print(f"[{mode.upper()}] Warmup {args.model_a}")
            _stream_request(
                base_url,
                mode,
                "warmup",
                args.model_a,
                args.warmup_tokens,
                args.request_timeout,
            )

            for index in range(args.requests):
                result = _stream_request(
                    base_url,
                    mode,
                    "steady",
                    args.model_a,
                    args.max_tokens,
                    args.request_timeout,
                )
                measurements.append(result)
                print(
                    f"[{mode.upper()}] steady {index + 1}/{args.requests}: "
                    f"E2E={result.e2e_s:.3f}s, TTFT={result.ttft_s:.3f}s, "
                    f"tokens={result.completion_tokens}, "
                    f"tok/s={result.output_tokens_per_second:.2f}"
                )

            for cycle in range(1, args.switch_cycles + 1):
                for direction, model_name in (
                    ("a-to-b", args.model_b),
                    ("b-to-a", args.model_a),
                ):
                    pair_id = f"cycle-{cycle}-{direction}"
                    deploy_s = _deploy(
                        base_url,
                        models[model_name],
                        args.request_timeout,
                    )
                    deployments.append(Deployment(mode, model_name, pair_id, deploy_s))
                    switched = _stream_request(
                        base_url,
                        mode,
                        "switch",
                        model_name,
                        args.switch_max_tokens,
                        args.request_timeout,
                        pair_id,
                    )
                    control = _stream_request(
                        base_url,
                        mode,
                        "switch_control",
                        model_name,
                        args.switch_max_tokens,
                        args.request_timeout,
                        pair_id,
                    )
                    measurements.extend((switched, control))
                    overhead = max(0.0, switched.e2e_s - control.e2e_s)
                    print(
                        f"[{mode.upper()}] {direction} cycle {cycle}: "
                        f"deploy={deploy_s:.3f}s, switch-request={switched.e2e_s:.3f}s, "
                        f"active-control={control.e2e_s:.3f}s, "
                        f"estimated-switch={overhead:.3f}s"
                    )
        except BaseException as exc:
            failure = exc
        finally:
            _stop_server(process)

    if failure is not None:
        print(f"\n===== {mode.upper()} server log tail =====", file=sys.stderr)
        print(_tail(log_path), file=sys.stderr)
        raise failure
    _validate_mode_log(mode, log_path)
    return measurements, deployments, log_path


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


def _summarize(
    modes: list[str],
    measurements: list[Measurement],
    deployments: list[Deployment],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
        mode_rows = [row for row in measurements if row.mode == mode]
        steady = [row for row in mode_rows if row.kind == "steady"]
        switched = {
            row.pair_id: row
            for row in mode_rows
            if row.kind == "switch" and row.pair_id is not None
        }
        controls = {
            row.pair_id: row
            for row in mode_rows
            if row.kind == "switch_control" and row.pair_id is not None
        }
        overheads = [
            max(0.0, row.e2e_s - controls[pair_id].e2e_s)
            for pair_id, row in switched.items()
            if pair_id in controls
        ]
        summary[mode] = {
            "steady_e2e_s": _stats([row.e2e_s for row in steady]),
            "steady_ttft_s": _stats(
                [row.ttft_s for row in steady if row.ttft_s is not None]
            ),
            "steady_output_tokens_per_second": _stats(
                [row.output_tokens_per_second for row in steady]
            ),
            "switch_request_e2e_s": _stats([row.e2e_s for row in switched.values()]),
            "switch_control_e2e_s": _stats([row.e2e_s for row in controls.values()]),
            "estimated_switch_overhead_s": _stats(overheads),
            "deploy_api_s": _stats(
                [row.latency_s for row in deployments if row.mode == mode]
            ),
        }

    if "off" in summary and "load" in summary:
        off_e2e = summary["off"]["steady_e2e_s"]["mean"]
        load_e2e = summary["load"]["steady_e2e_s"]["mean"]
        off_switch = summary["off"]["estimated_switch_overhead_s"]["mean"]
        load_switch = summary["load"]["estimated_switch_overhead_s"]["mean"]
        summary["comparison"] = {
            "steady_e2e_speedup_off_over_load": (
                off_e2e / load_e2e if load_e2e > 0 else None
            ),
            "steady_e2e_latency_reduction_percent": (
                (off_e2e - load_e2e) / off_e2e * 100 if off_e2e > 0 else None
            ),
            "switch_overhead_speedup_off_over_load": (
                off_switch / load_switch if load_switch > 0 else None
            ),
            "switch_overhead_reduction_percent": (
                (off_switch - load_switch) / off_switch * 100
                if off_switch > 0
                else None
            ),
        }
    return summary


def _write_results(
    output: Path,
    args: argparse.Namespace,
    measurements: list[Measurement],
    deployments: list[Deployment],
    logs: dict[str, Path],
    summary: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "switch_max_tokens": args.switch_max_tokens,
        "requests_per_mode": args.requests,
        "switch_cycles_per_mode": args.switch_cycles,
        "logs": {mode: str(path) for mode, path in logs.items()},
        "summary": summary,
        "measurements": [
            {**asdict(row), "output_tokens_per_second": row.output_tokens_per_second}
            for row in measurements
        ],
        "deployments": [asdict(row) for row in deployments],
    }
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(asdict(measurements[0]).keys()) + ["output_tokens_per_second"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in measurements:
            writer.writerow(
                {
                    **asdict(row),
                    "output_tokens_per_second": row.output_tokens_per_second,
                }
            )
    print(f"\nDetailed JSON: {output}")
    print(f"Detailed CSV:  {csv_path}")


def _print_summary(modes: list[str], summary: dict[str, Any]) -> None:
    print("\n========== CUDA Graph latency comparison ==========")
    print(
        "mode   steady E2E mean   steady TTFT   output tok/s   "
        "switch req   active control   switch overhead"
    )
    for mode in modes:
        item = summary[mode]
        print(
            f"{mode:<6} "
            f"{item['steady_e2e_s']['mean']:>15.3f}s "
            f"{item['steady_ttft_s']['mean']:>12.3f}s "
            f"{item['steady_output_tokens_per_second']['mean']:>13.2f} "
            f"{item['switch_request_e2e_s']['mean']:>11.3f}s "
            f"{item['switch_control_e2e_s']['mean']:>15.3f}s "
            f"{item['estimated_switch_overhead_s']['mean']:>15.3f}s"
        )
    comparison = summary.get("comparison")
    if comparison:
        print(
            "\nLOAD vs OFF steady E2E: "
            f"{comparison['steady_e2e_speedup_off_over_load']:.3f}x, "
            f"latency reduction {comparison['steady_e2e_latency_reduction_percent']:.2f}%"
        )
        switch_speedup = comparison["switch_overhead_speedup_off_over_load"]
        switch_reduction = comparison["switch_overhead_reduction_percent"]
        if switch_speedup is not None and switch_reduction is not None:
            print(
                "LOAD vs OFF estimated switch overhead: "
                f"{switch_speedup:.3f}x, reduction {switch_reduction:.2f}%"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-a", default=MODEL_A)
    parser.add_argument("--model-b", default=MODEL_B)
    parser.add_argument(
        "--modes", nargs="+", choices=("off", "load"), default=["off", "load"]
    )
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--switch-cycles", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--switch-max-tokens", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--hook", type=Path, default=DEFAULT_HOOK)
    parser.add_argument(
        "--cuda-compat",
        type=Path,
        default=Path.home() / ".local" / "cuda-12.9" / "compat",
    )
    args = parser.parse_args()
    if args.requests <= 0 or args.switch_cycles <= 0:
        parser.error("--requests and --switch-cycles must be positive")
    if args.max_tokens <= 0 or args.switch_max_tokens <= 0:
        parser.error("token counts must be positive")
    return args


def main() -> int:
    args = _parse_args()
    args.config = args.config.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.hook = args.hook.expanduser().resolve()
    args.cuda_compat = args.cuda_compat.expanduser().resolve()
    if not args.hook.is_file():
        raise RuntimeError(f"Foundry hook does not exist: {args.hook}")
    if not args.cuda_compat.is_dir():
        raise RuntimeError(
            f"CUDA compatibility directory does not exist: {args.cuda_compat}"
        )
    if _port_is_open(args.host, args.port):
        raise RuntimeError(
            f"Port {args.host}:{args.port} is already in use; stop the existing server first"
        )

    source = _load_config(args.config, (args.model_a, args.model_b))
    models = {
        item["name"]: item
        for item in source["models"]
        if item["name"] in (args.model_a, args.model_b)
    }
    print("Request payload:")
    print(
        json.dumps(
            {
                "model": args.model_a,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    measurements: list[Measurement] = []
    deployments: list[Deployment] = []
    logs: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="aegaeon-cuda-graph-bench-") as temp_dir:
        for mode in args.modes:
            config_path = Path(temp_dir) / f"{mode}.yaml"
            _write_mode_config(source, mode, config_path)
            mode_measurements, mode_deployments, log_path = _run_mode(
                args,
                mode,
                config_path,
                models,
            )
            measurements.extend(mode_measurements)
            deployments.extend(mode_deployments)
            logs[mode] = log_path
            deadline = time.monotonic() + 30
            while _port_is_open(args.host, args.port) and time.monotonic() < deadline:
                time.sleep(1)
            if _port_is_open(args.host, args.port):
                raise RuntimeError(
                    f"Port {args.host}:{args.port} did not close after {mode}"
                )

    summary = _summarize(args.modes, measurements, deployments)
    _print_summary(args.modes, summary)
    _write_results(args.output, args, measurements, deployments, logs, summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
