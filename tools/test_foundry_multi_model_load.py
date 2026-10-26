#!/usr/bin/env python3
"""End-to-end smoke test for Aegaeon's multi-model Foundry LOAD path.

The test starts an empty Aegaeon service, verifies that the decode Worker loads
all configured model archives exactly once, and forces graph switches A -> B -> A.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmark" / "foundry-vmm-qwen-family-load.yaml"
DEFAULT_HOOK = ROOT / "foundry" / "python" / "foundry" / "libcuda_hook.so"
DEFAULT_LOG = ROOT / "logs" / "foundry-multi-model-smoke.log"
MODEL_A = "Qwen/Qwen3-4B"
MODEL_B = "Qwen/Qwen3.5-9B"


def _archive_key(model_name: str) -> str:
    return quote(model_name, safe="").replace(".", "%2E")


def _load_and_validate_config(
    config_path: Path,
    model_names: list[str],
) -> dict:
    if not config_path.is_file():
        raise RuntimeError(f"Config does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config is not a YAML mapping: {config_path}")

    server = raw.get("server", {})
    graph = server.get("cuda_graph", {})
    if graph.get("mode") != "load":
        raise RuntimeError("server.cuda_graph.mode must be 'load'")
    if server.get("startup_models") != []:
        raise RuntimeError(
            "server.startup_models must be [] so the smoke test starts empty"
        )
    if int(server.get("num_decode_engines", 0)) != 1:
        raise RuntimeError("Smoke test requires exactly one decode engine")

    configured_models = {
        entry.get("name"): entry
        for entry in raw.get("models", [])
        if isinstance(entry, dict)
    }
    missing_models = [name for name in model_names if name not in configured_models]
    if missing_models:
        raise RuntimeError(f"Models missing from config: {missing_models}")

    archive_root = Path(str(graph.get("archive_dir", ""))).expanduser().resolve()
    if not archive_root.is_dir():
        raise RuntimeError(f"Archive catalog does not exist: {archive_root}")

    batch_sizes = sorted(
        {int(value) for value in graph.get("batch_sizes", [])}, reverse=True
    )
    seq_len_buckets = [int(value) for value in graph.get("seq_len_buckets", [])]
    if not batch_sizes or len(seq_len_buckets) != 1:
        raise RuntimeError(
            "Smoke test requires non-empty batch_sizes and one seq_len_buckets value"
        )

    tensor_parallel_size = int(server.get("tensor_parallel_size", 1))
    required_static_files = (
        "fatbin_image_packed.img",
        "fatbin_entrypoint_packed.txt",
        "vmm_layout.json",
        "graph_manifest.json",
    )
    missing_paths: list[Path] = []
    for model_name in model_names:
        for rank in range(tensor_parallel_size):
            rank_dir = (
                archive_root / "models" / _archive_key(model_name) / f"rank_{rank}"
            )
            for filename in required_static_files:
                path = rank_dir / filename
                if not path.is_file():
                    missing_paths.append(path)
            max_seq_len = seq_len_buckets[0]
            for graph_index, batch_size in enumerate(batch_sizes):
                path = rank_dir / (
                    f"graph_{graph_index}_decode_bs{batch_size}_"
                    f"seq{max_seq_len}.json"
                )
                if not path.is_file():
                    missing_paths.append(path)

    if missing_paths:
        details = "\n".join(f"  - {path}" for path in missing_paths[:10])
        suffix = "\n  - ..." if len(missing_paths) > 10 else ""
        raise RuntimeError(
            f"Archive catalog is incomplete ({len(missing_paths)} missing files):\n"
            f"{details}{suffix}"
        )

    print(
        f"Preflight OK: {len(model_names)} model archives, "
        f"TP={tensor_parallel_size}, {len(batch_sizes)} graph shapes per rank"
    )
    return raw


def _http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 30,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


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


def _deploy(
    base_url: str,
    model_name: str,
    model_path: str,
    max_model_len: int | None,
    timeout: float,
) -> None:
    print(f"Deploying {model_name} ...")
    payload: dict[str, object] = {
        "model": model_name,
        "model_path": model_path,
        "replica_count": 1,
    }
    if max_model_len is not None:
        payload["max_model_len"] = max_model_len
    response = _http_json(
        "POST",
        f"{base_url}/v1/models/deploy",
        payload,
        timeout=timeout,
    )
    status = response.get("status")
    if status not in (None, "ok"):
        raise RuntimeError(f"Unexpected deploy response for {model_name}: {response}")


def _generate(base_url: str, model_name: str, timeout: float) -> str:
    print(f"Running completion with {model_name} ...")
    response = _http_json(
        "POST",
        f"{base_url}/v1/completions",
        {
            "model": model_name,
            "prompt": "The capital of France is",
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "return_token_ids": True,
        },
        timeout=timeout,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"No completion choices for {model_name}: {response}")
    token_ids = choices[0].get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise RuntimeError(f"No generated token IDs for {model_name}: {response}")
    text = str(choices[0].get("text", ""))
    print(f"  token_ids={token_ids}, text={text!r}")
    return text


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


def _validate_log(log_path: Path) -> tuple[int, int]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    load_matches = re.findall(
        r"\[HOOK\] Loaded (\d+) unique binaries from (\d+) packed archives",
        log_text,
    )
    if not load_matches:
        raise RuntimeError("Did not find Foundry multi-archive load marker in the log")
    wrong_counts = [count for _, count in load_matches if int(count) != 2]
    if wrong_counts:
        raise RuntimeError(f"Unexpected archive counts in Foundry log: {wrong_counts}")

    graph_load_count = len(
        re.findall(r"loaded \d+ Foundry decode graph shapes", log_text)
    )
    if graph_load_count < 3:
        raise RuntimeError(
            "Expected at least three active-model graph loads for A -> B -> A, "
            f"found {graph_load_count}"
        )
    if "Foundry CUDA binaries are already loaded" in log_text:
        raise RuntimeError(
            "A switch attempted to invoke the one-shot binary loader again"
        )

    unique_binary_count = int(load_matches[0][0])
    return unique_binary_count, graph_load_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-a", default=MODEL_A)
    parser.add_argument("--model-b", default=MODEL_B)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray address passed to Aegaeon (default: auto)",
    )
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--hook", type=Path, default=DEFAULT_HOOK)
    parser.add_argument(
        "--cuda-compat",
        type=Path,
        default=Path.home() / ".local" / "cuda-12.9" / "compat",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config and archive contents without starting Aegaeon",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    model_names = [args.model_a, args.model_b]
    raw_config = _load_and_validate_config(config_path, model_names)
    if args.preflight_only:
        return 0

    hook_path = args.hook.expanduser().resolve()
    cuda_compat = args.cuda_compat.expanduser().resolve()
    if not hook_path.is_file():
        raise RuntimeError(f"Foundry hook does not exist: {hook_path}")
    if not cuda_compat.is_dir():
        raise RuntimeError(
            f"CUDA compatibility directory does not exist: {cuda_compat}"
        )

    base_url = f"http://{args.host}:{args.port}"
    try:
        _http_json("GET", f"{base_url}/health", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"Another Aegaeon server is already listening at {base_url}")

    log_path = args.log.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AEGAEON_FOUNDRY_HOOK_PATH"] = str(hook_path)
    env["AEGAEON_CUDA_COMPAT_PATH"] = str(cuda_compat)
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
    print("Starting:", " ".join(command))
    print(f"Server log: {log_path}")

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
            print("Aegaeon is healthy; starting A -> B -> A sequence")
            outputs: list[tuple[str, str]] = []
            configured_models = {entry["name"]: entry for entry in raw_config["models"]}
            for model_name in (args.model_a, args.model_b, args.model_a):
                model_config = configured_models[model_name]
                _deploy(
                    base_url,
                    model_name,
                    str(model_config["path"]),
                    model_config.get("max_model_len"),
                    args.request_timeout,
                )
                outputs.append(
                    (
                        model_name,
                        _generate(base_url, model_name, args.request_timeout),
                    )
                )
        except BaseException as exc:
            failure = exc
        finally:
            _stop_server(process)

    if failure is not None:
        print("\n===== server log tail =====", file=sys.stderr)
        print(_tail(log_path), file=sys.stderr)
        raise failure

    try:
        unique_binary_count, graph_load_count = _validate_log(log_path)
    except Exception:
        print("\n===== server log tail =====", file=sys.stderr)
        print(_tail(log_path), file=sys.stderr)
        raise

    print("\nPASS: multi-model Foundry LOAD smoke test completed")
    print(f"  unique CUDA binaries: {unique_binary_count}")
    print("  packed archives per decode Worker: 2")
    print(f"  active-model graph loads observed: {graph_load_count}")
    print(f"  full log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
