import os
import argparse
import json
import math
import multiprocessing
import subprocess
import sys
import tempfile
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed

DIR = os.path.dirname(os.path.abspath(__file__))

CASES = [
    (ilen, bs)
    for ilen in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
    if ilen * bs <= 8192 and (ilen + 10) * bs < 10240
]


def load_models_from_config(config_path: str) -> list:
    from aegaeon.model_registry import ModelSpecRegistry

    return list(ModelSpecRegistry.from_yaml(config_path))


def _validate_tp(tp: int) -> None:
    if tp != 1:
        raise ValueError("Profiling currently supports only --tp 1")


def _visible_devices() -> list[str]:
    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No CUDA GPUs are visible; profiling requires a CUDA GPU")
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return [str(i) for i in range(count)]
    devices = [device.strip() for device in configured.split(",")]
    if len(devices) < count or any(not device for device in devices[:count]):
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the visible CUDA GPUs")
    return devices[:count]


def build_workloads(models: list, tp: int = 1, devices: list[str] = None) -> list:
    _validate_tp(tp)
    devices = _visible_devices() if devices is None else devices
    if not devices:
        raise RuntimeError("No CUDA GPUs are visible; profiling requires a CUDA GPU")
    workloads = []
    for i, spec in enumerate(models):
        device = devices[i % len(devices)]
        workloads.append((device, spec))
    return workloads


def profile_once(cuda_visible_devices: str, spec, tp: int):
    _validate_tp(tp)
    import torch

    script_path = os.path.join(DIR, "..", "benchmark", "benchmark_latency.py")
    devices = _visible_devices()
    if cuda_visible_devices not in devices:
        raise ValueError(f"GPU {cuda_visible_devices!r} is outside CUDA_VISIBLE_DEVICES")
    device_index = devices.index(cuda_visible_devices)
    device_name = torch.cuda.get_device_name(device_index).removeprefix("NVIDIA ").replace(" ", "_")
    model_path = spec.path()

    cmd = [
        sys.executable, script_path,
        "--model", model_path, "--enforce-eager",
        "--dtype", "bfloat16", "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.90", "--max-model-len", "8192",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    log_dir = os.path.join(DIR, "..", "logs", f"profile-{spec.name}-{tp}x{device_name}")
    profile_root = os.path.expanduser(os.environ.get("AEGAEON_PROFILE_ROOT") or DIR)
    out_dir = os.path.abspath(os.path.join(profile_root, spec.profile_key(), device_name))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    for input_len, batch_size in CASES:
        output_json = os.path.join(out_dir, f"i{input_len}b{batch_size}.json")
        output_log = os.path.join(log_dir, f"i{input_len}b{batch_size}.log")

        print(f"------------- {spec.name} i{input_len}b{batch_size} -------------")
        # Publish only a newly generated result; a stale profile must not mask
        # a benchmark that exits successfully without writing its output.
        with tempfile.TemporaryDirectory(prefix=".profile-", dir=out_dir) as temp_dir:
            temporary_json = os.path.join(temp_dir, os.path.basename(output_json))
            full_cmd = cmd + [
                "--input-len", str(input_len), "--batch-size", str(batch_size),
                "--output-json", temporary_json,
            ]
            print(subprocess.list2cmdline(full_cmd))
            try:
                with open(output_log, "w", encoding="utf-8") as log:
                    subprocess.run(
                        full_cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True,
                    )
            except (subprocess.CalledProcessError, OSError) as error:
                raise RuntimeError(
                    f"Profiling failed for {spec.name} i{input_len}b{batch_size}: "
                    f"{error}. See {output_log}"
                ) from error

            try:
                with open(temporary_json, "r", encoding="utf-8") as result:
                    data = json.load(result)
                for key in ("prefill_latencies", "decode_latencies"):
                    values = data.get(key) if isinstance(data, dict) else None
                    if not isinstance(values, list) or len(values) < 3 or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 for value in values
                    ):
                        raise ValueError(f"Invalid {key}: expected at least 3 finite, nonnegative samples")
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"Profiling produced missing or invalid output for {spec.name} "
                    f"i{input_len}b{batch_size}: {error}. See {output_log}"
                ) from error
            os.replace(temporary_json, output_json)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to YAML model config file.",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size (only 1 is supported)."
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy URL for model downloads (e.g. http://localhost:9090).",
    )
    args = parser.parse_args(argv)

    if args.proxy:
        os.environ["AEGAEON_MODEL_DOWNLOAD_PROXY"] = args.proxy

    try:
        _validate_tp(args.tp)
        devices = _visible_devices()
        models = load_models_from_config(args.model_config)
        if not models:
            raise ValueError("The model config contains no models to profile")
        workloads = build_workloads(models, tp=args.tp, devices=devices)
        num_workers = min(len(devices), len(workloads))

        print(
            f"Found {len(models)} models, {len(workloads)} workloads "
            f"(tp={args.tp}, gpus={len(devices)})"
        )
        for device, spec in workloads:
            print(f"  device {device}: {spec.name}")

        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            it = iter(workloads)
            while batch := list(islice(it, num_workers)):
                futures = [
                    executor.submit(profile_once, device, spec, args.tp)
                    for device, spec in batch
                ]
                for fut in as_completed(futures):
                    fut.result()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"Profiling failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
