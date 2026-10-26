#!/usr/bin/env python3
"""GPU integration test: eager -> SAVE -> fresh-process LOAD for three models.

Uses a new run directory and never overwrites an existing archive. Compares
complete greedy token IDs, not just successful HTTP responses. The reference
is Aegaeon OFF (not an independent vLLM correctness oracle). Default scope is
TP=1, batch=1, text-only, one sequence bucket. Also checks Gemma SWA crossing
and a multi-model LOAD A -> B -> C -> A switch sequence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
from urllib.parse import quote

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_foundry_multi_model_load import (  # noqa: E402
    _deploy,
    _stop_server,
    _tail,
    _wait_for_health,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "What are the differences between processes and threads"
MODEL_NAMES = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3.5-9B",
    "LLM-Research/gemma-3-4b-it",
)


def parse_packed_hashes(text):
    """Read Foundry's hash/options/count/entrypoints/--- metadata records."""
    lines, cursor, hashes = text.splitlines(), 0, set()
    while cursor < len(lines):
        if cursor + 3 > len(lines) or not re.fullmatch(
            r"[0-9a-fA-F]{1,16}", lines[cursor]
        ):
            raise AssertionError("Invalid packed binary metadata/hash")
        count = int(lines[cursor + 2])
        end = cursor + 3 + count
        if count < 0 or end >= len(lines) or lines[end] != "---":
            raise AssertionError("Truncated packed metadata or wrong entrypoint count")
        hashes.add(int(lines[cursor], 16))
        cursor = end + 1
    if not hashes:
        raise AssertionError("Packed archive has no binary hashes")
    return hashes


def model_rank_dir(args, model):
    key = quote(model["name"], safe="").replace(".", "%2E")
    return args.archive / "models" / key / "rank_0"


def compare_tokens(reference, actual):
    for key in ("prompt_token_ids", "token_ids"):
        expected, received = reference[key], actual[key]
        if expected != received:
            first = next(
                (
                    i
                    for i, pair in enumerate(zip(expected, received))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(received)),
            )
            raise AssertionError(
                f"{key} mismatch at index {first}: "
                f"OFF length={len(expected)}, tested length={len(received)}, "
                f"OFF slice={expected[max(0, first-3):first+4]}, "
                f"tested slice={received[max(0, first-3):first+4]}"
            )


def parse_sse(lines):
    """Reject incomplete/error streams and retain exact server token IDs."""
    result = {"text": "", "prompt_token_ids": None, "token_ids": None}
    done = False
    final = False
    usage = None
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        item = json.loads(data)
        if "error" in item:
            raise RuntimeError(f"SSE error: {item['error']}")
        if "prompt_token_ids" in item:
            result["prompt_token_ids"] = item["prompt_token_ids"]
        if item.get("usage"):
            usage = item["usage"]
        for choice in item.get("choices", []):
            result["text"] += choice.get("delta", {}).get("content", "") or ""
            if choice.get("finish_reason") is not None:
                final = True
                result["finish_reason"] = choice["finish_reason"]
                result["token_ids"] = choice.get("token_ids")
    if not done or not final:
        raise RuntimeError("Incomplete SSE: final generation chunk or [DONE] missing")
    if not isinstance(result["prompt_token_ids"], list):
        raise RuntimeError("SSE is missing prompt_token_ids")
    if not isinstance(result["token_ids"], list) or len(result["token_ids"]) < 1:
        raise RuntimeError("SSE has no generated token IDs")
    if not usage or usage.get("completion_tokens") != len(result["token_ids"]):
        raise RuntimeError("SSE usage/token-ID count mismatch")
    result["usage"] = usage
    return result


def request_chat(args, model, case, destination):
    payload = {
        "model": model["name"],
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    if case.get("ignore_eos"):
        payload["ignore_eos"] = True
    start = time.perf_counter()
    request = Request(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    # Persist chunks as they arrive, including partial streams on failure.
    with destination.open("wb") as file:
        with opener.open(request, timeout=args.request_timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")

            def recording_lines():
                for line in response:
                    file.write(line)
                    yield line

            result = parse_sse(recording_lines())
    result["e2e_s"] = time.perf_counter() - start
    result["case"] = case
    sequence_end = len(result["prompt_token_ids"]) + len(result["token_ids"])
    if sequence_end > args.seq_len:
        raise AssertionError(
            f"Request sequence {sequence_end} exceeds Graph bucket {args.seq_len}; "
            "this would allow eager fallback"
        )
    if len(result["token_ids"]) < 2:
        raise AssertionError("Need at least two tokens to exercise decode Graph replay")
    return result


@contextmanager
def server(args, template, mode, models, label):
    config = copy.deepcopy(template)
    config["models"] = models
    config["server"]["startup_models"] = []
    graph = config["server"]["cuda_graph"]
    graph.update(
        mode=mode,
        archive_dir=str(args.archive),
        batch_sizes=[1],
        seq_len_buckets=[args.seq_len],
        strict=True,
    )
    config_path = args.run_dir / f"{label}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = args.run_dir / f"{label}.log"
    env = os.environ.copy()
    # Repeated A -> B -> C -> A load markers must not be hidden by Ray.
    env["RAY_DEDUP_LOGS"] = "0"
    env["AEGAEON_FOUNDRY_HOOK_PATH"] = str(args.hook)
    env["AEGAEON_CUDA_COMPAT_PATH"] = str(args.cuda_compat)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "foundry/python"), env.get("PYTHONPATH", "")]
    )
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
    print(f"Starting {label}; log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_health(
                process, f"http://{args.host}:{args.port}", args.startup_timeout
            )
            yield log_path
        except BaseException:
            print(_tail(log_path, 60), file=sys.stderr)
            raise
        finally:
            _stop_server(process)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in ("[HOOK] ERROR", "CUDA error", "Traceback"):
        if marker in text:
            raise AssertionError(f"{label} log contains {marker}; see {log_path}")
    if mode == "save" and "captured 1 Foundry decode graphs" not in text:
        raise AssertionError(f"No SAVE capture marker in {log_path}")
    if mode in ("save", "load"):
        replays = re.findall(r"Foundry decode replay active: model=([^,\r\n]+)", text)
        expected_replays = [model["name"] for model in models]
        if len(models) > 1:
            expected_replays = getattr(
                args, "switch_order", expected_replays + [models[0]["name"]]
            )
        if replays != expected_replays:
            raise AssertionError(
                f"Missing/wrong actual Graph replay markers: {replays}, expected={expected_replays}"
            )
    if mode == "load":
        union = re.findall(
            r"Loaded (\d+) unique binaries from (\d+) packed archives", text
        )
        if len(union) != 1 or int(union[0][1]) != len(models) or int(union[0][0]) < 1:
            raise AssertionError(f"Wrong one-shot archive-union load: {union}")
        if "captured 1 Foundry decode graphs" in text:
            raise AssertionError(
                "LOAD unexpectedly captured new Graphs instead of restoring"
            )
        expected_hashes = set().union(
            *(
                parse_packed_hashes(
                    (
                        model_rank_dir(args, model) / "fatbin_entrypoint_packed.txt"
                    ).read_text()
                )
                for model in models
            )
        )
        if int(union[0][0]) != len(expected_hashes):
            raise AssertionError(
                f"Binary hash union mismatch: Worker={union[0][0]}, archive union={len(expected_hashes)}"
            )
        required_loads = len(expected_replays)
        count = text.count("loaded 1 Foundry decode graph shapes")
        if count < required_loads:
            raise AssertionError(
                f"Expected {required_loads} Graph loads, found {count}"
            )
    if mode == "off" and "Foundry decode graph" in text:
        raise AssertionError("OFF reference unexpectedly used Foundry Graphs")


def deploy(args, model):
    _deploy(
        f"http://{args.host}:{args.port}",
        model["name"],
        model["path"],
        model["max_model_len"],
        args.request_timeout,
    )


def validate_archive(args, model):
    directory = model_rank_dir(args, model)
    for name in (
        "fatbin_image_packed.img",
        "fatbin_entrypoint_packed.txt",
        "vmm_layout.json",
        "graph_manifest.json",
        f"graph_0_decode_bs1_seq{args.seq_len}.json",
    ):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing/empty archive: {path}")
    metadata = json.loads((directory / "vmm_layout.json").read_text())
    if metadata.get("final_alloc_offset", 0) < metadata["model_layout_end_offset"]:
        raise AssertionError("Invalid archive final allocation offset")
    parse_packed_hashes((directory / "fatbin_entrypoint_packed.txt").read_text())


def make_boundary_case(args, model, window):
    """Use the server's actual chat template to calibrate just below SWA W."""
    target = window - 64
    repeats = target
    for attempt in range(6):
        prompt = (
            "Read the following words and explain processes and threads:\n"
            + " test" * repeats
        )
        probe = {"prompt": prompt, "max_tokens": 2, "ignore_eos": True}
        result = request_chat(
            args, model, probe, args.run_dir / f"gemma-calibrate-{attempt}.sse"
        )
        length = len(result["prompt_token_ids"])
        if window - 80 <= length <= window - 48:
            return {
                "prompt": prompt,
                "max_tokens": 128,
                "ignore_eos": True,
                "sliding_window": window,
            }
        repeats = max(1, round(repeats * target / max(1, length)))
    raise AssertionError(
        "Could not calibrate Gemma prompt below sliding-window boundary"
    )


def write_report(args, report):
    (args.run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def record_comparison(report, checks, label, reference, actual):
    """A numerical mismatch must not skip the archive's fresh-process LOAD."""
    try:
        compare_tokens(reference, actual)
    except AssertionError as exc:
        checks[label] = {"status": "fail", "error": str(exc)}
        report["errors"].append(f"{label}: {exc}")
        print(f"FAIL {label}: {exc}", file=sys.stderr, flush=True)
    else:
        checks[label] = {"status": "pass", "tokens": len(actual["token_ids"])}
        print(f"PASS {label}: {len(actual['token_ids'])} exact tokens", flush=True)


def summarize_checks(report):
    models = report["models"]
    roundtrip = len(models) == 3 and all(
        entry.get("roundtrip_completed")
        and any("/SAVE-vs-LOAD/" in label for label in entry["checks"])
        and all(
            check["status"] == "pass"
            for label, check in entry["checks"].items()
            if "/SAVE-vs-LOAD/" in label
        )
        for entry in models.values()
    )
    switches = report.get("multi_model_load", [])
    switch_ok = (
        report.get("multi_model_load_runtime_status") == "pass"
        and isinstance(switches, list)
        and len(switches) == 4
        and all(
            any(label.endswith("/save-vs-LOAD") for label in item["checks"])
            and all(
                check["status"] == "pass"
                for label, check in item["checks"].items()
                if label.endswith("/save-vs-LOAD")
            )
            for item in switches
        )
    )
    report["save_load_roundtrip_status"] = "pass" if roundtrip else "fail"
    report["switch_save_equivalence_status"] = "pass" if switch_ok else "fail"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmark/foundry-vmm-qwen-family-load.yaml",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--gemma-path",
        type=Path,
        default=Path("/srv/models/LLM-Research/gemma-3-4b-it"),
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--switch-tokens", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--hook", type=Path, default=ROOT / "foundry/python/foundry/libcuda_hook.so"
    )
    parser.add_argument(
        "--cuda-compat", type=Path, default=Path.home() / ".local/cuda-12.9/compat"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 2 or not 2 <= args.switch_tokens <= args.max_tokens:
        parser.error("Need max-tokens >= switch-tokens >= 2")
    template = yaml.safe_load(args.config.expanduser().read_text())
    if int(template["server"].get("tensor_parallel_size", 1)) != 1:
        parser.error("This test currently requires TP=1")
    if template["server"].get("num_decode_engines") != 1:
        parser.error("This test requires one decode engine for deterministic switches")
    models = [
        copy.deepcopy(next(m for m in template["models"] if m["name"] == name))
        for name in MODEL_NAMES[:2]
    ]
    models.append(
        {
            "name": MODEL_NAMES[2],
            "path": str(args.gemma_path),
            "params": 4.3,
            "profile": "gemma_3_4b_it",
            "max_model_len": args.seq_len,
        }
    )
    for model in models:
        model["max_model_len"] = args.seq_len
        if not Path(model["path"]).is_dir():
            raise RuntimeError(f"Model path missing: {model['path']}")
        if not (ROOT / "profiles" / model["profile"]).is_dir():
            raise RuntimeError(f"Model profile missing: {model['profile']}")
    hf = json.loads((args.gemma_path / "config.json").read_text())
    window = int(hf["text_config"]["sliding_window"])
    if args.seq_len <= max(args.max_tokens + 128, window + 128):
        parser.error("seq-len must cover the output budget and Gemma SWA test")
    if not args.hook.is_file() or not args.cuda_compat.is_dir():
        raise RuntimeError("Foundry hook or CUDA compatibility directory is missing")
    print(f"Preflight OK: {MODEL_NAMES}; Gemma sliding_window={window}", flush=True)
    if args.preflight_only:
        return 0
    try:
        with socket.create_connection((args.host, args.port), timeout=1):
            raise RuntimeError(
                f"Port {args.port} is already in use; no existing server will be stopped"
            )
    except OSError:
        pass
    args.run_dir = (
        (
            args.run_dir
            or ROOT / "work" / f"foundry-roundtrip-{datetime.now():%Y%m%d-%H%M%S}"
        )
        .expanduser()
        .resolve()
    )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    args.archive = args.run_dir / "archive"
    # Health/deploy helpers use urllib's default proxy handler.
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    report = {
        "scope": "text-only TP1 batch1",
        "status": "running",
        "models": {},
        "archive_root": str(args.archive),
        "errors": [],
    }
    write_report(args, report)
    references = {}
    for index, model in enumerate(models):
        name = model["name"]
        initial_errors = len(report["errors"])
        entry = report["models"][name] = {
            "status": "running",
            "phases": {},
            "checks": {},
            "roundtrip_completed": False,
        }
        try:
            cases = {"chat": {"prompt": PROMPT, "max_tokens": args.max_tokens}}
            with server(args, template, "off", [model], f"{index}-off"):
                deploy(args, model)
                if index == 2:
                    cases["swa-crossing"] = make_boundary_case(args, model, window)
                reference = {}
                for case_name, case in cases.items():
                    result = request_chat(
                        args, model, case, args.run_dir / f"{index}-off-{case_name}.sse"
                    )
                    if case_name == "swa-crossing":
                        length = len(result["prompt_token_ids"])
                        if not length < window < length + len(result["token_ids"]) - 32:
                            raise AssertionError(
                                "Gemma test did not cross SWA and page-retirement boundary"
                            )
                        if len(result["token_ids"]) != 128:
                            raise AssertionError(
                                "Forced Gemma boundary decode did not generate 128 tokens"
                            )
                    reference[case_name] = result
                entry["phases"]["off"] = reference
                repeated = entry["phases"]["off_repeat"] = {}
                for case_name, case in cases.items():
                    result = request_chat(
                        args,
                        model,
                        case,
                        args.run_dir / f"{index}-off-repeat-{case_name}.sse",
                    )
                    repeated[case_name] = result
                    record_comparison(
                        report,
                        entry["checks"],
                        f"{name}/OFF-repeat/{case_name}",
                        reference[case_name],
                        result,
                    )
            write_report(args, report)
            for mode in ("save", "load"):
                phase = entry["phases"][mode] = {}
                with server(args, template, mode, [model], f"{index}-{mode}"):
                    deploy(args, model)
                    for case_name, case in cases.items():
                        result = request_chat(
                            args,
                            model,
                            case,
                            args.run_dir / f"{index}-{mode}-{case_name}.sse",
                        )
                        phase[case_name] = result
                        record_comparison(
                            report,
                            entry["checks"],
                            f"{name}/OFF-vs-{mode}/{case_name}",
                            reference[case_name],
                            result,
                        )
                        if mode == "load":
                            record_comparison(
                                report,
                                entry["checks"],
                                f"{name}/SAVE-vs-LOAD/{case_name}",
                                entry["phases"]["save"][case_name],
                                result,
                            )
                validate_archive(args, model)
                write_report(args, report)
            entry["roundtrip_completed"] = True
            references[name] = {
                "off": reference["chat"],
                "save": entry["phases"]["save"]["chat"],
            }
            entry["status"] = (
                "pass" if len(report["errors"]) == initial_errors else "fail"
            )
        except KeyboardInterrupt:
            entry["status"] = report["status"] = "interrupted"
            write_report(args, report)
            raise
        except Exception as exc:
            entry["status"] = "fail"
            entry["error"] = str(exc)
            report["errors"].append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}", file=sys.stderr, flush=True)
        write_report(args, report)
    if all(entry["roundtrip_completed"] for entry in report["models"].values()):
        try:
            switched = report["multi_model_load"] = []
            report["multi_model_load_runtime_status"] = "running"
            with server(args, template, "load", models, "multi-model-load"):
                for request_index in (0, 1, 2, 0):
                    model = models[request_index]
                    deploy(args, model)
                    case = {"prompt": PROMPT, "max_tokens": args.switch_tokens}
                    result = request_chat(
                        args, model, case, args.run_dir / f"switch-{len(switched)}.sse"
                    )
                    checks = {}
                    for reference_mode in ("off", "save"):
                        reference = copy.deepcopy(
                            references[model["name"]][reference_mode]
                        )
                        reference["token_ids"] = reference["token_ids"][
                            : args.switch_tokens
                        ]
                        record_comparison(
                            report,
                            checks,
                            f"switch-{len(switched)}/{model['name']}/{reference_mode}-vs-LOAD",
                            reference,
                            result,
                        )
                    switched.append(
                        {"model": model["name"], "checks": checks, **result}
                    )
            report["multi_model_load_runtime_status"] = "pass"
        except KeyboardInterrupt:
            report["status"] = "interrupted"
            write_report(args, report)
            raise
        except Exception as exc:
            report["multi_model_load_runtime_status"] = "fail"
            report["errors"].append(f"Multi-model LOAD: {exc}")
    else:
        report["multi_model_load"] = (
            "skipped because a single-model roundtrip could not complete"
        )
    report["status"] = "fail" if report["errors"] else "pass"
    summarize_checks(report)
    write_report(args, report)
    print(f"SAVE vs fresh LOAD: {report['save_load_roundtrip_status'].upper()}")
    print(
        f"Multi-model switch vs SAVE: {report['switch_save_equivalence_status'].upper()}"
    )
    print(
        f"\n{report['status'].upper()}: report={args.run_dir / 'report.json'}",
        flush=True,
    )
    print(
        f"Configs, complete SSE streams, logs and archives retained in {args.run_dir}"
    )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
