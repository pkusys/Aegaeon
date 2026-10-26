#!/usr/bin/env python3
"""No-warmup model-switch request latency: OFF versus existing Graph LOAD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
from urllib.request import ProxyHandler, Request, build_opener

import yaml

from test_foundry_three_model_roundtrip import (
    MODEL_NAMES,
    PROMPT,
    ROOT,
    compare_tokens,
    deploy,
    parse_sse,
    server,
    validate_archive,
)

DEFAULT_SOURCE = ROOT / "work/foundry-three-model-fixed-v2-20260917"
ORDER = (MODEL_NAMES[1], MODEL_NAMES[0], MODEL_NAMES[2])


def read_timed_stream(lines, start, clock=time.perf_counter, raw=None):
    """Timestamp received content, completion and DONE independently.

    The API does not stream individual token IDs. Content chunks can contain
    multiple tokens, and EOS may emit no text. Therefore last_content_s is
    the last visible output; generation_finish_s is the received confirmation
    that ALL output tokens finished. Neither is a server GPU timestamp.
    """
    if raw is None:
        raw = []
    first = last = finished = done = None
    for line in lines:
        received = clock() - start
        raw.append(line)
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data:"):
            continue
        data = decoded[5:].strip()
        if data == "[DONE]":
            done = received
            continue
        event = json.loads(data)
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                if first is None:
                    first = received
                last = received
            if choice.get("finish_reason") is not None:
                finished = received
    result = parse_sse(raw)  # Reject errors/incomplete streams/invalid token IDs.
    if first is None or finished is None or done is None:
        raise RuntimeError("Missing visible output, finish chunk or DONE timestamp")
    result.update(
        ttft_s=first,
        last_content_s=last,
        generation_finish_s=finished,
        stream_done_s=done,
        output_tokens=len(result["token_ids"]),
        prompt_tokens=len(result["prompt_token_ids"]),
    )
    return result, raw


def timed_request(args, model, destination):
    payload = {
        "model": model["name"],
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    request = Request(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))  # Equivalent to curl --noproxy.
    raw = []
    start = time.perf_counter()  # Includes HTTP connection and response latency.
    try:
        with opener.open(request, timeout=args.request_timeout) as response:
            result, raw = read_timed_stream(response, start, raw=raw)
    finally:
        # Do not put filesystem writes inside the measured receive loop.
        destination.write_bytes(b"".join(raw))
    if result["prompt_tokens"] + result["output_tokens"] > args.seq_len:
        raise AssertionError("Sequence exceeds Graph bucket; eager fallback possible")
    if result["output_tokens"] < 2:
        raise AssertionError("Need >=2 output tokens to exercise decode Graph")
    if args.ignore_eos and result["output_tokens"] != args.max_tokens:
        raise AssertionError("Forced-length request did not generate max_tokens")
    return result


def summarize(rows):
    # Every supplied request counts, including the first one. No warmup filter.
    return {
        key: statistics.median(row[key] for row in rows)
        for key in (
            "ttft_s",
            "last_content_s",
            "generation_finish_s",
            "stream_done_s",
            "output_tokens",
            "prompt_tokens",
        )
    }


def request_schedule(repeats, order=ORDER):
    """Every consecutive request targets a different model, including rounds."""
    previous = None
    # Close the final cycle so the first model also has `repeats` genuine
    # switch samples; the empty -> first model request is additional.
    for step, name in enumerate(order * repeats + (order[0],)):
        yield {
            "step": step,
            "round": step // len(order),
            "model": name,
            "previous_model": previous,
            "request_type": "initial_load" if previous is None else "model_switch",
        }
        previous = name


def model_groups(all_three=False):
    if all_three:
        return (ORDER,)
    return ((ORDER[0], ORDER[1]), (ORDER[1], ORDER[2]), (ORDER[2], ORDER[0]))


def cpu_cache_gb(models, override=0):
    if override:
        return override
    # Registry uses BF16 bytes/parameter plus 20% headroom. CPU ONLY.
    return math.ceil(
        sum(float(model["params"]) for model in models) * 2e9 * 1.2 / 2**30
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Compatibility flag; only 0 is accepted (no warmup)",
    )
    parser.add_argument(
        "--cpu-model-cache-gb",
        type=int,
        default=0,
        help="CPU cache GiB; 0 auto-estimates for each pair (not GPU memory)",
    )
    parser.add_argument(
        "--all-three",
        action="store_true",
        help="Optional three-model CPU cache; default tests pairs",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        choices=(0, 1, 2),
        help="Only run one default pair; otherwise run all pairs",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--load-first", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 2 or args.cpu_model_cache_gb < 0:
        parser.error("Require repeats>=1, max-tokens>=2, cpu-model-cache-gb>=0")
    if args.warmup != 0:
        parser.error("This is a switch-latency test: warmup must be 0")
    baseline = json.loads((args.source_run / "report.json").read_text())
    if baseline.get("status") != "pass":
        raise RuntimeError("Require an existing PASS SAVE/LOAD source run")
    template = yaml.safe_load((args.source_run / "multi-model-load.yaml").read_text())
    by_name = {model["name"]: model for model in template["models"]}
    models = [by_name[name] for name in ORDER]
    groups = model_groups(args.all_three)
    if args.pair_index is not None:
        if args.all_three:
            parser.error("--pair-index cannot be combined with --all-three")
        groups = (groups[args.pair_index],)
    args.seq_len = template["server"]["cuda_graph"]["seq_len_buckets"][0]
    args.archive = Path(baseline["archive_root"])
    args.hook = ROOT / "foundry/python/foundry/libcuda_hook.so"
    args.cuda_compat = Path.home() / ".local/cuda-12.9/compat"
    for model in models:
        validate_archive(args, model)
        if not Path(model["path"]).is_dir():
            raise RuntimeError(f"Missing model: {model['path']}")
    if not args.hook.is_file():
        raise RuntimeError(f"Missing hook: {args.hook}")
    print(
        f"Preflight OK: existing archives, models={list(ORDER)}, bucket={args.seq_len}"
    )
    for index, order in enumerate(groups):
        cache = cpu_cache_gb([by_name[name] for name in order], args.cpu_model_cache_gb)
        print(
            f"Group {index}: {list(order)}, CPU weight cache={cache} GiB, zero warmup"
        )
    if args.preflight_only:
        return 0
    try:
        with socket.create_connection((args.host, args.port), timeout=1):
            raise RuntimeError("Port occupied; existing service will not be stopped")
    except OSError:
        pass
    if args.run_dir is None:
        args.run_dir = ROOT / "work" / time.strftime("graph-latency-%Y%m%d-%H%M%S")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / ".gitignore").write_text("*\n")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    report = {
        "status": "running",
        "source_run": str(args.source_run),
        "archive": str(args.archive),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "repeats": args.repeats,
        "warmup": 0,
        "cpu_model_cache_gb": args.cpu_model_cache_gb,
        "groups": [
            {
                "pair": index,
                "models": list(order),
                "cpu_cache_gb": cpu_cache_gb(
                    [by_name[name] for name in order], args.cpu_model_cache_gb
                ),
            }
            for index, order in enumerate(groups)
        ],
        "scope": "TP1 text serial batch1, every request switches model, existing LOAD only",
        "protocol": "Register CPU weights once; zero warmup; request-driven GPU switches",
        "metric": "client request start to received finish_reason chunk",
        "phase_order": ["load", "off"] if args.load_first else ["off", "load"],
        "requests": [],
        "deployments": [],
        "summary": [],
        "errors": [],
    }
    references = {}

    def save():
        (args.run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    try:
        for mode in report["phase_order"]:
            for group_index, order in enumerate(groups):
                group_models = [by_name[name] for name in order]
                group_template = json.loads(json.dumps(template))
                group_template["server"]["model_cache_size_gb"] = cpu_cache_gb(
                    group_models, args.cpu_model_cache_gb
                )
                args.switch_order = [
                    visit["model"] for visit in request_schedule(args.repeats, order)
                ]
                with server(
                    args,
                    group_template,
                    mode,
                    group_models,
                    f"pair-{group_index}-{mode}",
                ):
                    # Register/cache CPU weights once. This is NOT GPU inference
                    # or Graph loading; Worker switches happen inside chat requests.
                    for model in group_models:
                        name = model["name"]
                        start = time.perf_counter()
                        deploy(args, model)
                        report["deployments"].append(
                            {
                                "mode": mode,
                                "pair": group_index,
                                "model": name,
                                "deploy_http_s": time.perf_counter() - start,
                            }
                        )
                    for visit in request_schedule(args.repeats, order):
                        name = visit["model"]
                        destination = (
                            args.run_dir
                            / f"pair-{group_index}-{mode}-step-{visit['step']}.sse"
                        )
                        result = timed_request(args, by_name[name], destination)
                        if name in references:
                            compare_tokens(references[name], result)
                        else:
                            references[name] = result
                        row = {
                            "mode": mode,
                            "pair": group_index,
                            **visit,
                            "warmup": False,
                            "sse_path": str(destination),
                            **result,
                        }
                        report["requests"].append(row)
                        save()
                        print(
                            f"{mode:4} step={visit['step']} "
                            f"{visit['previous_model'] or '(empty)'} -> {name}: "
                            f"tokens={result['output_tokens']} "
                            f"finish={result['generation_finish_s']:.3f}s "
                            f"last-content={result['last_content_s']:.3f}s "
                            f"TTFT={result['ttft_s']:.3f}s",
                            flush=True,
                        )
        for group_index, order in enumerate(groups):
            for name in order:
                for category in ("all_requests", "model_switch", "initial_load"):
                    selected = {
                        mode: [
                            row
                            for row in report["requests"]
                            if row["pair"] == group_index
                            and row["model"] == name
                            and row["mode"] == mode
                            and (
                                category == "all_requests"
                                or row["request_type"] == category
                            )
                        ]
                        for mode in ("off", "load")
                    }
                    if not selected["off"]:
                        continue
                    stats = {mode: summarize(rows) for mode, rows in selected.items()}
                    off = stats["off"]["generation_finish_s"]
                    load = stats["load"]["generation_finish_s"]
                    report["summary"].append(
                        {
                            "model": name,
                            "pair": group_index,
                            "category": category,
                            "samples_per_mode": len(selected["off"]),
                            "off": stats["off"],
                            "load": stats["load"],
                            "speedup": off / load,
                            "reduction_percent": (1 - load / off) * 100,
                        }
                    )
        report["status"] = "pass"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["errors"].append("Interrupted by user")
    except Exception as exc:
        report["status"] = "fail"
        report["errors"].append(str(exc))
    save()
    if report["status"] == "pass":
        with (args.run_dir / "summary.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "model",
                    "pair",
                    "category",
                    "samples_per_mode",
                    "output_tokens",
                    "off_finish_median_s",
                    "load_finish_median_s",
                    "speedup",
                    "reduction_percent",
                    "off_last_content_median_s",
                    "load_last_content_median_s",
                ]
            )
            print(
                "\nModel | category | samples | tokens | OFF median(s) | LOAD median(s) | speedup"
            )
            for entry in report["summary"]:
                off, load = entry["off"], entry["load"]
                writer.writerow(
                    [
                        entry["model"],
                        entry["pair"],
                        entry["category"],
                        entry["samples_per_mode"],
                        off["output_tokens"],
                        off["generation_finish_s"],
                        load["generation_finish_s"],
                        entry["speedup"],
                        entry["reduction_percent"],
                        off["last_content_s"],
                        load["last_content_s"],
                    ]
                )
                print(
                    f"pair-{entry['pair']}/{entry['model']} | {entry['category']} | "
                    f"{entry['samples_per_mode']} | {off['output_tokens']:g} | "
                    f"{off['generation_finish_s']:.3f} | "
                    f"{load['generation_finish_s']:.3f} | {entry['speedup']:.3f}x"
                )
    print(f"\n{report['status'].upper()}: {args.run_dir / 'report.json'}", flush=True)
    if report["errors"]:
        print("\n".join(report["errors"]))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
