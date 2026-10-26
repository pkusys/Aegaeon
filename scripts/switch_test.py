#!/usr/bin/env python3
"""Model-switch correctness test on real weights.

Starts one isolated server per model pair (both models listed in the config's
``models:`` so the CPU QuickCache pre-caches both weight sets), then:

1. Sequential phase: alternating requests A -> B -> A -> B -> A force the
   scheduler to switch the engine between models (``worker.switch``, which
   resets worker state, re-initialises the vLLM WorkspaceManager and reloads
   weights from the CPU cache).
2. Concurrent phase: staggered interleaved requests exercise the decode-phase
   prefetch switch path.

Every request must produce meaningful output. Switch transitions are verified
against the ``/v1/aegaeon/runtime`` ``model_switch`` event log. Results are
appended to ``results/switch_test.jsonl``.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_common import (  # noqa: E402
    cfg_model_entry,
    deploy_body_for,
    device_dict,
    load_matrix,
    model_name_for,
    model_path_for,
)
from arch_real import (  # noqa: E402
    _has_chat_template,
    _http,
    kill_server,
    meaningful,
    model_weight_bytes,
    wait_healthy,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs_switch")
RESULT_FILE = os.path.join(RESULTS_DIR, "switch_test.jsonl")
SERVER_CONFIG = os.path.join(RESULTS_DIR, "switch_test_server.yaml")

PROMPTS = {
    "chat": "Please introduce yourself in one or two sentences.",
    "completion": "The capital city of France is",
}

# Name -> (arch, family) from the matrix, selected by `--pair`.
PAIRS = [
    #("Qwen3.5-0.8B", "Qwen2.5-1.5B"),    # MoE <-> dense (WorkspaceManager)
    #("Llama-3.2-1B", "Qwen3-4B"),        # dense <-> dense
    #("Phi-3-mini-4k", "gemma-3-1b-it"),  # dense <-> dense
    #("gemma-2-2b-it", "gemma-3-1b-it"),  # gemma-2 (soft-capped attn) <-> gemma-3
    #("Qwen2.5-VL-3B", "Qwen3-4B"),       # multimodal <-> dense (text path)
    #("chatglm3-6b", "Mistral-7B-v0.3"),  # large dense <-> large dense
    #("Llama-2-13B", "Qwen2.5-1.5B"),
    #("InternLM2.5-7B", "Qwen3-4B"),
    #("Granite3-2B", "StableLM-3B"),
    #("Apertus-8B", "AFM-4.5B"),
    #("Command-R7B", "Dots-ocr"),
    #("ERNIE4.5-0.3B", "Falcon-rw-7B"),
    #("Gemma3n", "Gemma1.1-2B"),
    #("Gemma4-E2B", "glm-4-9b-chat-hf"),
    #("pythia-12b", "gpt2"),
    #("HRM-1B", "InternLM3-8B"),
    #("Hunyuan-7B", "IQuest-7B"),
    #("Jais2-8B", "LFM2.5-350M"),
    #("MiMo-7B", "MiniCPM-1B"),
    #("MiniCPM3-4B", "mpt-7b"),
    #("Minitron-8B", "Olmo3-7B"),
    #("Olmo-Hybrid-7B", "opt-125m"),
    #("Orion-14B", "Ouro-1.4B"),
    #("openPangu-7B", "phi-2"),
    #("plamo2-1b", "TeleChat2-3B"),
    #("plamo2-1b", "Zamba2-1.2B"),
    #("bloom-7b", "Exaone3-7.8B"),
    #("Exaone4-1.2B", "FalconMamba-7B"),
    #("Falcon-H1-0.5B", "GLM4-9B-0414"),
    #("gpt-j-6b", "granite4-tiny"),
    #("Hyper-14B", "mamba-130m-hf"),
    #("mamba-codestral-7B", "mamba-130m-hf"),
    ("Nemotron-H-8B", "mamba-130m-hf")
]


def write_server_config(entries):
    weight_total = sum(model_weight_bytes(model_path_for(e)) for e in entries)
    cache_gb = max(8, int((weight_total / (1024**3)) + 4))
    cfg = {
        "server": {
            # SSM/hybrid pairs (Mamba2, Nemotron-H, ...) have large composite
            # KV blocks, so the default 4 GiB CPU swap fills during concurrent
            # switching; give the swap pool enough headroom.
            "cpu_num_slabs": 32,
            "cpu_slab_size_bytes": 1073741824,
            "memory_utilization": 0.775,
            "worker_num_cpus": 4,
            "nnodes": 1,
            "num_prefill_engines": 0,
            "num_decode_engines": 0,
            "num_engines": 1,
            "model_cache_size_gb": cache_gb,
            "dummy_weights": False,
        },
        "models": [
            cfg_model_entry(e) for e in entries
        ],
        "devices": [device_dict()],
    }
    with open(SERVER_CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return SERVER_CONFIG


def start_server(port, log_file, gpu):
    env = dict(os.environ)
    env["AEGAEON_CONFIG_PATH"] = SERVER_CONFIG
    env["AEGAEON_LOG_FILE"] = log_file
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = open(log_file + ".stdout", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "aegaeon", "start", "--config", SERVER_CONFIG,
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._out_file = out
    return proc


def send_request(base, entry, max_tokens=512, timeout=600):
    """One chat/completion request; returns a result dict (never raises)."""
    name = model_name_for(entry)
    path = model_path_for(entry)
    t0 = time.time()
    try:
        use_chat = _has_chat_template(path)
        if use_chat:
            payload = {
                "model": name,
                "messages": [{"role": "user", "content": PROMPTS["chat"]}],
                "max_tokens": max_tokens,
                "ignore_eos": True,
            }
            code, body = _http("POST", base + "/v1/chat/completions", payload, timeout=timeout)
            endpoint = "chat"
        else:
            payload = {"model": name, "prompt": PROMPTS["completion"],
                       "max_tokens": max_tokens, "ignore_eos": True}
            code, body = _http("POST", base + "/v1/completions", payload, timeout=timeout)
            endpoint = "completion"
    except Exception as e:
        return {"status": "error", "endpoint": "?", "error": f"{type(e).__name__}: {e}",
                "elapsed": round(time.time() - t0, 1)}

    if code != 200:
        return {"status": "http_error", "code": code, "endpoint": endpoint,
                "error": json.dumps(body, ensure_ascii=False)[:400],
                "elapsed": round(time.time() - t0, 1)}
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"] if endpoint == "chat" else choice["text"]
        finish = str(choice.get("finish_reason", ""))
        ctokens = body["usage"]["completion_tokens"]
    except (KeyError, IndexError, TypeError):
        text = json.dumps(body)[:200]
        finish, ctokens = "?", 0
    text = text or ""
    ok = meaningful(text)
    return {
        "status": "pass" if ok else "fail",
        "endpoint": endpoint,
        "meaningful": ok,
        "finish_reason": finish,
        "completion_tokens": ctokens,
        "output": text,
        "elapsed": round(time.time() - t0, 1),
    }


def concurrent_pair(base, entry_a, entry_b, max_tokens=512):
    """Staggered A/B requests to force interleaved scheduling + prefetch."""
    results = {}

    def _send(tag, entry, delay):
        if delay:
            time.sleep(delay)
        results[tag] = send_request(base, entry, max_tokens=max_tokens)

    t0 = time.time()
    threads = [
        threading.Thread(target=_send, args=("a", entry_a, 0.0)),
        threading.Thread(target=_send, args=("b", entry_b, 0.2)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=900)
    for th in threads:
        if th.is_alive():
            results["a"] = {"status": "timeout", "error": "thread did not finish"}
            results["b"] = {"status": "timeout", "error": "thread did not finish"}
    return results, round(time.time() - t0, 1)


def runtime_switches(base):
    """Fetch switch events from /v1/aegaeon/runtime."""
    code, body = _http("GET", base + "/v1/aegaeon/runtime", timeout=30)
    if code != 200:
        return [], str(body)
    events = body.get("events", []) if isinstance(body, dict) else []
    return [e for e in events if e.get("type") == "model_switch"], ""


def log_warnings(log_file):
    """Count non-prefetch-fallback tracebacks in the server log.

    A failed weight prefetch logs a traceback but is an intentional, handled
    path (the worker falls back to a direct load), so those blocks are not
    counted. Any other traceback is a real error.
    """
    n = 0
    try:
        with open(log_file, errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return -1
    skip_block = False
    for line in lines:
        if "falling back to direct load" in line:
            skip_block = True
        elif "Traceback" in line and not skip_block:
            n += 1
        elif line.strip() and not line.lstrip().startswith(("File ", "  ", "Error", "ValueError")):
            skip_block = False
    return n


def run_pair(entry_a, entry_b, port, gpu, cycles=2, max_tokens=512):
    base = f"http://127.0.0.1:{port}"
    tag = f"{entry_a['name']}_x_{entry_b['name']}"
    log_file = os.path.join(LOG_DIR, f"{tag}.log")
    record = {
        "pair": tag,
        "arch_a": entry_a.get("arch", ""),
        "arch_b": entry_b.get("arch", ""),
        "risk_a": entry_a.get("risk", ""),
        "risk_b": entry_b.get("risk", ""),
        "cycles": cycles,
        "requests": [],
    }
    t0 = time.time()
    proc = None
    try:
        write_server_config([entry_a, entry_b])
        proc = start_server(port, log_file, gpu)
        ok, msg = wait_healthy(base, timeout=900)
        if not ok:
            record.update(status="start_fail", error=msg, duration=round(time.time() - t0, 1))
            return record, proc

        # Both models are already cached from the config; deploy is idempotent.
        for entry in (entry_a, entry_b):
            code, body = _http("POST", base + "/v1/models/deploy",
                               deploy_body_for(entry, model_path_for(entry)), timeout=300)
            if code != 200:
                record.update(status="deploy_fail",
                              error=f"{model_name_for(entry)}: {json.dumps(body, ensure_ascii=False)[:300]}",
                              duration=round(time.time() - t0, 1))
                return record, proc

        # --- Sequential phase: A -> B -> A -> B -> A ---
        seq = []
        for i in range(cycles):
            seq.append(("A", entry_a))
            seq.append(("B", entry_b))
        seq.append(("A", entry_a))
        for step, (label, entry) in enumerate(seq):
            req = send_request(base, entry, max_tokens=max_tokens)
            req["step"] = f"seq{step}:{label}"
            req["model"] = model_name_for(entry)
            record["requests"].append(req)
            ok_text = req["status"]
            print(f"    [seq {step} -> {label}] {ok_text:10s} {req.get('elapsed', '?')}s"
                  f" {req.get('finish_reason', '')}")
            if req["status"] not in ("pass",):
                print(f"        {req.get('error', req.get('output', ''))[:200]}")

        # --- Concurrent phase: staggered interleaved requests ---
        for order in ("A-first", "B-first"):
            a, b = (entry_a, entry_b) if order == "A-first" else (entry_b, entry_a)
            results, elapsed = concurrent_pair(base, a, b, max_tokens=max_tokens)
            for tag2, req in results.items():
                req = dict(req)
                req["step"] = f"concurrent:{order}:{tag2}"
                req["model"] = model_name_for(a if tag2 == "a" else b)
                record["requests"].append(req)
                print(f"    [concurrent {order} {tag2}] {req['status']:10s} {req.get('elapsed', '?')}s"
                      f" {req.get('finish_reason', '')}")

        # --- Switch event verification ---
        switches, err = runtime_switches(base)
        expected = 2 * cycles + 1  # initial + each transition in A->B->A->...->A
        record["switch_events"] = [
            {"from": e.get("from_model"), "to": e.get("to_model"),
             "prefetch": e.get("prefetch_model")} for e in switches
        ]
        record["switch_event_count"] = len(switches)

        warnings = log_warnings(log_file)
        record["log_tracebacks"] = warnings
        record["duration"] = round(time.time() - t0, 1)

        reqs_ok = all(r["status"] == "pass" for r in record["requests"])
        events_ok = len(switches) >= expected
        if reqs_ok and events_ok and warnings == 0:
            record["status"] = "pass"
        else:
            record["status"] = "fail"
            record["error"] = json.dumps({
                "meaningful_fail": sum(1 for r in record["requests"] if r["status"] != "pass"),
                "switch_events": len(switches),
                "expected_min": expected,
                "tracebacks": warnings,
                "runtime_err": err or "",
            })
        return record, proc
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}",
                      duration=round(time.time() - t0, 1))
        return record, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(ROOT, "tests", "arch_matrix.yaml"))
    ap.add_argument("--pair", default=None,
                    help="comma-separated names (e.g. 'Qwen3.5-0.8B,Qwen2.5-1.5B'); default: all pairs")
    ap.add_argument("--port", type=int, default=19000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-concurrent", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    matrix = {e["name"]: e for e in load_matrix(args.matrix)}

    if args.pair:
        names = [n.strip() for n in args.pair.split(",") if n.strip()]
        pairs = [names]
    else:
        pairs = list(PAIRS)

    results = []
    for i, names in enumerate(pairs):
        missing = [n for n in names if n not in matrix]
        if missing:
            print(f"SKIP pair {names}: unknown names {missing}")
            continue
        print(f"\n=== pair {i + 1}/{len(pairs)}: {names[0]} x {names[1]} ===")
        entry_a, entry_b = matrix[names[0]], matrix[names[1]]
        record, proc = run_pair(entry_a, entry_b, args.port + i * 10, args.gpu,
                                cycles=args.cycles, max_tokens=args.max_tokens)
        kill_server(proc)
        time.sleep(5)
        results.append(record)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_ok = sum(1 for r in record["requests"] if r["status"] == "pass")
        n_all = len(record["requests"])
        print(f"  {record.get('status', '?').upper():10s} requests {n_ok}/{n_all}"
              f" switches={record.get('switch_event_count', 0)}"
              f" tracebacks={record.get('log_tracebacks', '?')} ({record.get('duration', '?')}s)")
        if record.get("error"):
            print(f"  detail: {str(record['error'])[:300]}")

    n_pass = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{n_pass}/{len(results)} pairs passed -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
