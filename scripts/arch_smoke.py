#!/usr/bin/env python3
"""Phase 1 architecture smoke test.

For each model in the matrix: start an isolated Aegaeon server in dummy-weights
mode, deploy the model, send one chat request, and record PASS/FAIL.

Dummy mode constructs the model on the GPU with random buffers, so this
exercises the full pipeline (config parse -> Aegaeon ModelConfig -> vLLM arch
registration -> model build -> forward -> KV cache -> decode) without
transferring any weights. Garbage output is expected and counts as PASS.

Results are appended to ``results/arch_smoke.jsonl``.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_common import device_dict, model_name_for, model_path_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
RESULT_FILE = os.path.join(RESULTS_DIR, "arch_smoke.jsonl")
SERVER_CONFIG = os.path.join(RESULTS_DIR, "arch_smoke_server.yaml")

MODEL_ROOT = "/srv/models"

HEADERS = {"Content-Type": "application/json"}

CHAT_PROMPT = "Hello, please introduce yourself in one sentence."


def load_matrix(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw.get("models", [])


def entries_to_test(matrix, model=None):
    out = []
    for e in matrix:
        if model and e["name"] != model:
            continue
        if e.get("skip"):
            continue
        if not (e.get("repo") or "").strip():
            continue
        out.append((e, model_path_for(e)))
    return out


def write_server_config(entries):
    # NOTE: the models list is intentionally empty. Listing a model here makes
    # the Controller load its real weights into the shared CPU cache at
    # startup (even in dummy mode), which breaks config-only smoke dirs and
    # models larger than the cache. Models are instead deployed via the API,
    # whose dummy path registers a weight-free handle.
    cfg = {
        "server": {
            "cpu_num_slabs": 4,
            "cpu_slab_size_bytes": 1073741824,
            "memory_utilization": 0.25,
            "worker_num_cpus": 4,
            "nnodes": 1,
            "num_prefill_engines": 0,
            "num_decode_engines": 0,
            "num_engines": 1,
            "model_cache_size_gb": 4,
            "dummy_weights": True,
        },
        "models": [],
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
    out_file = open(log_file + ".stdout", "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "aegaeon", "start",
            "--config", SERVER_CONFIG,
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=out_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._out_file = out_file
    return proc


def kill_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
    try:
        proc._out_file.close()
    except Exception:
        pass


def wait_healthy(base, timeout):
    deadline = time.time() + timeout
    no_proxy = {"http": None, "https": None}
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=5, proxies=no_proxy)
            if r.status_code == 200:
                return True, ""
        except requests.RequestException:
            pass
        time.sleep(3)
    return False, "server not healthy within timeout"


def http_json(method, url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    # Bypass HTTP(S)_PROXY: the sandbox proxy wedges localhost CONNECT tunnels.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, {"text": raw}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"text": raw}
    except urllib.error.URLError as e:
        raise TimeoutError(f"connection error: {e.reason}")
    except socket.timeout:
        raise TimeoutError("socket timeout")
    except TimeoutError:
        raise


def run_one(entry, path, port, gpu, start_timeout=300, retries=2, req_timeout=300):
    name = model_name_for(entry)
    base = f"http://127.0.0.1:{port}"
    log_file = os.path.join(LOG_DIR, f"{entry['name'].replace('/', '_')}.log")
    record = {
        "name": name,
        "matrix_name": entry["name"],
        "arch": entry.get("arch", ""),
        "family": entry.get("family", ""),
        "risk": entry.get("risk", ""),
        "repo": entry.get("repo", ""),
        "phase": "smoke",
        "endpoint": "chat",
    }
    last_record = None
    for attempt in range(retries + 1):
        proc = None
        t0 = time.time()
        try:
            proc = start_server(port, log_file, gpu)
            ok, msg = wait_healthy(base, timeout=start_timeout)
            if not ok:
                record.update(
                    status="start_fail", error=msg,
                    duration=round(time.time() - t0, 1), attempt=attempt,
                )
                last_record = dict(record)
                kill_server(proc)
                time.sleep(8)
                continue

            code, body = http_json(
                "POST", base + "/v1/models/deploy",
                {"model": name, "model_path": path}, timeout=120,
            )
            if code != 200:
                record.update(
                    status="deploy_fail",
                    error=json.dumps(body, ensure_ascii=False)[:500],
                    duration=round(time.time() - t0, 1), attempt=attempt,
                )
                last_record = dict(record)
                kill_server(proc)
                return record, proc

            code, body = http_json(
                "POST", base + "/v1/chat/completions",
                {
                    "model": name,
                    "messages": [{"role": "user", "content": CHAT_PROMPT}],
                    "max_tokens": 512, "ignore_eos": True,
                },
                timeout=req_timeout,
            )
            endpoint_used = "chat"
            if code != 200:
                err = json.dumps(body, ensure_ascii=False)
                if "chat_template" in err and "cannot render" in err:
                    # Model lacks a chat template -> fall back to raw completion.
                    code, body = http_json(
                        "POST", base + "/v1/completions",
                        {"model": name, "prompt": CHAT_PROMPT, "max_tokens": 512, "ignore_eos": True},
                        timeout=req_timeout,
                    )
                    endpoint_used = "chat+completion"
            if code != 200:
                record.update(
                    status="fail",
                    error=json.dumps(body, ensure_ascii=False)[:800],
                    duration=round(time.time() - t0, 1), attempt=attempt,
                )
                return record, proc

            text = ""
            completion_tokens = 0
            finish_reason = ""
            try:
                choice = body["choices"][0]
                if endpoint_used == "chat":
                    text = choice["message"]["content"] or ""
                else:
                    text = choice["text"] or ""
                finish_reason = str(choice.get("finish_reason", ""))
            except (KeyError, IndexError, TypeError):
                pass
            try:
                completion_tokens = int(body["usage"]["completion_tokens"])
            except (KeyError, TypeError):
                pass
            record["endpoint"] = endpoint_used
            record["completion_tokens"] = completion_tokens
            record["finish_reason"] = finish_reason
            # The decode loop must actually emit tokens for the architecture to
            # count as functional; with dummy weights the text itself is garbage.
            record.update(
                status="pass" if completion_tokens > 0 else "fail",
                output_preview=text[:120],
                duration=round(time.time() - t0, 1), attempt=attempt,
            )
            return record, proc
        except TimeoutError as e:
            record.update(status="timeout", error=str(e),
                          duration=round(time.time() - t0, 1), attempt=attempt)
            last_record = dict(record)
            kill_server(proc)
            time.sleep(8)
        except Exception as e:
            record.update(
                status="error",
                error=f"{type(e).__name__}: {e}",
                duration=round(time.time() - t0, 1), attempt=attempt,
            )
            last_record = dict(record)
            kill_server(proc)
            time.sleep(8)
    return last_record, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(ROOT, "tests", "arch_matrix.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--all-in-one", action="store_true",
                    help="run all models on one server instead of per-model isolation")
    ap.add_argument("--gpu-fallback", action="store_true",
                    help="retry a GPU-0 failure on GPU 1")
    ap.add_argument("--req-timeout", type=int, default=300,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=2,
                    help="server-start retries")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    matrix = load_matrix(args.matrix)
    entries = entries_to_test(matrix, args.model)
    if not entries:
        print("nothing to test (matrix empty / all skipped)")
        sys.exit(1)

    print(f"{'NAME':24s} {'ARCH':34s} {'STATUS'}")
    results = []
    if args.all_in_one:
        write_server_config(entries)
        port = args.port
        proc = None
        for entry, path in entries:
            record, proc = run_one(entry, path, port, args.gpu, retries=args.retries, req_timeout=args.req_timeout)
            results.append(record)
            _print_row(record)
            _save_record(record)
            # undeploy between models keeps switches realistic; rely on restart per entry
        if proc is not None:
            kill_server(proc)
    else:
        for i, (entry, path) in enumerate(entries):
            write_server_config([(entry, path)])
            record, proc = run_one(entry, path, args.port + i * 10, args.gpu, retries=args.retries, req_timeout=args.req_timeout)
            kill_server(proc)
            time.sleep(5)  # let ray reap the previous run's actors
            results.append(record)
            _print_row(record)
            _save_record(record)
            if record["status"] in ("fail", "error", "timeout", "start_fail") and args.gpu_fallback and args.gpu == 0:
                print(f"  -> retrying {entry['name']} on GPU 1")
                record2, proc2 = run_one(entry, path, args.port + i * 10 + 5, 1, req_timeout=args.req_timeout)
                kill_server(proc2)
                if record2["status"] == "pass":
                    record.update(status="pass", status_note="pass-on-gpu1", output_preview=record2.get("output_preview", ""))
                else:
                    record["status"] = "fail"
                    record["error"] = f"gpu0: {record.get('error')}; gpu1: {record2.get('error')}"
                _save_record(record)

    n_pass = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{len(results)} tested, {n_pass} pass -> {RESULT_FILE}", flush=True)


def _save_record(record):
    with open(RESULT_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_row(record):
    status = record["status"]
    mark = "PASS" if status == "pass" else status.upper()
    print(f"{record['name']:24s} {record['arch'][:34]:34s} {mark}  ({record['duration']}s)", flush=True)
    if status != "pass" and record.get("error"):
        print(f"    {record['error'][:300]}", flush=True)


if __name__ == "__main__":
    import socket  # noqa: F401 (used by http_json)
    main()
