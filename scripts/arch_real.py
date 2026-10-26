#!/usr/bin/env python3
"""Phase 2: real-weights verification for the passing architectures.

Starts an isolated server per model with ``dummy_weights: false``, deploys via
the API, sends a real prompt and checks that the generated text is meaningful
(non-empty, varied characters, whitespace-separated words). Results are
appended to ``results/arch_real.jsonl``.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_common import (  # noqa: E402
    cfg_model_entry,
    deploy_body_for,
    device_dict,
    model_name_for,
    model_path_for,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs_real")
RESULT_FILE = os.path.join(RESULTS_DIR, "arch_real.jsonl")
SERVER_CONFIG = os.path.join(RESULTS_DIR, "arch_real_server.yaml")
MODEL_ROOT = "/srv/models"

PROMPTS = {
    "chat": "Please introduce yourself in one or two sentences.",
    "completion": "The capital city of France is",
}


def load_matrix(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw.get("models", [])


def model_weight_bytes(path):
    import glob
    total = 0
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        total += os.path.getsize(f)
    return total


def select_entries(matrix, names=None):
    out = []
    for e in matrix:
        if e.get("skip"):
            continue
        if e.get("real") is False:
            continue
        if names and e["name"] not in names:
            continue
        if not (e.get("repo") or "").strip():
            continue
        out.append(e)
    return out


def write_server_config(entry, weight_bytes):
    cache_gb = max(4, int((weight_bytes / (1024**3)) + 4))
    # Listing the model in `models:` lets the engine size the GPU KV cache
    # accounting for the model weights (model_size is subtracted), so a large
    # model does not exhaust the arena. Real weights are present locally.
    cfg = {
        "server": {
            "cpu_num_slabs": 4,
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
            cfg_model_entry(entry)
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


def meaningful(text):
    if not text or len(text) < 10:
        return False
    distinct = set(text)
    if len(distinct) < 3:
        return False
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return len(words) >= 2


def run_one(entry, port, gpu, start_timeout=600):
    name = model_name_for(entry)
    path = model_path_for(entry)
    base = f"http://127.0.0.1:{port}"
    log_file = os.path.join(LOG_DIR, f"{entry['name'].replace('/', '_')}.log")
    weight_bytes = model_weight_bytes(path)
    record = {
        "name": name, "matrix_name": entry["name"], "arch": entry.get("arch", ""),
        "family": entry.get("family", ""), "weight_bytes": weight_bytes,
        "phase": "real",
    }
    t0 = time.time()
    proc = None
    try:
        write_server_config(entry, weight_bytes)
        proc = start_server(port, log_file, gpu)
        ok, msg = wait_healthy(base, timeout=start_timeout)
        if not ok:
            record.update(status="start_fail", error=msg, duration=round(time.time() - t0, 1))
            return record, proc

        code, body = _http("POST", base + "/v1/models/deploy",
                           deploy_body_for(entry, path),
                           timeout=300)
        if code != 200:
            record.update(status="deploy_fail",
                          error=json.dumps(body, ensure_ascii=False)[:500],
                          duration=round(time.time() - t0, 1))
            return record, proc

        # Use chat if the tokenizer has a chat template; otherwise completion.
        use_chat = _has_chat_template(path)
        if use_chat:
            payload = {"model": name,
                       "messages": [{"role": "user", "content": PROMPTS["chat"]}],
                       "max_tokens": 512, "ignore_eos": True}
            code, body = _http("POST", base + "/v1/chat/completions", payload, timeout=600)
        else:
            payload = {"model": name, "prompt": PROMPTS["completion"],
                       "max_tokens": 512, "ignore_eos": True}
            code, body = _http("POST", base + "/v1/completions", payload, timeout=600)

        record["endpoint"] = "chat" if use_chat else "completion"
        if code != 200:
            record.update(status="fail",
                          error=json.dumps(body, ensure_ascii=False)[:800],
                          duration=round(time.time() - t0, 1))
            return record, proc

        try:
            choice = body["choices"][0]
            text = choice["message"]["content"] if use_chat else choice["text"]
            record["finish_reason"] = str(choice.get("finish_reason", ""))
            record["completion_tokens"] = body["usage"]["completion_tokens"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(body)[:300]
            record["completion_tokens"] = 0
        text = text or ""
        record["output"] = text
        record["meaningful"] = meaningful(text)
        record["status"] = "pass" if meaningful(text) else "fail"
        record["duration"] = round(time.time() - t0, 1)
        return record, proc
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}",
                      duration=round(time.time() - t0, 1))
        return record, proc


def _has_chat_template(path):
    try:
        import json
        with open(os.path.join(path, "tokenizer_config.json")) as f:
            cfg = json.load(f)
        if cfg.get("chat_template"):
            return True
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tok.chat_template is not None
    except Exception:
        return False


def _http(method, url, body=None, timeout=300):
    import urllib.request
    import urllib.error
    import socket
    data = json.dumps(body).encode() if body is not None else None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"text": raw}
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        raise TimeoutError(f"http error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(ROOT, "tests", "arch_matrix.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=19000)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    matrix = load_matrix(args.matrix)
    names = [n.strip() for n in (args.model or "").split(",") if n.strip()] or None
    entries = select_entries(matrix, names)
    if not entries:
        print("nothing to test (no real:true models selected)")
        sys.exit(1)

    results = []
    for i, entry in enumerate(entries):
        record, proc = run_one(entry, args.port + i * 10, args.gpu)
        kill_server(proc)
        time.sleep(5)
        results.append(record)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = record["status"]
        print(f"{record['name'][:40]:40s} {record['arch'][:30]:30s} {status.upper():10s} ({record['duration']}s)")
        out = record.get("output", "")
        if record["status"] == "pass":
            print(f"    -> {out[:180]}")
        elif record.get("error"):
            print(f"    {record['error'][:200]}")
        else:
            print(f"    (output not meaningful: {out[:120]!r})")

    n_pass = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{n_pass}/{len(results)} meaningful outputs -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
