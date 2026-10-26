#!/usr/bin/env python3
"""ModelScope download speed test -- runs on Windows and Linux.

Measures the effective download speed from ModelScope for a model repo, and
optionally the speed of a subsequent ``scp`` transfer to a server. Run the SAME
script on both machines to compare:

    laptop download :  python ms_speed_test.py --model Qwen/Qwen3-4B
    laptop -> server:  python ms_speed_test.py --model Qwen/Qwen3-4B --ssh user@host --ssh-path /srv/models
    server direct   :  python ms_speed_test.py --model Qwen/Qwen3-4B

Only the Python standard library is required. If ``modelscope`` is installed,
``--sdk`` measures a real snapshot download instead of the raw stream test.

Windows note: if a system HTTP(S)_PROXY interferes with local/ModelScope
traffic, add ``--no-proxy``.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

API = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
RESOLVE = "https://modelscope.cn/models/{model}/resolve/master/{path}"
UA = {"User-Agent": "Mozilla/5.0"}
MB = 1024 * 1024


def mb(n: int) -> float:
    return n / MB


def _open(url: str, no_proxy: bool, timeout: int):
    import urllib.request

    if no_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()
    return opener.open(urllib.request.Request(url, headers=UA), timeout=timeout)


def get_repo_files(model: str):
    with _open(API.format(model=model), no_proxy=False, timeout=30) as resp:
        data = json.load(resp)
    files = []
    for f in data.get("Data", {}).get("Files", []):
        files.append({"path": f.get("Path", ""), "size": int(f.get("Size", 0))})
    return files


def pick_file(files, pattern: str = None):
    if pattern:
        for f in files:
            if pattern in f["path"]:
                return f
    st = [f for f in files if f["path"].endswith(".safetensors") and f["size"] > 0]
    if not st:
        st = [f for f in files if f["size"] > 0]
    return max(st, key=lambda f: f["size"]) if st else None


def stream_test(url: str, target_bytes: int, no_proxy: bool, timeout: int = 120):
    """Stream up to ``target_bytes`` from ``url`` and time it."""
    chunk = 1 * MB
    start = time.perf_counter()
    got = 0
    with _open(url, no_proxy, timeout) as resp:
        while got < target_bytes:
            data = resp.read(min(chunk, target_bytes - got))
            if not data:
                break
            got += len(data)
    elapsed = time.perf_counter() - start
    return got, elapsed


def sdk_download(model: str, file_pattern: str, work_dir: str):
    """Full-file download via the modelscope SDK (if installed)."""
    from modelscope.hub.snapshot_download import snapshot_download

    start = time.perf_counter()
    snapshot_download(
        model,
        local_dir=work_dir,
        allow_file_pattern=[file_pattern],
        ignore_file_pattern=["*.bin", "*.pth"],
    )
    elapsed = time.perf_counter() - start
    total = 0
    for root, _dirs, names in os.walk(work_dir):
        for name in names:
            total += os.path.getsize(os.path.join(root, name))
    return total, elapsed


def ssh_speed(dest: str, ssh_path: str, size_mb: int, port: int = None):
    """Generate a temp file and time an scp transfer to the server."""
    fd, tmp = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\0" * (size_mb * MB))
    remote = f"{dest}:{ssh_path.rstrip('/')}/ms_speed_test_tmp.bin"
    cmd = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if port:
        cmd += ["-P", str(port)]
    cmd += [tmp, remote]
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start
    os.remove(tmp)
    return size_mb * MB, elapsed


def fmt_speed(bps: float) -> str:
    return f"{mb(bps):.2f} MiB/s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="ModelScope model id")
    ap.add_argument("--file", default=None,
                    help="specific file substring to test (default: largest *.safetensors)")
    ap.add_argument("--bytes", type=int, default=200,
                    help="bytes (MiB) to stream in the raw speed test (default 200)")
    ap.add_argument("--sdk", action="store_true",
                    help="use modelscope SDK full-file download instead of the raw stream test")
    ap.add_argument("--no-proxy", action="store_true", help="bypass HTTP(S)_PROXY")
    ap.add_argument("--ssh", default=None, help="scp destination, e.g. user@server")
    ap.add_argument("--ssh-path", default="/tmp", help="remote directory for the ssh test")
    ap.add_argument("--ssh-port", type=int, default=None)
    ap.add_argument("--ssh-bytes", type=int, default=256,
                    help="temporary file size (MiB) for the ssh test (default 256)")
    ap.add_argument("--list-files", action="store_true", help="only list repo files and sizes")
    args = ap.parse_args()

    print(f"== ModelScope speed test | model: {args.model} | platform: {sys.platform}")
    try:
        files = get_repo_files(args.model)
    except Exception as e:
        print(f"ERROR: cannot query repo: {type(e).__name__}: {e}")
        sys.exit(1)

    total_repo = sum(f["size"] for f in files)
    print(f"   repo files: {len(files)} | total size: {total_repo / (1024**3):.2f} GB")

    if args.list_files:
        for f in sorted(files, key=lambda x: -x["size"]):
            if f["size"] > 0:
                print(f"     {mb(f['size']):9.2f} MiB  {f['path']}")
        sys.exit(0)

    test = pick_file(files, args.file)
    if test is None:
        print("ERROR: no downloadable file found")
        sys.exit(1)

    print(f"   test file : {test['path']} ({mb(test['size']):.2f} MiB)")

    results = {}

    # --- Raw stream test (stdlib) ---
    if not args.sdk:
        url = RESOLVE.format(model=args.model, path=test["path"])
        target = min(args.bytes * MB, test["size"])
        print(f"\n[1] raw stream test: {mb(target):.0f} MiB of {test['path']}")
        try:
            got, elapsed = stream_test(url, target, args.no_proxy)
            speed = got / elapsed
            results["download"] = (got, elapsed, speed)
            eta = (test["size"] - got) / speed if speed > 0 else float("inf")
            print(f"    downloaded {mb(got):.1f} MiB in {elapsed:.2f}s -> {fmt_speed(speed)}")
            if speed > 0:
                print(f"    ETA for full file ({mb(test['size']):.1f} MiB): {eta/60:.1f} min")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

    # --- SDK full-file download (if installed) ---
    if args.sdk:
        print(f"\n[1] modelscope SDK download: {test['path']}")
        try:
            got, elapsed = sdk_download(args.model, test["path"], tempfile.mkdtemp(prefix="ms_speed_"))
            speed = got / elapsed
            results["download"] = (got, elapsed, speed)
            print(f"    downloaded {mb(got):.1f} MiB in {elapsed:.2f}s -> {fmt_speed(speed)}")
        except ImportError:
            print("    ERROR: modelscope SDK not installed. Run: pip install modelscope")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

    # --- scp transfer test (laptop -> server leg) ---
    if args.ssh:
        print(f"\n[2] scp transfer: {args.ssh_bytes} MiB -> {args.ssh}:{args.ssh_path}")
        try:
            got, elapsed = ssh_speed(args.ssh, args.ssh_path, args.ssh_bytes, args.ssh_port)
            speed = got / elapsed
            results["ssh"] = (got, elapsed, speed)
            print(f"    transferred {mb(got):.1f} MiB in {elapsed:.2f}s -> {fmt_speed(speed)}")
        except FileNotFoundError:
            print("    ERROR: scp not found (install OpenSSH client on Windows)")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

    # --- Summary ---
    print("\n== Summary ==")
    if "download" in results:
        got, elapsed, speed = results["download"]
        print(f"  ModelScope download : {mb(got):.1f} MiB / {elapsed:.2f}s = {fmt_speed(speed)}")
    if "ssh" in results:
        got, elapsed, speed = results["ssh"]
        print(f"  scp transfer        : {mb(got):.1f} MiB / {elapsed:.2f}s = {fmt_speed(speed)}")
    if "download" in results and "ssh" in results:
        _, _, d = results["download"]
        _, _, s = results["ssh"]
        total = d * s / (d + s) if d + s > 0 else 0
        if total > 0:
            print(f"  combined (download then ssh): {fmt_speed(total)}")
        print("  => compare against 'server direct' run of the same script.")


if __name__ == "__main__":
    main()
