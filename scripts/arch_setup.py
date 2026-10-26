#!/usr/bin/env python3
"""Fetch model files for the architecture test matrix (ModelScope).

``smoke`` (default): download only config/tokenizer/remote-code files so
architecture tests run without full weights.
``full``: download complete weights (skipping redundant .bin/.pth/.onnx/...
duplicates).  If a repo only ships legacy ``.bin``/``.pth``, they are
downloaded and converted to ``.safetensors`` in place (source removed), so
every full download ends QuickLoader-ready.

Downloads run through aria2c.  LFS weight files are resolved (via the proxy,
briefly) to a signed CDN URL and fetched straight from the CDN with up to 16
connections/file; non-LFS metadata (config/tokenizer/...) has no CDN object and
is fetched through the proxy with retries.  Set
``AEGAEON_DOWNLOADER=modelscope`` to fall back to ``snapshot_download``.

Usage:
    python scripts/arch_setup.py --mode smoke  --model Gemma3n          # quick
    python scripts/arch_setup.py --mode full   --model Hunyuan-7B       # weights
    python scripts/arch_setup.py --dry-run --mode full --model Hunyuan-7B
    python scripts/arch_setup.py --mode full --ssh user@fast-box        # push out
    python scripts/arch_setup.py --list-models
"""
import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_ROOT = "/srv/models"
API = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
RESOLVE = "https://modelscope.cn/models/{model}/resolve/master/{path}"
UA = {"User-Agent": "Mozilla/5.0"}

ARIA2C = shutil.which("aria2c")
# aria2 allows at most 16 connections per file; that already reaches ~20 MB/s
# on this host's direct CDN path (vs ~2.4 MB/s and flaky behaviour via proxy).
ARIA2_CONNS = min(int(os.environ.get("AEGAEON_ARIA2_CONNECTIONS", "16")), 16)

SMOKE_PATTERNS = [
    "config.json", "configuration.json", "generation_config.json",
    "tokenizer*", "merges.txt", "vocab.json", "chat_template*",
    "preprocessor_config.json", "processor_config.json", "*.py",
]
IGNORE = [
    "*.bin", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.ot", "*.onnx",
    "*.tflite", "*.pb", "consolidated.safetensors", "original/*", "onnx/*",
]
# Bin-only repos: fetch legacy weights so they can be converted in place.
IGNORE_NO_BIN = [p for p in IGNORE if p not in ("*.bin", "*.pth")]


def load_matrix(path: str):
    with open(path) as f:
        return yaml.safe_load(f).get("models", [])


def repo_files(model_id: str):
    with urllib.request.urlopen(
        urllib.request.Request(API.format(model=model_id), headers=UA), timeout=30
    ) as resp:
        data = json.load(resp)
    return [{"path": f.get("Path", ""), "size": int(f.get("Size", 0)),
             "lfs": bool(f.get("IsLFS", False))}
            for f in data.get("Data", {}).get("Files", [])]


def ignored(path: str, patterns: list = None) -> bool:
    patterns = IGNORE if patterns is None else patterns
    name = os.path.basename(path)
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(path, p)
               for p in patterns)


def _has_safetensors(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return any(
        name.endswith(".safetensors") and name != "consolidated.safetensors"
        for _, _, names in os.walk(directory) for name in names
    )


def _repo_has_safetensors(repo: str) -> bool:
    try:
        files = repo_files(repo)
    except Exception:  # noqa: BLE001 - assume safetensors on listing failure
        return True
    return any(
        f["path"].endswith(".safetensors")
        and os.path.basename(f["path"]) != "consolidated.safetensors"
        for f in files
    )


def _full_ignore_for(repo: str, target: str) -> list:
    """Skip .bin/.pth only when safetensors are already/remotely available."""
    if (target and _has_safetensors(target)) or _repo_has_safetensors(repo):
        return IGNORE
    return IGNORE_NO_BIN


def _wanted_files(files: list, patterns: list, ignore: list) -> list:
    wanted = []
    for f in files:
        path = f["path"]
        if not int(f.get("size", 0)):          # directory/empty placeholders
            continue
        if ignore and ignored(path, ignore):
            continue
        if patterns and not any(
            fnmatch.fnmatch(path, p) or fnmatch.fnmatch(os.path.basename(path), p)
            for p in patterns
        ):
            continue
        wanted.append(f)
    return wanted


def _purge_partials(directory: str) -> None:
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if name.endswith(".incomplete"):
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass


def _resolve_direct(url: str, attempts: int = 5, timeout: int = 8):
    """Resolve to the final CDN URL (via proxy) and probe direct Range access.

    The resolve/probe hops go through the proxy (modelscope.cn is only
    reachable that way); per-request the proxy is occasionally flaky, so each
    hop has a short timeout and the whole resolution is retried a few times.
    The object bytes are then pulled straight from the CDN without the proxy.
    """
    final = None
    for _ in range(attempts):
        req = urllib.request.Request(url, headers=UA)
        req.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.build_opener().open(req, timeout=timeout) as resp:
                final = resp.geturl()
        except Exception:  # noqa: BLE001
            continue
        try:
            probe = urllib.request.Request(
                final, headers={**UA, "Range": "bytes=0-4095"}
            )
            with urllib.request.build_opener(
                urllib.request.ProxyHandler({})
            ).open(probe, timeout=timeout) as resp:
                if resp.status == 206:
                    return final, True
        except Exception:  # noqa: BLE001
            continue
    return final, False


def _run_aria2(url: str, dest_dir: str, name: str) -> None:
    """Run aria2c for one LFS/CDN file (direct, proxy env stripped)."""
    cmd = [
        ARIA2C, "-c", "--auto-file-renaming=false",
        "--allow-overwrite=true", "--max-tries=5", "--retry-wait=3",
        "--timeout=15", "--connect-timeout=10",
        "--file-allocation=none", "--console-log-level=warn",
        "--summary-interval=0",
        "-x", str(ARIA2_CONNS), "-s", str(ARIA2_CONNS), "-k1M",
        "-d", dest_dir or ".", "-o", name,
        "--header=User-Agent: Mozilla/5.0",
    ]
    env = {k: v for k, v in os.environ.items()
           if "proxy" not in k.lower()}
    cmd += [url]
    subprocess.run(cmd, check=True, env=env)


def _fetch_plain(url: str, dest: str, attempts: int = 3, timeout: int = 60):
    """Fetch one non-LFS metadata file with a plain GET through the proxy.

    The modelscope blob server for these files does not honour byte ranges, so
    aria2's segmented download fails on them; a plain streaming GET works.
    """
    for attempt in range(attempts):
        tmp = dest + ".part"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.build_opener().open(
                req, timeout=timeout
            ) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(tmp, dest)
            return
        except Exception:  # noqa: BLE001
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt == attempts - 1:
                raise


def download_repo_aria2(repo: str, target: str, patterns: list, ignore: list) -> None:
    """aria2 download.

    LFS weight files are resolved (via the proxy) to a signed CDN URL and then
    fetched straight from the CDN (fast, proxy-free).  Non-LFS metadata files
    (config/tokenizer/...) have no CDN object -- they only exist on the proxied
    modelscope.cn host -- so they go through the proxy with short timeouts and
    are retried a few times.
    """
    import time

    files = _wanted_files(repo_files(repo), patterns, ignore)
    if not files:
        return
    _purge_partials(target)
    os.makedirs(target, exist_ok=True)
    total = sum(f["size"] for f in files)
    direct_gb = sum(f["size"] for f in files if f.get("lfs")) / (1024**3)
    print(f"  aria2c: {len(files)} files, {total / (1024**3):.2f} GB -> {target} "
          f"({direct_gb:.2f} GB LFS direct, {ARIA2_CONNS} conns/file)")

    failed = []
    for f in files:
        path, size = f["path"], int(f.get("size", 0))
        dest = os.path.join(target, path)
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        if (
            os.path.isfile(dest)
            and not os.path.exists(dest + ".aria2")
            and os.path.getsize(dest) == size
        ):
            print(f"    {path}  (present)")
            continue

        resolve_url = RESOLVE.format(model=repo,
                                     path=urllib.parse.quote(path, safe="/"))
        print(f"    {path}  ({size / (1024**2):.1f} MiB) "
              f"[{'LFS/CDN' if f.get('lfs') else 'proxy'}]")
        start = time.time()
        try:
            if f.get("lfs"):
                # Resolve to the signed CDN object (one short proxied hop), then
                # download direct from the CDN with aria2.
                final_url, direct_ok = _resolve_direct(resolve_url)
                if not direct_ok:
                    raise RuntimeError("could not resolve to CDN")
                _run_aria2(final_url, dest_dir or ".", os.path.basename(path))
            else:
                # Metadata: no CDN object and no byte-range support on the blob
                # server; fetch with a plain GET through the proxy.
                _fetch_plain(resolve_url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR {path}: {exc}; rerun to retry")
            failed.append(path)
            continue
        got = os.path.getsize(dest)
        if got:
            print(f"      -> {got / (1024**2):.1f} MiB in "
                  f"{time.time() - start:.1f}s "
                  f"({got / (time.time() - start) / (1024**2):.1f} MiB/s)")
    if failed:
        raise RuntimeError(
            f"{len(failed)} file(s) failed to download "
            f"(proxy flaky?); rerun to retry: {failed[:5]}"
        )


def downloader() -> str:
    if os.environ.get("AEGAEON_DOWNLOADER", "aria2") == "modelscope":
        return "modelscope"
    return "aria2" if ARIA2C else "modelscope"


def download_repo(repo: str, target: str, patterns: list, ignore: list) -> None:
    if downloader() == "aria2":
        download_repo_aria2(repo, target, patterns, ignore)
        return
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(repo, local_dir=target,
                      allow_file_pattern=patterns, ignore_file_pattern=ignore)


def local_dir_for(out_root: str, repo: str) -> str:
    org, _, name = repo.partition("/")
    return os.path.join(out_root, org, name or org)


def _read_arch(path: str):
    with open(os.path.join(path, "config.json")) as f:
        return list(json.load(f).get("architectures") or [])


def _smoke_keep(files: list) -> list:
    return [
        f for f in files
        if not ignored(f["path"])
        and any(
            fnmatch.fnmatch(f["path"], p)
            or fnmatch.fnmatch(os.path.basename(f["path"]), p)
            for p in SMOKE_PATTERNS
        )
    ]


def setup_entry(entry, mode: str, force: bool, out_root: str, dry_run: bool):
    name = entry["name"]
    repo = entry.get("repo") or ""
    if entry.get("skip"):
        return ("skip", name, entry.get("skip"))
    if repo.startswith("local:"):
        path = repo[len("local:"):]
        if not os.path.isfile(os.path.join(path, "config.json")):
            return ("error", name, f"local dir missing config.json: {path}")
        return ("local", name, path, _read_arch(path))

    target = local_dir_for(out_root, repo)
    marker = os.path.join(target, "config.json")

    if dry_run:
        keep = repo_files(repo)
        if mode == "smoke":
            keep = _smoke_keep(keep)
        else:
            ignore = _full_ignore_for(repo, None)
            keep = [f for f in keep if not ignored(f["path"], ignore)]
        keep.sort(key=lambda f: -f["size"])
        print(f"  {name:22s} would download {len(keep)} files, "
              f"{sum(f['size'] for f in keep) / (1024**3):.2f} GB -> {target}")
        for f in keep[:8]:
            print(f"    {f['size'] / (1024**2):9.2f} MiB  {f['path']}")
        return ("dry-run", name, target)

    os.makedirs(target, exist_ok=True)
    if mode == "full":
        patterns, ignore = None, _full_ignore_for(repo, target)
    else:
        patterns, ignore = SMOKE_PATTERNS, None
        if os.path.isfile(marker) and not force:
            return ("present", name, target)

    print(f"[{mode}] downloading {repo} -> {target} (via {downloader()})")
    download_repo(repo, target, patterns, ignore)
    if not os.path.isfile(marker):
        return ("error", name, f"config.json missing after download: {target}")
    arch = _read_arch(target)

    if mode == "full" and not _has_safetensors(target):
        from convert_bin_to_safetensors import convert_directory, should_convert
        if should_convert(target):
            print(f"  {name}: converting .bin/.pth -> safetensors ...")
            try:
                result = convert_directory(target)
            except Exception as exc:  # noqa: BLE001
                return ("error", name, f"conversion failed: {exc}")
            print(f"  converted {result['tensors']} tensors "
                  f"-> {result['shards']} shards")
    return ("ok", name, target, arch)


def verify(out_root: str, repo: str) -> dict:
    dst = local_dir_for(out_root, repo)
    total = 0
    st_files = 0
    for root, _dirs, names in os.walk(dst):
        for name in names:
            total += os.path.getsize(os.path.join(root, name))
            if name.endswith(".safetensors") and name != "consolidated.safetensors":
                st_files += 1
    return {"cfg": os.path.isfile(os.path.join(dst, "config.json")),
            "st": st_files, "bytes": total}


def ssh_transfer(repo: str, out_root: str, dest: str, ssh_path: str):
    src = local_dir_for(out_root, repo)
    remote = f"{dest}:{ssh_path.rstrip('/')}/{os.path.relpath(src, out_root)}"
    if shutil.which("rsync"):
        cmd = ["rsync", "-avP", "--partial", src.rstrip(os.sep) + os.sep, remote]
    else:
        cmd = ["scp", "-o", "BatchMode=yes", "-r", src, remote]
    print(f"  -> {' '.join(cmd[:4])} ...")
    subprocess.run(cmd, check=True)


def ensure_profile(entry: dict) -> int:
    """Create the placeholder latency profile for ``entry`` if missing.

    Returns 1 when a profile was created, 0 when it already exists (or the
    entry has no repo), and -1 when the profile template is unavailable.
    """
    if entry.get("skip") or not (entry.get("repo") or "").strip():
        return 0
    from arch_common import model_name_for, profile_dir_for, PROFILE_TEMPLATE
    if not os.path.isdir(PROFILE_TEMPLATE):
        return -1
    dst = profile_dir_for(model_name_for(entry))
    if os.path.isdir(dst):
        return 0
    shutil.copytree(PROFILE_TEMPLATE, dst)
    print(f"  profile created  {model_name_for(entry):28s} {dst}")
    return 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--matrix", default=os.path.join(ROOT, "tests", "arch_matrix.yaml"))
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--force", action="store_true",
                    help="re-download smoke files even if config.json is present")
    ap.add_argument("--model", default=None, help="only process this model name")
    ap.add_argument("--out", default=MODEL_ROOT, help=f"download root "
                    f"(default: {MODEL_ROOT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files that would download without downloading")
    ap.add_argument("--ssh", default=None,
                    help="transfer to server after full download, e.g. user@server")
    ap.add_argument("--ssh-path", default=MODEL_ROOT,
                    help="remote directory for --ssh")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model list from the matrix and exit")
    args = ap.parse_args()

    entries = load_matrix(args.matrix)
    if args.model:
        entries = [e for e in entries if e["name"] == args.model]
        if not entries:
            sys.exit(f"model '{args.model}' not in matrix")

    if args.list_models:
        for i, e in enumerate(entries, 1):
            print(f"  {i}. {e['name']:28s} {e.get('repo','') or '(local)':46s} "
                  f"{e.get('arch',''):38s} real={e.get('real')}")
        return

    if not args.dry_run and downloader() == "modelscope":
        try:
            import modelscope  # noqa: F401
        except ImportError:
            sys.exit("modelscope not installed (or aria2c missing)")

    os.makedirs(args.out, exist_ok=True)
    summary, ok_repos = [], []
    profiles_created = 0
    template_warned = False
    for entry in entries:
        result = setup_entry(entry, args.mode, args.force, args.out, args.dry_run)
        status, name = result[0], result[1]
        if status == "ok":
            if args.mode == "full":
                info = verify(args.out, entry.get("repo"))
                vstatus = "OK" if info["cfg"] and info["st"] > 0 else "WARN"
                print(f"  PASS {name:22s} {result[3]}  [{vstatus}] "
                      f"config={info['cfg']} safetensors={info['st']} "
                      f"size={info['bytes'] / (1024**3):.2f} GB")
                if vstatus == "OK":
                    ok_repos.append(entry.get("repo"))
            else:
                print(f"  PASS {name:22s} {result[3]}")
            summary.append((name, "ok"))
        elif status == "local":
            print(f"  PASS {name:22s} {result[3]}  (local {result[2]})")
            summary.append((name, "ok"))
        elif status == "dry-run":
            summary.append((name, "dry-run"))
        else:
            print(f"  {status.upper():7s} {name:22s} {result[2]}")
            summary.append((name, status))

        # Every processed model gets its latency profile automatically.
        if not args.dry_run:
            made = ensure_profile(entry)
            if made == -1 and not template_warned:
                print("  warning: profile template missing; skipping profiles")
                template_warned = True
            elif made == 1:
                profiles_created += 1

    if args.ssh and ok_repos:
        for repo in ok_repos:
            ssh_transfer(repo, args.out, args.ssh, args.ssh_path)

    print(f"\n{len(summary)} entries, "
          f"{sum(1 for _, s in summary if s == 'ok')} ready"
          + (f"; {profiles_created} profiles created" if profiles_created else ""))


if __name__ == "__main__":
    main()
