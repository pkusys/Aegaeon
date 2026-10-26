#!/usr/bin/env python3
"""Resolve the vLLM architecture that Aegaeon logs at startup.

When Aegaeon (re)loads a model it constructs a vLLM ``ModelConfig`` and prints
``Resolved architecture: <arch>`` (vLLM ``config/model.py``).  That <arch> is
the value returned by ``ModelRegistry.inspect_model_cls``: the first entry of
the model's ``config.json`` ``architectures`` list that vLLM can load, or a
name resolved through transformers/auto_map fallbacks.  It is the registry
*key*, which is not always the loaded class name:

    ChatGLMModel                 -> loads vllm...chatglm:ChatGLMForCausalLM
    Cohere2ForCausalLM           -> loads vllm...commandr:CohereForCausalLM
    StableLmForCausalLM          -> loads vllm...stablelm:StablelmForCausalLM

This script reports the SAME string vLLM would log (plus the actual class that
is loaded), so the output matches the Aegaeon startup log line instead of the
family name guessed from the config.

Resolution mirrors Aegaeon's own model config: ``trust_remote_code=True`` and
``dtype=bfloat16`` (aegaeon.config.ModelConfig always forces bf16).  A repo id
is resolved against the local mirror under ``/srv/models/<org>/<name>/``;
if absent, only ``config.json`` (+ ``*.py`` for remote-code archs) is fetched
from ModelScope (HuggingFace is blocked on this host).

Usage:
    python scripts/arch_lookup.py LLM-Research/gemma-3-4b-it
    python scripts/arch_lookup.py /srv/models/Llama/Llama-2-13b-chat-ms
    python scripts/arch_lookup.py --matrix --model Llama-3.2-1B --model Gemma3n
    python scripts/arch_lookup.py --matrix            # whole tests/arch_matrix.yaml
"""
import argparse
import glob
import json
import logging
import math
import os
import sys

MODEL_ROOT = "/srv/models"

SMOKE_PATTERNS = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "*.py",
    "tokenizer*",
]

# Aegaeon builds every model with these settings (aegaeon/config.py
# ModelConfig.torch_dtype -> bfloat16).  Use them so the resolved architecture
# is identical to what a running Aegaeon worker/controller would log.
LOOKUP_DTYPE = "bfloat16"
LOOKUP_TRUST_REMOTE_CODE = True

# vLLM loggers are chatty during ModelConfig construction; keep the screen
# readable while still surfacing real resolution failures (raised as errors).
for _name in ("vllm", "vllm.config.model", "vllm.model_executor.models.registry"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def resolve_config(arg: str) -> str:
    if arg.startswith("local:"):
        arg = arg[len("local:"):]
    if os.path.isdir(arg):
        return os.path.join(arg, "config.json")

    org, _, name = arg.partition("/")
    target = os.path.join(MODEL_ROOT, org, name or org)
    local = os.path.join(target, "config.json")
    if os.path.isfile(local):
        return local

    print(f"config not mirrored at {local}; fetching from ModelScope ...")
    from modelscope.hub.snapshot_download import snapshot_download

    try:
        snapshot_download(
            arg,
            local_dir=target,
            allow_file_pattern=SMOKE_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: could not fetch '{arg}': {exc}") from exc
    if not os.path.isfile(local):
        raise SystemExit(f"error: config.json not found for '{arg}'")
    return local


def read_architectures(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)
    return list(cfg.get("architectures") or [])


def count_params(model_path: str):
    total = 0
    for pattern in ("*.safetensors", "*/*.safetensors"):
        for sf_file in sorted(glob.glob(os.path.join(model_path, pattern))):
            import safetensors

            with safetensors.safe_open(sf_file, framework="pt") as sf:
                for key in sf.keys():
                    total += math.prod(sf.get_slice(key).get_shape())
    return total


def resolve_architectures(config_path: str) -> dict:
    """Return the runtime architecture exactly as Aegaeon/vLLM would resolve it.

    Constructing the real ``vllm.config.ModelConfig`` reproduces
    ``Resolved architecture: ...`` (vLLM stores the result in
    ``ModelConfig._architecture``), so the value here is byte-for-byte the one
    Aegaeon logs at model switch/load time.
    """
    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import ModelRegistry

    # Importing vllm reconfigures its loggers; quiet them only after import so
    # resolution output stays readable (real failures are raised, not logged).
    for _name in ("vllm", "vllm.config.model",
                  "vllm.model_executor.models.registry"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    model_dir = os.path.dirname(config_path)
    result = {
        "ok": False,
        "config_archs": read_architectures(config_path),
        "resolved": None,
        "module": None,
        "class_name": None,
        "error": None,
    }
    try:
        mc = ModelConfig(
            model=model_dir,
            trust_remote_code=LOOKUP_TRUST_REMOTE_CODE,
            dtype=LOOKUP_DTYPE,
        )
    except Exception as exc:  # noqa: BLE001 - report vLLM's resolution failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["ok"] = True
    resolved = getattr(mc, "_architecture", None)
    result["resolved"] = resolved

    if resolved is not None:
        try:
            model_cls = ModelRegistry._try_load_model_cls(resolved)
        except Exception as exc:  # noqa: BLE001
            model_cls = None
            result["error"] = f"class load failed: {type(exc).__name__}: {exc}"
        if model_cls is not None:
            result["module"] = model_cls.__module__
            result["class_name"] = model_cls.__name__
    return result


def format_report(arg: str, config_path: str, r: dict) -> list:
    lines = []
    model_dir = os.path.dirname(config_path)
    params = count_params(model_dir)
    if params:
        lines.append(
            f"  params: {params / 1e9:.2f}B  "
            f"(~{params * 2 / (1024 ** 3):.1f}GiB bf16)"
        )

    if not r["ok"]:
        lines.append(f"  config architectures: {r['config_archs']}")
        lines.append(f"  => NOT SUPPORTED / resolution failed: {r['error']}")
        return lines

    lines.append(f"  config architectures: {r['config_archs']}")
    if r["resolved"] is None:
        lines.append("  => resolved architecture: <none>")
        return lines

    loaded = (
        f"{r['module']}:{r['class_name']}"
        if r["class_name"]
        else "<registry key not directly loadable>"
    )
    alias = (
        ""
        if r["class_name"] is None or r["class_name"] == r["resolved"]
        else "   <-- loaded class differs from the resolved key"
    )
    lines.append(f"  => resolved architecture (Aegaeon log): {r['resolved']}")
    lines.append(f"  => loaded vLLM class: {loaded}{alias}")
    if r["error"]:
        lines.append(f"  warning: {r['error']}")
    return lines


def _lookup_one(arg: str) -> None:
    config_path = resolve_config(arg)
    r = resolve_architectures(config_path)
    print(f"\n{arg}")
    print(f"  config: {config_path}")
    for line in format_report(arg, config_path, r):
        print(line)


def _iter_matrix(matrix_path: str, filters: list) -> None:
    import yaml

    with open(matrix_path) as f:
        entries = yaml.safe_load(f).get("models", [])

    if filters:
        wanted = set(filters)
        entries = [e for e in entries if e.get("name") in wanted]
        missing = wanted - {e.get("name") for e in entries}
        if missing:
            print(f"  model(s) not in matrix: {sorted(missing)}")

    header = (f"{'name':22s} {'config architectures':38s} "
              f"{'resolved (log)':34s} loaded class")
    print(header)
    print("-" * len(header))
    for entry in entries:
        name = entry.get("name", "?")
        repo = entry.get("repo") or ""
        if entry.get("skip"):
            print(f"{name:22s} {str(repo)[:38]:38s} SKIP: {entry['skip']}")
            continue
        if not repo.strip():
            print(f"{name:22s} {'':38s} NO REPO")
            continue
        try:
            config_path = resolve_config(repo)
        except SystemExit as exc:
            print(f"{name:22s} {str(repo)[:38]:38s} ERROR: {exc}")
            continue
        r = resolve_architectures(config_path)
        cfg_archs = ", ".join(r["config_archs"])[:38]
        if not r["ok"]:
            print(f"{name:22s} {cfg_archs:38s} FAILED: {r['error']}")
            continue
        resolved = r["resolved"] or "<none>"
        loaded = (
            r["class_name"] or r["module"] or "-"
        )
        print(f"{name:22s} {cfg_archs:38s} {resolved:34s} {loaded}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "model", nargs="*",
        help="ModelScope repo id (org/name) or local path to the model dir",
    )
    ap.add_argument(
        "--matrix",
        action="store_true",
        help="iterate over the model matrix (tests/arch_matrix.yaml)",
    )
    ap.add_argument(
        "--matrix-path",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "arch_matrix.yaml",
        ),
        help="path to the YAML matrix (default: tests/arch_matrix.yaml)",
    )
    ap.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="in matrix mode, only these model names (repeatable)",
    )
    args = ap.parse_args()

    if args.matrix:
        _iter_matrix(args.matrix_path, args.models or [])
        return

    if not args.model:
        ap.error("provide at least one model (repo id / path) or --matrix")

    for arg in args.model:
        try:
            _lookup_one(arg)
        except SystemExit as exc:
            print(exc)


if __name__ == "__main__":
    main()
