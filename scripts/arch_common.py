#!/usr/bin/env python3
"""Shared helpers for the architecture test scripts."""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_ROOT = "/srv/models"

# Derived from aegaeon's device module (aegaeon.device_registry) instead of
# being hardcoded here: DEVICE_NAME is the logical DeviceSpec name and
# DEVICE_PROFILE the profile subdirectory name carried in DeviceSpec.extra.
from aegaeon.device_registry import DeviceRegistry  # noqa: E402

PROFILE_ROOT = os.path.join(ROOT, "profiles")


def _device_registry() -> DeviceRegistry:
    """The runtime device registry: aegaeon config YAML, else built-in defaults."""
    candidates = []
    if os.environ.get("AEGAEON_CONFIG_PATH"):
        candidates.append(os.environ["AEGAEON_CONFIG_PATH"])
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return DeviceRegistry.from_yaml(path)
            except Exception:
                continue
    return DeviceRegistry.default()


def _resolve_device():
    """Detect the local GPU through aegaeon's DeviceRegistry.

    The logical name (e.g. ``A100_40GB``) is resolved from the CUDA device name
    via the registry's alias map; the profile directory key (e.g.
    ``A100-PCIE-40GB``) is taken from the DeviceSpec ``extra["profile"]``.
    Falls back to the first known device when CUDA is unavailable.
    """
    reg = _device_registry()
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            spec = reg.from_str(name)
            if spec is not None:
                return spec
    except Exception:
        pass
    for alias in ("a100_40gb", "a100"):
        spec = reg.from_str(alias)
        if spec is not None:
            return spec
    return next(iter(reg))


_DEVICE = _resolve_device()
DEVICE_NAME = _DEVICE.name
DEVICE_PROFILE = str(_DEVICE.extra.get("profile") or _DEVICE.name)


def device_dict() -> dict:
    """Server-config ``devices`` entry for the current GPU.

    All values come from aegaeon's resolved ``DeviceSpec`` (name, memory,
    bandwidth and the profile subdirectory key), so the test servers and the
    runtime agree on device identity.
    """
    entry = {
        "name": DEVICE_NAME,
        "memory": int(_DEVICE.memory) if _DEVICE.memory.is_integer() else _DEVICE.memory,
        "profile": DEVICE_PROFILE,
    }
    if _DEVICE.bandwidth is not None:
        bw = _DEVICE.bandwidth
        entry["bandwidth"] = int(bw) if isinstance(bw, float) and bw.is_integer() else bw
    return entry

# A profile whose latency numbers are structurally valid; used as the template
# for placeholder profiles of models that have never been profiled.
PROFILE_TEMPLATE = os.path.join(
    PROFILE_ROOT, "Qwen", "Qwen2.5-1.5B-Instruct", DEVICE_PROFILE
)


def load_matrix(path=None):
    if path is None:
        path = os.path.join(ROOT, "tests", "arch_matrix.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw.get("models", [])


def model_name_for(entry) -> str:
    """Full model name (<org>/<repo>) used for the API, registry and profiles.

    Local entries resolve to the path relative to MODEL_ROOT so the name
    matches the on-disk layout, e.g. ``/srv/models/Qwen/Qwen3-4B``
    -> ``Qwen/Qwen3-4B``.
    """
    repo = entry.get("repo") or ""
    if repo.startswith("local:"):
        path = repo[len("local:"):]
        rel = os.path.relpath(path, MODEL_ROOT)
        if rel.startswith("..") or os.path.isabs(rel):
            return os.path.basename(path)
        return rel
    return repo


def model_path_for(entry) -> str:
    repo = entry.get("repo") or ""
    if repo.startswith("local:"):
        return repo[len("local:"):]
    org, _, name = repo.partition("/")
    return os.path.join(MODEL_ROOT, org, name)


def profile_dir_for(name: str) -> str:
    return os.path.join(PROFILE_ROOT, name, DEVICE_PROFILE)


def cfg_model_entry(entry) -> dict:
    """Server-config ``models`` entry for one matrix row.

    Propagates the optional per-model ``max_model_len`` so engines cap the
    vLLM context (e.g. PLaMo2's 10M-position RoPE table) instead of deriving
    the raw config value.
    """
    return {
        "name": model_name_for(entry),
        "path": model_path_for(entry),
        **({"max_model_len": entry["max_model_len"]}
           if entry.get("max_model_len") is not None else {}),
    }


def deploy_body_for(entry, model_path=None) -> dict:
    """HTTP deploy payload for one matrix row (``max_model_len`` aware)."""
    body = {"model": model_name_for(entry)}
    if model_path is not None:
        body["model_path"] = model_path
    if entry.get("max_model_len") is not None:
        body["max_model_len"] = entry["max_model_len"]
    return body
