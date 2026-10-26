#!/usr/bin/env python3
"""Convert legacy PyTorch checkpoints (``.bin`` / ``.pth``) to ``.safetensors``.

Aegaeon can only ingest safetensors checkpoints: ``QuickCache`` / ``ModelMeta``
glob ``*.safetensors`` and raise ``NotImplementedError`` when a model has none
(see ``aegaeon/loader/meta.py`` and ``aegaeon/loader/cache.py``).  This script
closes that gap automatically.

Everything is derived from the environment -- no arguments are required:

1. Candidate model dirs are discovered from
   * the model matrix (``tests/arch_matrix.yaml``: ``local:`` paths and
     mirrored ``<org>/<repo>`` dirs under ``MODEL_ROOT``), and
   * every dir under ``MODEL_ROOT`` that contains a ``config.json``.
2. A dir is converted only when it has at least one ``.bin``/``.pth`` weight
   file and does NOT yet contain any ``*.safetensors`` (so re-running is a
   no-op, and dirs that already carry both formats are left untouched).
3. The state dict is read with ``torch.load`` (original tensor dtypes are
   preserved -- this is a byte-faithful format migration), re-sharded into
   ``model-0000N-of-0000M.safetensors`` plus ``model.safetensors.index.json``
   in the same directory, and the legacy weight files are then removed.

Safety valves (optional, unset by default; none required for normal use):

* ``AEGAEON_MODEL_ROOT`` - override the default model mirror root
  (``/srv/models``).
* ``AEGAEON_SHARD_BYTES`` - max safetensors shard size in bytes
  (default 4 GiB).
* ``AEGAEON_KEEP_BIN=1`` - convert but do not delete the ``.bin``/``.pth``
  source files.
"""
import glob
import json
import math
import os
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_ROOT = os.environ.get("AEGAEON_MODEL_ROOT", "/srv/models")
SHARD_BYTES = int(os.environ.get("AEGAEON_SHARD_BYTES", str(4 * (1024 ** 3))))
KEEP_BIN = os.environ.get("AEGAEON_KEEP_BIN", "0") == "1"

SOURCE_PATTERNS = ("*.bin", "*.pth")
SAFETENSOR_PATTERNS = ("*.safetensors", "**/*.safetensors")
# Never descend into a directory we created for retired sources.
_SKIP_DIR_PARTS = {"bin_originals", "safetensors_originals"}


def gb(value: float) -> str:
    return f"{value / (1024 ** 3):.2f}GiB"


def _find_files(directory: str, patterns) -> list:
    found = []
    for pattern in patterns:
        for hit in sorted(glob.glob(os.path.join(directory, pattern), recursive=True)):
            rel = os.path.relpath(hit, directory)
            if any(part in _SKIP_DIR_PARTS for part in rel.split(os.sep)):
                continue
            if os.path.isfile(hit):
                found.append(hit)
    return found


def candidate_dirs_from_matrix() -> list:
    import yaml

    matrix_path = os.path.join(REPO_ROOT, "tests", "arch_matrix.yaml")
    if not os.path.isfile(matrix_path):
        return []
    with open(matrix_path) as f:
        entries = yaml.safe_load(f).get("models", [])

    dirs = set()
    for entry in entries:
        if entry.get("skip"):
            continue
        repo = entry.get("repo") or ""
        if repo.startswith("local:"):
            path = repo[len("local:"):]
        elif "/" in repo:
            org, _, name = repo.partition("/")
            path = os.path.join(MODEL_ROOT, org, name or org)
        else:
            continue
        if os.path.isdir(path):
            dirs.add(os.path.normpath(path))
    return sorted(dirs)


def candidate_dirs_from_mirror() -> list:
    """All dirs under MODEL_ROOT that look like a model dir (have config.json)."""
    dirs = set()
    if not os.path.isdir(MODEL_ROOT):
        return []
    # Model layout is <root>/<org>/<repo>/... ; depth cap keeps the walk cheap.
    for root, subdirs, files in os.walk(MODEL_ROOT):
        depth = os.path.relpath(root, MODEL_ROOT).count(os.sep)
        subdirs[:] = [
            s for s in subdirs if s not in _SKIP_DIR_PARTS and not s.startswith(".")
        ]
        if "config.json" in files and depth >= 1:
            dirs.add(os.path.normpath(root))
        if depth >= 4:
            subdirs[:] = []
    return sorted(dirs)


def should_convert(directory: str) -> bool:
    if not os.path.isfile(os.path.join(directory, "config.json")):
        return False
    if _find_files(directory, SAFETENSOR_PATTERNS):
        return False  # already has safetensors -> nothing to do
    return bool(_find_files(directory, SOURCE_PATTERNS))


def _unwrap_state_dict(loaded) -> OrderedDict:
    # Common wrappers: {'state_dict': {...}} or {'model': {...}, ...}.
    if isinstance(loaded, dict) and set(loaded) == {"state_dict"}:
        loaded = loaded["state_dict"]
    if isinstance(loaded, dict):
        tensors = OrderedDict()
        for key, value in loaded.items():
            if key in ("state_dict", "model") and isinstance(value, dict) \
                    and all(hasattr(v, "shape") for v in value.values()):
                for sub, tensor in value.items():
                    tensors[f"{key}.{sub}"] = tensor
            elif hasattr(value, "shape"):
                tensors[key] = value
        return tensors
    raise ValueError(f"unsupported checkpoint container: {type(loaded).__name__}")


def convert_directory(directory: str) -> dict:
    import torch
    import safetensors.torch as st

    sources = _find_files(directory, SOURCE_PATTERNS)
    state = OrderedDict()
    for path in sources:
        checkpoint = torch.load(path, map_location="cpu")
        state.update(_unwrap_state_dict(checkpoint))
        del checkpoint

    def _convertible(tensor) -> bool:
        dtype = tensor.dtype
        if tensor.is_complex() or not hasattr(tensor, "shape"):
            return False
        return dtype in (torch.float16, torch.float32, torch.float64,
                         torch.bfloat16, torch.float8_e4m3fn,
                         torch.float8_e5m2, torch.uint8, torch.int8,
                         torch.int16, torch.int32, torch.int64, torch.bool)

    tensors = [(name, tensor) for name, tensor in state.items()
               if _convertible(tensor)]
    dropped = [name for name, tensor in state.items()
               if not _convertible(tensor)]
    if not tensors:
        raise ValueError("no floating-point weight tensors found")

    # Tied weights (e.g. embed_tokens/lm_head) share one storage across several
    # keys. safetensors refuses aliased tensors, so clone every duplicate key so
    # the key set and values match the original .bin exactly.
    seen_storage = {}
    tie_replicated = []
    for index, (name, tensor) in enumerate(tensors):
        key = tensor.untyped_storage().data_ptr()
        if key in seen_storage:
            tensors[index] = (name, tensor.clone())
            tie_replicated.append(name)
        else:
            seen_storage[key] = name

    total = sum(t.numel() * t.element_size() for _, t in tensors)

    # Build sharding plan (a single tensor larger than SHARD_BYTES is kept whole).
    shards: list[list] = [[]]
    current = 0
    for name, tensor in tensors:
        size = tensor.numel() * tensor.element_size()
        if shards[-1] and current + size > SHARD_BYTES:
            shards.append([])
            current = 0
        shards[-1].append((name, tensor))
        current += size

    num_shards = len(shards)
    weight_map = {}
    written = []
    try:
        for index, shard in enumerate(shards, start=1):
            payload = {name: tensor.contiguous() for name, tensor in shard}
            out_path = os.path.join(
                directory, f"model-{index:05d}-of-{num_shards:05d}.safetensors"
            )
            st.save_file(payload, out_path)
            written.append(out_path)
            for name in payload:
                weight_map[name] = os.path.basename(out_path)
    except Exception:
        # Do not leave half-written shards behind: a retry must not mistake them
        # for a completed conversion (should_convert skips dirs with any *.safetensors).
        for out_path in written:
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise

    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(
            {"metadata": {"format": "pt"}, "weight_map": weight_map,
             "total_size": total},
            f,
            indent=1,
            sort_keys=True,
        )

    freed = 0
    for path in sources:
        freed += os.path.getsize(path)
        if not KEEP_BIN:
            os.remove(path)

    return {
        "shards": num_shards,
        "tensors": len(tensors),
        "bytes": total,
        "freed": freed if not KEEP_BIN else 0,
        "dropped": dropped,
        "tie_replicated": tie_replicated,
    }


def main() -> None:
    if not os.path.isdir(MODEL_ROOT):
        sys.exit(f"model root not found: {MODEL_ROOT}")

    candidates = set()
    candidates.update(candidate_dirs_from_matrix())
    candidates.update(candidate_dirs_from_mirror())
    todo = sorted(d for d in candidates if should_convert(d))

    if not todo:
        print(
            f"no model dirs with legacy .bin/.pth (and without safetensors) "
            f"under {MODEL_ROOT}"
        )
        return

    for directory in todo:
        print(f"[convert] {directory}")
        try:
            result = convert_directory(directory)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        print(
            f"  -> {result['tensors']} tensors ({gb(result['bytes'])}) in "
            f"{result['shards']} shards"
        )
        if result.get("tie_replicated"):
            print(
                f"  tie-replicated {len(result['tie_replicated'])} weight(s): "
                f"{result['tie_replicated'][:5]}"
            )
        if result["freed"]:
            print(f"  -> removed {gb(result['freed'])} of legacy weights")
        if result["dropped"]:
            print(f"  warning: skipped unsupported tensors: {result['dropped'][:8]}")
    print("done")


if __name__ == "__main__":
    main()
