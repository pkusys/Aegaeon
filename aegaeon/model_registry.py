import glob
import os
import re
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import yaml

from aegaeon.logger import init_logger

logger = init_logger(__name__)

_B_TO_BYTES = 2**30


@dataclass
class ModelSpec:
    name: str
    model_path: str = ""
    params: Optional[float] = None
    profile: str = ""
    max_model_len: Optional[int] = None
    id: Optional[int] = None
    tool_parser: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, ModelSpec):
            return NotImplemented
        return self.name == other.name

    def lb_key(self) -> int:
        return self.id

    def __str__(self):
        return self.name

    def profile_key(self) -> str:
        """Return the filesystem key used for this model's profile data.

        The public API model name is not necessarily a valid profile directory
        name (for example ``Qwen/Qwen3-4B``).  Config files may therefore set a
        stable ``profile`` alias without changing the externally visible model
        name.  Dynamically deployed Hugging Face model IDs are normalized to a
        deterministic alias, e.g. ``Qwen/Qwen3.5-9B`` -> ``qwen3_5_9b``.
        """
        configured = self.profile or None
        if configured is None:
            model_id = self.name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            value = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
        else:
            value = str(configured).strip()
        if not value:
            raise ValueError(f"Model '{self.name}' has an empty profile alias")
        if os.path.isabs(value) or ".." in value.replace("\\", "/").split("/"):
            raise ValueError(
                f"Model '{self.name}' has an unsafe profile alias: {value!r}"
            )
        return value

    def path(self) -> str:
        if not self.model_path:
            from aegaeon.model_downloader import download_model

            path = download_model(self.name)
            object.__setattr__(self, "model_path", path)
        return self.model_path

    def n_params(self) -> float:
        from aegaeon.model_registry import get_registry

        return get_registry().get_or_compute_params(self)


class ModelSpecRegistry:
    def __init__(self, specs: List[ModelSpec]):
        self._specs: List[ModelSpec] = list(specs)
        self._configured_names = frozenset(spec.name for spec in self._specs)
        self._by_name: Dict[str, ModelSpec] = {}
        self._by_path: Dict[str, ModelSpec] = {}
        self._computed_params_cache: Dict[str, float] = {}

        _validate_and_assign_ids(self._specs)

        for spec in self._specs:
            if spec.name in self._by_name:
                raise ValueError(f"Duplicate model name '{spec.name}' in registry")
            self._by_name[spec.name] = spec
            self._by_path[spec.model_path] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    def __contains__(self, item):
        return item in self._specs

    def get_by_name(self, name: str) -> Optional[ModelSpec]:
        return self._by_name.get(name)

    def get_by_path(self, path: str) -> Optional[ModelSpec]:
        return self._by_path.get(path)

    def from_str(self, s: str, default=None) -> Optional[ModelSpec]:
        spec = self._by_name.get(s)
        if spec is not None:
            return spec
        return default

    def get_or_compute_params(self, spec: ModelSpec) -> float:
        if spec.params is not None:
            return spec.params
        if spec.model_path in self._computed_params_cache:
            return self._computed_params_cache[spec.model_path]
        computed = _compute_params(spec.model_path)
        if computed is not None:
            self._computed_params_cache[spec.model_path] = computed
            spec.params = computed
            return computed
        raise ValueError(
            f"Model '{spec.name}' has no params specified and auto-computation "
            f"failed for path '{spec.model_path}'. Please add 'params' in the YAML config."
        )

    def estimate_cache_size_bytes(
        self, models: List[ModelSpec], overhead: float = 1.2
    ) -> int:
        total = 0
        for spec in models:
            total += self.get_or_compute_params(spec)
        return int(total * 2 * overhead)  # bf16/fp16 = 2 bytes per param

    @classmethod
    def from_yaml(cls, path: str) -> "ModelSpecRegistry":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model config YAML not found: {path}")
        with open(path, "r") as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse YAML file '{path}': {e}") from e
        return cls.from_dict(raw, source=path)

    @classmethod
    def from_dict(cls, raw, source: str = "<content>") -> "ModelSpecRegistry":
        raw_models = [] if raw is None else raw.get("models", [])
        specs: List[ModelSpec] = []
        for idx, entry in enumerate(raw_models):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Entry {idx} in '{source}' is not a mapping; "
                    f"each model entry must have a 'name' field"
                )
            id_raw = entry.get("id")
            name = entry.get("name")
            model_path = entry.get("path", "") or ""

            if not name:
                raise ValueError(
                    f"Entry {idx} in '{source}' is missing required field 'name'"
                )

            params_raw = entry.get("params")
            params: Optional[float] = None
            if params_raw is not None:
                if not isinstance(params_raw, (int, float)):
                    raise ValueError(
                        f"Entry '{name}' in '{source}': 'params' must be a number, got {type(params_raw).__name__}"
                    )
                params = float(params_raw) * _B_TO_BYTES
            
            profile = str(entry.get("profile", ""))

            tool_parser_raw = entry.get("tool_parser")
            tool_parser: Optional[str] = None
            if tool_parser_raw is not None:
                if not isinstance(tool_parser_raw, str) or not tool_parser_raw.strip():
                    raise ValueError(
                        f"Entry '{name}' in '{source}': 'tool_parser' must be a non-empty string"
                    )
                tool_parser = tool_parser_raw.strip()

            max_model_len_raw = entry.get("max_model_len")
            max_model_len: Optional[int] = None 
            if max_model_len_raw is not None:
                if not isinstance(max_model_len_raw, int):
                    raise ValueError(
                        f"Entry '{name}' in '{source}': 'max_model_len' must be an integer"
                    )
                max_model_len = int(max_model_len_raw)

            specs.append(ModelSpec(
                name=str(name),
                model_path=str(model_path),
                params=params,
                profile=profile,
                tool_parser=tool_parser,
                max_model_len=max_model_len,
                id=int(id_raw) if id_raw is not None else None,
            ))

        return cls(specs)

    @classmethod
    def default(cls) -> "ModelSpecRegistry":
        return cls(_DEFAULT_SPECS)

    def register_dynamic(self, spec: ModelSpec):
        """Register a dynamically-deployed model into the registry."""
        if spec.name in self._by_name:
            existing = self._by_name[spec.name]
            if spec.model_path and spec.model_path != existing.model_path:
                if self._by_path.get(existing.model_path) is existing:
                    self._by_path.pop(existing.model_path, None)
                existing.model_path = spec.model_path
                self._by_path[spec.model_path] = existing
            if spec.max_model_len is not None:
                existing.max_model_len = spec.max_model_len
            if spec.tool_parser is not None:
                existing.tool_parser = spec.tool_parser
            return existing
        if spec.id is None:
            spec.id = hash(spec.name) & 0x7FFFFFFF
        self._specs.append(spec)
        self._by_name[spec.name] = spec
        if spec.model_path:
            self._by_path[spec.model_path] = spec
        return spec

    def unregister_dynamic(self, model_name: str) -> Optional[ModelSpec]:
        """Remove runtime-only entries, retaining configured model metadata."""
        if model_name in self._configured_names:
            return None
        spec = self._by_name.pop(model_name, None)
        if spec is None:
            return None
        self._specs = [s for s in self._specs if s.name != model_name]
        if spec.model_path:
            self._by_path.pop(spec.model_path, None)
        return spec


# Backward-compatible name used by the group-aware cache branch and older
# profiling/extension code. Keep ModelSpecRegistry as the canonical name so it
# is not confused with vLLM's model-class registry.
ModelRegistry = ModelSpecRegistry


def _validate_and_assign_ids(specs: List[ModelSpec]) -> None:
    ids_present = [spec.id is not None for spec in specs]
    if not any(ids_present):
        for i, spec in enumerate(specs):
            spec.id = i
    elif not all(ids_present):
        missing = [
            specs[i].name for i, present in enumerate(ids_present) if not present
        ]
        raise ValueError(
            f"Mixed 'id' configuration: some models have an 'id' "
            f"while others do not. Either add 'id' to all models or remove it "
            f"entirely. Models missing 'id': {missing}"
        )


def _compute_params(model_path: str) -> Optional[float]:
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        try:
            import safetensors

            total: int = 0
            for sf_file in safetensor_files:
                with safetensors.safe_open(sf_file, framework="pt") as sf:
                    for key in sf.keys():
                        shape = sf.get_slice(key).get_shape()
                        total += math.prod(shape)
            return float(total)
        except Exception as e:
            logger.warning(
                "Auto-computation of params via safetensors failed for '%s': %s",
                model_path,
                e,
            )
    return None


_DEFAULT_SPECS: List[ModelSpec] = [
    ModelSpec(name="qwen2_1_5b", model_path="/srv/models/Qwen/Qwen2.5-1.5B-Instruct/", params=1.67 * _B_TO_BYTES),
    ModelSpec(name="qwen2_5_7b", model_path="/srv/models/Qwen/Qwen2.5-7B-Instruct/", params=7.1 * _B_TO_BYTES),
    ModelSpec(name="qwen_7b", model_path="/srv/models/Qwen-7B/",
              params=7.1 * _B_TO_BYTES),
    ModelSpec(name="Qwen/Qwen2.5-7B-Instruct", model_path="/srv/models/Qwen/Qwen2.5-7B-Instruct/",
              params=7.1 * _B_TO_BYTES,
              profile="qwen2_5_7b"),
    ModelSpec(name="qwen_7b_chat", model_path="/srv/models/Qwen-7B-Chat/",
              params=7.1 * _B_TO_BYTES),
    ModelSpec(name="llama2_7b", model_path="/srv/models/Llama-2-7b-hf/",
              params=6.3 * _B_TO_BYTES),
    ModelSpec(name="yi1_5_6b_chat", model_path="/srv/models/Yi-1.5-6B-Chat/",
              params=5.75 * _B_TO_BYTES),
    ModelSpec(name="qwen1_5_14b_chat", model_path="/srv/models/Qwen1.5-14B-Chat/",
              params=13.25 * _B_TO_BYTES),
    ModelSpec(name="llama2_13b_chat", model_path="/srv/models/Llama/Llama-2-13b-chat-ms/",
              params=13.2 * _B_TO_BYTES),
    ModelSpec(name="qwen_14b_chat", model_path="/srv/models/Qwen-14B-Chat/",
              params=13.25 * _B_TO_BYTES),
    ModelSpec(name="internlm2_5_7b_chat", model_path="/srv/models/internlm2_5-7b-chat/",
              params=7.1 * _B_TO_BYTES),
    ModelSpec(name="llava1_5_13b", model_path="/srv/models/llava-1.5-13b-hf/",
              params=12.5 * _B_TO_BYTES),
    ModelSpec(name="qwen1_5_moe_a2_7b_chat", model_path="/srv/models/Qwen1.5-MoE-A2.7B-Chat/",
              params=13.4 * _B_TO_BYTES),
    ModelSpec(name="yi1_5_9b_chat", model_path="/srv/models/Yi-1.5-9B-Chat/",
              params=8.25 * _B_TO_BYTES),
    ModelSpec(name="Qwen/Qwen2.5-1.5B-Instruct", model_path="/srv/models/Qwen/Qwen2.5-1.5B-Instruct/",
              params=1.67 * _B_TO_BYTES,
              profile="Qwen2.5-1.5B-Instruct"),
    ModelSpec(name="qwen2_5_72b", model_path="/srv/models/Qwen2.5-72B-Instruct/",
              params=72 * _B_TO_BYTES),
    ModelSpec(name="deepseek_v2_lite", model_path="/srv/models/DeepSeek-V2-Lite/",
              params=15.71 * _B_TO_BYTES),
    ModelSpec(name="Qwen/Qwen3-4B", model_path="/srv/models/Qwen/Qwen3-4B/",
              params=4.1 * _B_TO_BYTES,
              profile="qwen3_4b"),
]


_GLOBAL_REGISTRY: Optional[ModelSpecRegistry] = None


def get_registry() -> ModelSpecRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        raise RuntimeError(
            "Model registry is not set. "
            "Call set_registry() or ensure AEGAEON_CONFIG_PATH is configured."
        )
    return _GLOBAL_REGISTRY


def set_registry(registry: ModelSpecRegistry) -> None:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry
