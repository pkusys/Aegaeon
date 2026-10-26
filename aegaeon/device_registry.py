import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import yaml

from aegaeon.device_names import canonical_device_name
from aegaeon.logger import init_logger

logger = init_logger(__name__)

_PCIE_SPEED_GTPS = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0}


def _compute_pcie_bandwidth_gbs(gen: int, width: int) -> float:
    gtps = _PCIE_SPEED_GTPS.get(gen, 0)
    if gtps == 0:
        return 0.0
    encoding = 8 / 10 if gen <= 2 else 128 / 130
    return gtps * encoding * width / 8


def _detect_pcie_bandwidth_gbs(device_id: int = 0) -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
        width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
        return _compute_pcie_bandwidth_gbs(gen, width)
    except Exception as e:
        logger.warning("Failed to detect PCIe bandwidth: %s", e)
        return None


@dataclass
class DeviceSpec:
    name: str
    memory: float
    bandwidth: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, DeviceSpec):
            return NotImplemented
        return self.name == other.name

    def __str__(self):
        return self.name

    def mem_capacity_in_bytes(self) -> int:
        return int(self.memory * (2 ** 30))

    def pcie_bandwidth_in_bytes(self) -> float:
        if self.bandwidth is None:
            bw = _detect_pcie_bandwidth_gbs()
            if bw is None:
                raise ValueError(
                    f"Cannot determine PCIe bandwidth for device '{self.name}'. "
                    f"Add 'bandwidth' to the device config YAML, or ensure pynvml is available."
                )
            object.__setattr__(self, 'bandwidth', bw)
        return self.bandwidth * (2 ** 30)


class DeviceRegistry:
    def __init__(self, specs: List[DeviceSpec]):
        self._specs: List[DeviceSpec] = list(specs)
        self._by_name: Dict[str, DeviceSpec] = {}

        for spec in self._specs:
            if spec.name in self._by_name:
                raise ValueError(
                    f"Duplicate device name '{spec.name}' in registry"
                )
            self._by_name[spec.name] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    def __contains__(self, item):
        return item in self._specs

    def get_by_name(self, name: str) -> Optional[DeviceSpec]:
        return self._by_name.get(name)

    def from_str(self, s: str) -> Optional[DeviceSpec]:
        name = canonical_device_name(s)
        return next(
            (spec for spec in self._specs
             if canonical_device_name(spec.name) == name),
            None,
        )

    def detect_device(self, device_id: int = 0) -> Optional[DeviceSpec]:
        import torch
        try:
            name = torch.cuda.get_device_name(device_id)
        except Exception as e:
            logger.warning(
                "Failed to query CUDA device %d: %s", device_id, e
            )
            return None

        spec = self.from_str(name)
        if spec is not None:
            if spec.bandwidth is None:
                bw = _detect_pcie_bandwidth_gbs(device_id)
                if bw is not None:
                    object.__setattr__(spec, 'bandwidth', bw)
            return spec

        try:
            props = torch.cuda.get_device_properties(device_id)
            memory = props.total_memory / (2 ** 30)
        except Exception:
            memory = None

        if memory is None:
            logger.warning(
                "Cannot determine memory for unknown GPU '%s'. "
                "Add it to the device config YAML.", name
            )
            return None

        bandwidth = _detect_pcie_bandwidth_gbs(device_id)
        if bandwidth is None:
            logger.warning(
                "Unknown GPU '%s' — auto-detected memory=%.1f GiB, but bandwidth "
                "cannot be auto-detected. Add this GPU to the device config YAML "
                "with a 'bandwidth' field.", name, memory
            )
            return None

        spec = DeviceSpec(
            name=name,
            memory=memory,
            bandwidth=bandwidth,
        )
        self._by_name[spec.name] = spec
        self._specs.append(spec)
        logger.info(
            "Auto-detected unknown GPU '%s': memory=%.1f GiB, bandwidth=%.1f GB/s",
            name, memory, bandwidth,
        )
        return spec

    @classmethod
    def from_yaml(cls, path: str) -> "DeviceRegistry":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Device config YAML not found: {path}"
            )
        with open(path, "r") as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse YAML file '{path}': {e}"
                ) from e
        return cls.from_dict(raw, source=path)

    @classmethod
    def from_dict(cls, raw, source: str = "<content>") -> "DeviceRegistry":
        if raw is None or "devices" not in raw:
            return cls([])

        specs: List[DeviceSpec] = []
        for idx, entry in enumerate(raw["devices"]):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Device entry {idx} in '{source}' is not a mapping; "
                    f"each device entry must have 'name' and 'memory' fields"
                )
            name = entry.get("name")
            memory_raw = entry.get("memory")
            bandwidth_raw = entry.get("bandwidth")

            if not name:
                raise ValueError(
                    f"Device entry {idx} in '{source}' is missing required field 'name'"
                )
            if memory_raw is None:
                raise ValueError(
                    f"Device '{name}' in '{source}' is missing required field 'memory'"
                )

            extra = {
                k: v for k, v in entry.items()
                if k not in ("name", "memory", "bandwidth")
            }
            specs.append(DeviceSpec(
                name=str(name),
                memory=float(memory_raw),
                bandwidth=float(bandwidth_raw) if bandwidth_raw is not None else None,
                extra=extra,
            ))

        return cls(specs)

    @classmethod
    def default(cls) -> "DeviceRegistry":
        return cls(_DEFAULT_SPECS)


_DEFAULT_SPECS: List[DeviceSpec] = [
    DeviceSpec(name="A10", memory=24, bandwidth=32),
    DeviceSpec(name="A100", memory=80, bandwidth=32),
    DeviceSpec(name="V100", memory=16, bandwidth=16),
    DeviceSpec(name="H800", memory=80, bandwidth=64),
    DeviceSpec(name="H20", memory=96, bandwidth=32),
    DeviceSpec(name="H100", memory=80, bandwidth=32),
    DeviceSpec(name="A100_40GB", memory=39, bandwidth=32),
]


_GLOBAL_REGISTRY: Optional[DeviceRegistry] = None


def get_device_registry() -> DeviceRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        raise RuntimeError(
            "Device registry is not set. "
            "Call set_registry() or ensure AEGAEON_CONFIG_PATH is configured."
        )
    return _GLOBAL_REGISTRY


def get_registry() -> DeviceRegistry:
    """Backward-compatible alias for :func:`get_device_registry`."""
    return get_device_registry()


def set_registry(registry: DeviceRegistry) -> None:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry
