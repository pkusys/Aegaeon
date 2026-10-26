"""GPU name matching shared by runtime detection and deployment checks."""

_CUDA_ALIASES = {
    "nvidia a10": "a10",
    "nvidia a100": "a100",
    "nvidia a100-sxm4-80gb": "a100",
    "nvidia a100-pcie-40gb": "a100_40gb",
    "nvidia v100": "v100",
    "nvidia h800": "h800",
    "nvidia h20": "h20",
    "nvidia h100": "h100",
    "nvidia h100 80gb hbm3": "h100",
    "nvidia gb10": "gb10",
}


def canonical_device_name(name: str) -> str:
    normalized = name.strip().casefold()
    return _CUDA_ALIASES.get(normalized, normalized)
