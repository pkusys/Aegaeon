from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

GraphMode = Literal["off", "save", "load"]


@dataclass(frozen=True)
class CudaGraphConfig:
    mode: GraphMode = "off"
    # Root of the multi-model archive catalog. Model graphs and the shared
    # device-code runtime live in deterministic subdirectories below it.
    archive_dir: str = ""
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    seq_len_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    strict: bool = True

    # SAVE and LOAD must use the same GPU virtual address.
    base_address: int = 0x400000000000
    # The entire virtual address range available to Foundry.
    region_size_bytes: int = 256 * 1024**3
    # Isolates unstable CUDA/NCCL/cuBLAS initialization allocations.
    scratch_size_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if self.mode not in ("off", "save", "load"):
            raise ValueError("mode must be off, save, or load")
        if not isinstance(self.archive_dir, str):
            raise ValueError("archive_dir must be a string")
        if self.mode in ("save", "load") and not self.archive_dir:
            raise ValueError("archive_dir is required in save/load mode")
        if type(self.strict) is not bool:
            raise ValueError("strict must be a boolean")

        if not self.batch_sizes or any(
            type(size) is not int or size <= 0 for size in self.batch_sizes
        ):
            raise ValueError("batch_sizes must contain positive integers")
        if not self.seq_len_buckets or any(
            type(size) is not int or size <= 0 for size in self.seq_len_buckets
        ):
            raise ValueError("seq_len_buckets must contain positive integers")
        object.__setattr__(self, "batch_sizes", tuple(sorted(set(self.batch_sizes))))
        object.__setattr__(
            self,
            "seq_len_buckets",
            tuple(sorted(set(self.seq_len_buckets))),
        )

        alignment = 2 * 1024**2
        for name in ("base_address", "region_size_bytes", "scratch_size_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value % alignment != 0:
                raise ValueError(f"{name} must be 2 MiB aligned")

        if self.scratch_size_bytes >= self.region_size_bytes:
            raise ValueError(
                "scratch_size_bytes must be smaller than region_size_bytes"
            )

    @staticmethod
    def model_archive_key(model_name: str) -> str:
        """Return a filesystem-safe, reversible key for a public model name."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        return quote(model_name, safe="").replace(".", "%2E")

    def _archive_root(self) -> Path:
        if not self.archive_dir:
            raise ValueError("archive_dir is required to resolve archive paths")
        return Path(self.archive_dir).expanduser().resolve()

    def model_archive_dir(self, model_name: str) -> Path:
        """Return the catalog directory containing one model's graph files."""
        return self._archive_root() / "models" / self.model_archive_key(model_name)

    def runtime_archive_dir(self) -> Path:
        """Return the catalog directory for the shared Foundry device code."""
        return self._archive_root() / "runtime"

    @classmethod
    def from_dict(cls, raw: object) -> "CudaGraphConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("cuda_graph config must be a mapping")

        base_address = raw.get("base_address", cls.base_address)
        if isinstance(base_address, str):
            try:
                base_address = int(base_address, 0)
            except ValueError as exc:
                raise ValueError("base_address must be an integer address") from exc

        return cls(
            mode=raw.get("mode", cls.mode),
            archive_dir=raw.get("archive_dir", cls.archive_dir),
            batch_sizes=raw.get("batch_sizes", cls.batch_sizes),
            seq_len_buckets=raw.get("seq_len_buckets", cls.seq_len_buckets),
            strict=raw.get("strict", cls.strict),
            base_address=base_address,
            region_size_bytes=raw.get("region_size_bytes", cls.region_size_bytes),
            scratch_size_bytes=raw.get("scratch_size_bytes", cls.scratch_size_bytes),
        )
