import json
from pathlib import Path

from aegaeon.graph_config import CudaGraphConfig
from aegaeon.graph_bindings import (
    ATTENTION_POLICY,
    LAYOUT_ABI_VERSION,
    cuda_tensor_bindings,
    validate_bindings,
)


class FoundryRuntime:
    def __init__(self, graph_config: CudaGraphConfig) -> None:
        self.graph_config = graph_config
        self.active_model_name: str | None = None
        self.model_layout_start_offset: int | None = None
        self._model_layout_prepared = False

    def activate_model(self, model_name: str) -> None:
        """Select the model whose graph archive subsequent operations use."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        normalized_name = model_name.strip()
        # Keep model-name validation and on-disk encoding in one place.
        self.graph_config.model_archive_key(normalized_name)
        self.active_model_name = normalized_name

    def deactivate_model(self) -> None:
        """Clear the active model at a model-lifetime boundary."""
        self.active_model_name = None

    def _require_active_model_name(self) -> str:
        if self.active_model_name is None:
            raise RuntimeError(
                "Foundry model archive requested before a model was activated"
            )
        return self.active_model_name

    def active_model_archive_dir(self) -> Path:
        """Return the catalog directory for the currently active model."""
        return self.graph_config.model_archive_dir(self._require_active_model_name())

    def model_rank_archive_dir_for(
        self,
        model_name: str,
        rank: int,
    ) -> Path:
        """Return one rank's graph directory without changing the active model."""
        return self.graph_config.model_archive_dir(model_name) / f"rank_{rank}"

    def model_rank_archive_dir(self, rank: int) -> Path:
        """Return one rank's graph directory for the active model."""
        return self.model_rank_archive_dir_for(
            self._require_active_model_name(),
            rank,
        )

    def runtime_rank_archive_dir(self, rank: int) -> Path:
        """Return one rank's shared Foundry device-code directory."""
        return self.graph_config.runtime_archive_dir() / f"rank_{rank}"

    def _layout_path(self, rank: int) -> Path:
        return self.rank_archive_dir(rank) / "vmm_layout.json"

    def rank_archive_dir(self, rank: int) -> Path:
        """Return the active model's per-rank graph archive directory."""
        return self.model_rank_archive_dir(rank)

    def set_up_before_distributed(
        self,
        rank: int,
        model_names: list[str],
    ) -> None:
        from foundry import ops

        if self.graph_config.mode == "load":
            unique_model_names = sorted(set(model_names))
            if not unique_model_names:
                raise RuntimeError(
                    "CUDA Graph LOAD mode requires at least one configured model"
                )

            required_files = (
                "fatbin_image_packed.img",
                "fatbin_entrypoint_packed.txt",
                "vmm_layout.json",
            )
            archive_dirs: list[Path] = []
            for model_name in unique_model_names:
                archive_dir = self.model_rank_archive_dir_for(model_name, rank)
                for filename in required_files:
                    required_path = archive_dir / filename
                    if not required_path.is_file():
                        raise RuntimeError(
                            f"Foundry archive for model {model_name!r}, "
                            f"rank {rank} is incomplete: {required_path}"
                        )
                archive_dirs.append(archive_dir)
                with (archive_dir / "vmm_layout.json").open(
                    "r", encoding="utf-8"
                ) as file:
                    self._validate_layout_abi(json.load(file))

            # Loading modules creates the CUDA context. Foundry requires this
            # to happen before reserving the fixed-address VMM region.
            ops.set_skip_fatbin_processing(True)
            ops.load_cuda_modules_and_libraries_from_archives(
                [str(path) for path in archive_dirs]
            )

        ops.set_allocation_region(
            self.graph_config.base_address, self.graph_config.region_size_bytes
        )

    def set_up_after_distributed(self) -> None:
        """Move deterministic allocations past the NCCL scratch area."""
        from foundry import ops

        current_offset = ops.get_current_alloc_offset()
        scratch_boundary = self.graph_config.scratch_size_bytes
        if current_offset > scratch_boundary:
            message = (
                "Foundry allocations made during distributed initialization "
                f"already exceed the scratch boundary: {current_offset} > "
                f"{scratch_boundary} bytes"
            )
            if self.graph_config.strict:
                raise RuntimeError(message)
            # Non-strict mode preserves allocations that legitimately exceeded
            # the configured scratch estimate. This real post-initialization
            # high-water mark is the common start of every model layout.
            self.model_layout_start_offset = current_offset
            return

        ops.set_current_alloc_offset(scratch_boundary)
        self.model_layout_start_offset = scratch_boundary

    def prepare_model_layout(self, rank: int) -> None:
        from foundry import ops

        ops.resume_allocation_region()
        if self.graph_config.mode != "load":
            return

        metadata = self._read_layout(rank)
        self._validate_layout_abi(metadata)

        # Each model archive records an independent layout starting at the
        # common post-initialization boundary. A previous model leaves the
        # process-wide Foundry cursor at its own final offset; using that value
        # here would preallocate only the delta between two unrelated layouts.
        # The first prepare happens after all backend imports, so it is the
        # authoritative initialization high-water mark. On later prepares,
        # release the old physical mapping only after Worker._reset() has
        # destroyed its CUDA graphs/tensors. Re-setting the same allocation
        # region is intentionally idempotent in Foundry and resets its cursor;
        # advance it to the saved high-water mark so persistent NCCL/import
        # allocations are not overwritten.
        if self.model_layout_start_offset is None:
            raise RuntimeError(
                "Foundry model layout prepared before distributed setup completed"
            )
        current_offset = ops.get_current_alloc_offset()
        if not self._model_layout_prepared:
            self.model_layout_start_offset = max(
                self.model_layout_start_offset,
                current_offset,
            )
            self._model_layout_prepared = True
        else:
            ops.free_preallocated_region()
            ops.set_allocation_region(
                self.graph_config.base_address,
                self.graph_config.region_size_bytes,
            )
            ops.set_current_alloc_offset(self.model_layout_start_offset)

        layout_path = self._layout_path(rank)
        metadata = self._read_layout(rank)
        target_offset_value = metadata.get("final_alloc_offset")
        if target_offset_value is None:
            target_offset_value = metadata["model_layout_end_offset"]
        target_offset = int(target_offset_value)
        current_offset = ops.get_current_alloc_offset()
        preallocate_size = target_offset - current_offset

        if preallocate_size < 0:
            raise RuntimeError(
                "Foundry current offset exceeds saved allocation layout: "
                f"current={current_offset}, saved={target_offset}"
            )

        if preallocate_size > 0:
            success = ops.preallocate_region(preallocate_size)
            if not success:
                raise RuntimeError(
                    "Foundry failed to preallocate model layout: "
                    f"size={preallocate_size}"
                )

    def _read_layout(self, rank: int) -> dict:
        with self._layout_path(rank).open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _validate_layout_abi(metadata: dict) -> None:
        if (
            metadata.get("layout_abi_version") != LAYOUT_ABI_VERSION
            or metadata.get("attention_policy") != ATTENTION_POLICY
        ):
            raise RuntimeError(
                "Foundry archive uses an obsolete layout/attention ABI; "
                "re-record this model in SAVE mode before LOAD"
            )

    def _write_layout(self, rank: int, metadata: dict) -> None:
        path = self._layout_path(rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")

    def finish_model_layout(self, rank: int, tensors=None) -> int:
        from foundry import ops

        end_offset = ops.get_current_alloc_offset()
        if self.graph_config.mode == "save":
            layout_path = self._layout_path(rank)
            layout_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "layout_abi_version": LAYOUT_ABI_VERSION,
                "attention_policy": ATTENTION_POLICY,
                "base_address": self.graph_config.base_address,
                "region_size_bytes": self.graph_config.region_size_bytes,
                "scratch_size_bytes": self.graph_config.scratch_size_bytes,
                "model_layout_end_offset": end_offset,
                "model_bindings": cuda_tensor_bindings(tensors),
                "decode_graph_inputs": {},
            }
            with layout_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)
                file.write("\n")
        elif self.graph_config.mode == "load":
            metadata = self._read_layout(rank)
            if end_offset != metadata["model_layout_end_offset"]:
                raise RuntimeError(
                    "Foundry model layout mismatch: "
                    f"actual={end_offset}, saved={metadata['model_layout_end_offset']}"
                )
            validate_bindings(
                metadata["model_bindings"], cuda_tensor_bindings(tensors), "model/KV"
            )
        ops.stop_allocation_region()
        return end_offset

    def begin_graph_inputs(self, rank: int, graph_name: str) -> int:
        """Reserve the recorded workspace gap before creating static inputs.

        BLAS getters use PyTorch's allocator; their allocation count need not
        repeat across model switches. The entire saved VMM extent is mapped,
        so reserve the recorded gap explicitly, then verify every tensor.
        """
        from foundry import ops

        current = ops.get_current_alloc_offset()
        if self.graph_config.mode == "save":
            return current
        metadata = self._read_layout(rank)
        saved = metadata["decode_graph_inputs"][graph_name]["start_offset"]
        if current > saved:
            raise RuntimeError(
                f"Foundry static-input start exceeded: actual={current}, saved={saved}"
            )
        if current < saved:
            ops.set_current_alloc_offset(saved)
        return saved

    def finish_graph_inputs(
        self, rank: int, graph_name: str, start: int, tensors
    ) -> None:
        from foundry import ops

        metadata = self._read_layout(rank)
        entry = {
            "start_offset": start,
            "end_offset": ops.get_current_alloc_offset(),
            "bindings": cuda_tensor_bindings(tensors),
        }
        if self.graph_config.mode == "save":
            metadata["decode_graph_inputs"][graph_name] = entry
            self._write_layout(rank, metadata)
        else:
            saved = metadata["decode_graph_inputs"][graph_name]
            if entry["end_offset"] != saved["end_offset"]:
                raise RuntimeError(
                    f"Foundry static-input end mismatch for {graph_name}: actual={entry['end_offset']}, saved={saved['end_offset']}"
                )
            validate_bindings(saved["bindings"], entry["bindings"], graph_name)

    def validate_final_offset(self, rank: int) -> None:
        from foundry import ops

        saved = self._read_layout(rank)["final_alloc_offset"]
        actual = ops.get_current_alloc_offset()
        if actual != saved:
            raise RuntimeError(
                f"Foundry final layout mismatch: actual={actual}, saved={saved}"
            )

    def prepare_graph_workspaces(self, stream) -> None:
        """Initialize BLAS workspaces; static-input placement is restored separately."""
        import torch
        import foundry
        from foundry import ops

        foundry.resume_allocation_region()
        try:
            ops.preallocate_cublas_workspaces()
            with torch.cuda.stream(stream):
                ops.preallocate_cublas_workspaces()
            stream.synchronize()
        finally:
            foundry.stop_allocation_region()

    def finish_graph_archive(self, rank: int) -> int:
        """Persist the post-capture VMM high-water mark and stop allocation."""
        from foundry import ops

        final_offset = ops.get_current_alloc_offset()
        layout_path = self._layout_path(rank)
        with layout_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["final_alloc_offset"] = final_offset
        with layout_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        ops.stop_allocation_region()
        return final_offset
