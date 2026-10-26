import asyncio
import itertools
import json
import os
import time
import traceback
from threading import Lock
from typing import Any, Optional

import ray
import ray.exceptions
import torch
from ray.util.placement_group import PlacementGroup
from viztracer import VizTracer

from aegaeon.block_manager import BlockManager
from aegaeon.config import (
    NodeConfig,
    ParallelConfig,
    QuickLoaderConfig,
    get_model_config,
    get_server_config,
    register_model_config,
)
from aegaeon.decode_dispatcher import DecodeDispatcher
from aegaeon.lifetime import LifetimeEvent, LifetimeEventType
from aegaeon.graph_config import CudaGraphConfig
from aegaeon.loader import QuickCache
from aegaeon.loader.meta import ModelMeta
from aegaeon.logger import init_logger
from aegaeon.model_placement import (
    LeastLoadedPlacementPolicy,
    LeastOutstandingRequestsPolicy,
    ModelNotReadyError,
    ModelPlacementPolicy,
    ModelPlacementTable,
    ModelReplicaState,
    NodePlacementMetrics,
    RequestRoutingPolicy,
    RequestRoutingTable,
)
from aegaeon.model_registry import ModelSpec
from aegaeon.prefill_dispatcher import PrefillDispatcher
from aegaeon.request import Request, RequestCapacityError
from aegaeon.stage_engine import DecodeEngine, PrefillEngine, SimpleEngine, StepOutput
from aegaeon.utils import (
    TTFT_SLO,
    Counter,
    CudaRTLibrary,
    compute_request_latencies,
    compute_request_metrics,
    ensure_outfile,
)

logger = init_logger(__name__)

_SHM_DIR = os.environ.get("AEGAEON_SHM_DIR", "/dev/shm")
AEGAEON_CPU_CACHE_FILENAME = os.path.join(_SHM_DIR, "aegaeon_cpu_cache")

AEGAEON_MODEL_CACHE_FILENAME = os.path.join(_SHM_DIR, "aegaeon_model_cache")


_RUNTIME_ENV_KEYS = (
    "FAST_SWITCH",
    "AEGAEON_CONFIG_PATH",
    "AEGAEON_CONFIG_CONTENT",
    "AEGAEON_MODEL_DOWNLOAD_PROXY",
    "AEGAEON_MODEL_ROOT",
    "AEGAEON_MODEL_ROOTS",
    "AEGAEON_PROFILE_ROOT",
    "AEGAEON_DEFAULT_MODEL_REPO",
    "AEGAEON_SHM_DIR",
    "AEGAEON_CUDA_VISIBLE_DEVICES",
    "AEGAEON_CUDA_COMPAT_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "AEGAEON_DUMMY_WEIGHTS",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
    "AEGAEON_FOUNDRY_HOOK_PATH",
)


def _default_cuda_compat_paths() -> tuple[str, ...]:
    # A compatibility driver for an older CUDA wheel may break a newer wheel.
    # Probe only the version used by PyTorch in this Worker environment.
    cuda_version = torch.version.cuda
    if not cuda_version:
        return ()
    return (
        f"/usr/local/cuda-{cuda_version}/compat",
        f"~/.local/cuda-{cuda_version}/compat",
    )


def _foundry_worker_env(
    base_env: dict[str, str],
    cuda_graph_config: CudaGraphConfig | None = None,
    enable_foundry: bool = False,
) -> dict[str, str]:
    """Prepare CUDA compatibility and Foundry preload settings for a Worker.

    Dynamic-linker settings must be part of Ray's runtime environment because
    changing ``LD_PRELOAD`` after the Worker process starts is too late.
    """
    env_vars = dict(base_env)
    configured_path = env_vars.get("AEGAEON_CUDA_COMPAT_PATH")
    compat_path = os.path.expanduser(configured_path) if configured_path else None
    if compat_path is None:
        compat_path = next(
            (
                os.path.expanduser(path)
                for path in _default_cuda_compat_paths()
                if os.path.isdir(os.path.expanduser(path))
            ),
            None,
        )
    if configured_path and not os.path.isdir(compat_path):
        raise ValueError(
            "AEGAEON_CUDA_COMPAT_PATH does not exist or is not a directory: "
            f"{compat_path}"
        )
    if compat_path is not None:
        current_paths = [
            path
            for path in env_vars.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if path
        ]
        env_vars["LD_LIBRARY_PATH"] = os.pathsep.join(
            [compat_path] + [path for path in current_paths if path != compat_path]
        )

    if not enable_foundry or cuda_graph_config is None:
        return env_vars
    if cuda_graph_config.mode == "off":
        return env_vars
    configured_hook = env_vars.get("AEGAEON_FOUNDRY_HOOK_PATH")
    if not configured_hook:
        raise ValueError(
            "AEGAEON_FOUNDRY_HOOK_PATH is required when CUDA Graph mode is active"
        )
    hook_path = os.path.abspath(os.path.expanduser(configured_hook))
    if not os.path.isabs(configured_hook) or not os.path.isfile(hook_path):
        raise ValueError(
            "AEGAEON_FOUNDRY_HOOK_PATH must be an existing absolute file: "
            f"{hook_path}"
        )

    current_preloads = [
        path for path in env_vars.get("LD_PRELOAD", "").split(os.pathsep) if path
    ]
    env_vars["LD_PRELOAD"] = os.pathsep.join(
        [hook_path] + [path for path in current_preloads if path != hook_path]
    )
    env_vars["FOUNDRY_MODE"] = cuda_graph_config.mode
    # Keep one cuBLASLt workspace per device. Foundry preallocates this
    # workspace inside its deterministic VMM region before graph capture/load.
    env_vars["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = "1"
    env_vars["FOUNDRY_EARLY_RESERVE_BASE"] = hex(cuda_graph_config.base_address)
    env_vars["FOUNDRY_EARLY_RESERVE_SIZE"] = str(
        cuda_graph_config.region_size_bytes
    )
    return env_vars


def _node_runtime_env(base_env: dict[str, str], config: NodeConfig) -> dict[str, str]:
    """Return the environment assigned to one Controller and its Workers."""
    env_vars = dict(base_env)
    if config.cuda_visible_devices is not None:
        if not config.cuda_visible_devices.strip():
            raise ValueError(f"{config.node_id}.cuda_visible_devices must not be empty")
        env_vars["AEGAEON_CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    if config.shm_dir is not None:
        if not os.path.isabs(config.shm_dir):
            raise ValueError(f"{config.node_id}.shm_dir must be an absolute path")
        env_vars["AEGAEON_SHM_DIR"] = config.shm_dir
    return env_vars


def _get_worker_num_cpus():
    from aegaeon.config import get_server_config

    return get_server_config().worker_num_cpus


class LLMService:
    """
    LLMService: Implements LLM inference service in Aegaeon.

    ## Overview

    This class serves as the endpoint API for sending `generate()` requests to
    the managed cluster of nodes with stage engines and workers.

              enqueue                       select
    Endpoint ---------> PrefillDispatcher  -------> PrefillEngine
                                                        |
                                                        | prefilled
                                            select      v
               (Done) <------ DecodeEngine <------- DecodeDispatcher
                                    |                   ^
                                    |-------------------|
                                   (optional) load-balance

    [Data Plane]    The data plane of Aegaeon is managed by StageEngines
                    separately using Ray. Each StageEngine controls a collection
                    of Workers as ray.Actors which occupy GPUs and perform
                    LLM inference asynchronously.
                    The per-node BlockManagers act as central managers of KV cache,
                    coordinating block movement between cache spaces on a node.

                    NOTE: currently, direct block movement between GPUs are not
                    implemented (termed "migration"). Instead, the CPU cache
                    is used explicitly as a staging buffer.

    [Control Plane] The control plane of Aegaeon is based on `asyncio` and is
                    also disaggregated with per-node Controllers. Each Controller
                    corresponds to a GPU node and may consist of the following entities
                    and their event loops:

                    - PrefillDispatcher: A unique entity that queues and dispatches
                        new requests to PrefillEngines.
                    - PrefillEngine: An entity with Worker(s) that executes the
                        Prefill stage of a request.
                    - DecodeDispatcher: A unique entity that queues and (re)dispatches
                        prefilled requests to DecodeEngines.
                    - DecodeEngine: An entity with Worker(s) that executes the
                        Decode stage of a request.

                    All event loops are non-blocking; data synchronization
                    is achieved via CUDA events dispatched to Workers and thus
                    happen in the data plane only when necessary. See the header
                    comment in `block_manager.py` for details.

                    The full control plane is implemented with the Controllers as
                    async ray actors, scaling up to arbirarily many nodes.
    """

    def __init__(
        self,
        cluster_config: list[NodeConfig],
        model_placement_policy: ModelPlacementPolicy | None = None,
        request_routing_policy: RequestRoutingPolicy | None = None,
    ) -> None:
        from aegaeon.config import get_server_config

        if get_server_config().dummy_weights:
            os.environ["AEGAEON_DUMMY_WEIGHTS"] = "1"

        self.counter = Counter()
        self.model_placement_policy = (
            model_placement_policy or LeastLoadedPlacementPolicy()
        )
        self.last_model_placement_decision: dict[str, Any] | None = None
        self.request_routing_policy = (
            request_routing_policy or LeastOutstandingRequestsPolicy()
        )
        self.request_routing = RequestRoutingTable(self.request_routing_policy)
        self._routing_lock = Lock()
        self._model_operation_locks: dict[str, asyncio.Lock] = {}
        self._model_placement_stats_lock = Lock()
        self._model_placement_stats: dict[
            str, dict[str, dict[str, int]]
        ] = {}

        # TODO: this should be a per-node setting
        env_vars = {"FAST_SWITCH": "1"}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["AEGAEON_CUDA_VISIBLE_DEVICES"] = os.environ[
                "CUDA_VISIBLE_DEVICES"
            ]
        if "AEGAEON_LOG_FILE" in os.environ:
            env_vars["AEGAEON_LOG_FILE"] = os.environ["AEGAEON_LOG_FILE"]
        if "AEGAEON_CONFIG_PATH" in os.environ:
            env_vars["AEGAEON_CONFIG_PATH"] = os.environ["AEGAEON_CONFIG_PATH"]
            with open(env_vars["AEGAEON_CONFIG_PATH"]) as _f:
                env_vars["AEGAEON_CONFIG_CONTENT"] = _f.read()
        for key in (
            "AEGAEON_MODEL_DOWNLOAD_PROXY",
            "AEGAEON_MODEL_ROOT",
            "AEGAEON_MODEL_ROOTS",
            "AEGAEON_PROFILE_ROOT",
            "AEGAEON_DEFAULT_MODEL_REPO",
            "AEGAEON_SHM_DIR",
            "AEGAEON_CUDA_COMPAT_PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "AEGAEON_DUMMY_WEIGHTS",
            "AEGAEON_FOUNDRY_HOOK_PATH",
        ):
            if key in os.environ:
                env_vars[key] = os.environ[key]

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            if key in os.environ:
                env_vars[key] = os.environ[key]

        # Initialize per-node
        self.nodes: list[ray.ObjectRef[Controller]] = []
        self.node_ids = [config.node_id for config in cluster_config]
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("NodeConfig.node_id values must be unique")

        init_handles = []
        for i, config in enumerate(cluster_config):
            node_env_vars = _node_runtime_env(env_vars, config)
            node = (
                ray.remote(Controller)
                .options(
                    name=f"node_{i}",
                    runtime_env={"env_vars": node_env_vars},
                    resources={config.node_id: 0.01},
                )
                .remote()
            )
            init_handle = node.init.remote(config)
            self.nodes.append(node)
            init_handles.append(init_handle)
        self.nodes_by_id = dict(zip(self.node_ids, self.nodes))
        # ..the node must first finish initialize()
        init_results = ray.get(init_handles)
        self.model_placements = ModelPlacementTable.from_ready_models(
            {
                result["node_id"]: result["cached_models"]
                for result in init_results
            }
        )

        run_handles = []
        for node in self.nodes:
            run_handle = node.start_event_loops.remote()
            run_handles.append(run_handle)

        # ..and this should return the timeout exception if event loops are healthy
        def sanity_check():
            try:
                ray.get(run_handles, timeout=2.0)
            except ray.exceptions.GetTimeoutError:
                logger.info("(san-check) sound")

        sanity_check()

    def reserve_node(self, model: ModelSpec, routing_key: int):
        """Reserve the least-loaded READY replica for a request.

        The reservation must be released with ``release_node_reservation``
        after remote execution finishes.
        """
        with self._routing_lock:
            ready_nodes = self.model_placements.ready_nodes(model.name)
            if not ready_nodes:
                raise ModelNotReadyError(
                    f"Model '{model.name}' has no READY replica"
                )
            unknown_nodes = [
                node_id for node_id in ready_nodes if node_id not in self.nodes_by_id
            ]
            if unknown_nodes:
                raise RuntimeError(
                    f"Placement table references unknown node '{unknown_nodes[0]}'"
                )
            node_id = self.request_routing.reserve(
                ready_nodes, routing_key, model.name
            )
            return self.nodes_by_id[node_id]

    def release_node_reservation(self, routing_key: int) -> str | None:
        """Release a completed request's local routing reservation."""
        return self.request_routing.release(routing_key)

    def request_routing_snapshot(self) -> dict[str, Any]:
        """Return outstanding API-side routing load for every node."""
        loads = self.request_routing.node_loads()
        assignment_totals = self.request_routing.assignment_totals()
        model_assignment_totals = (
            self.request_routing.model_assignment_totals()
        )
        by_node = {node_id: loads.get(node_id, 0) for node_id in self.node_ids}
        cumulative_by_node = {
            node_id: assignment_totals.get(node_id, 0)
            for node_id in self.node_ids
        }
        return {
            "total": sum(by_node.values()),
            "by_node": by_node,
            "cumulative_assignments": {
                "total": sum(cumulative_by_node.values()),
                "by_node": cumulative_by_node,
                "by_model": {
                    model_name: {
                        "total": sum(node_totals.values()),
                        "by_node": {
                            node_id: node_totals.get(node_id, 0)
                            for node_id in self.node_ids
                        },
                    }
                    for model_name, node_totals in sorted(
                        model_assignment_totals.items()
                    )
                },
            },
        }

    def _record_model_placement_stat(
        self,
        metric: str,
        model_name: str,
        node_id: str,
        count: int = 1,
    ) -> None:
        with self._model_placement_stats_lock:
            models = self._model_placement_stats.setdefault(metric, {})
            nodes = models.setdefault(model_name, {})
            nodes[node_id] = nodes.get(node_id, 0) + count

    def model_placement_stats_snapshot(self) -> dict[str, Any]:
        """Return cumulative replica lifecycle counters for this service."""
        with self._model_placement_stats_lock:
            raw = {
                metric: {
                    model_name: dict(node_counts)
                    for model_name, node_counts in models.items()
                }
                for metric, models in self._model_placement_stats.items()
            }
        return {
            metric: {
                "total": sum(
                    count
                    for node_counts in models.values()
                    for count in node_counts.values()
                ),
                "by_node": {
                    node_id: sum(
                        node_counts.get(node_id, 0)
                        for node_counts in models.values()
                    )
                    for node_id in self.node_ids
                },
                "by_model": {
                    model_name: {
                        "total": sum(node_counts.values()),
                        "by_node": {
                            node_id: node_counts.get(node_id, 0)
                            for node_id in self.node_ids
                        },
                    }
                    for model_name, node_counts in sorted(models.items())
                },
            }
            for metric, models in sorted(raw.items())
        }

    async def _resolve_deployment_nodes(
        self,
        model: ModelSpec,
        model_path: str,
        node_ids: list[str] | None,
        replica_count: int,
    ) -> list[str]:
        """Validate explicit targets or select enough nodes for auto placement."""
        if node_ids is not None:
            if not node_ids:
                raise ValueError("node_ids must contain at least one node")

            unique_node_ids = list(dict.fromkeys(node_ids))
            unknown = [
                node_id
                for node_id in unique_node_ids
                if node_id not in self.nodes_by_id
            ]
            if unknown:
                raise ValueError(f"Unknown node_ids: {unknown}")
            self.last_model_placement_decision = {
                "mode": "explicit",
                "model": model.name,
                "selected_nodes": unique_node_ids,
            }
            return unique_node_ids

        if replica_count < 1:
            raise ValueError("replica_count must be at least 1")
        if replica_count > len(self.node_ids):
            raise ValueError(
                f"replica_count={replica_count} exceeds {len(self.node_ids)} nodes"
            )

        ready_nodes = list(self.model_placements.ready_nodes(model.name))
        if len(ready_nodes) >= replica_count:
            self.last_model_placement_decision = {
                "mode": "automatic",
                "model": model.name,
                "policy": type(self.model_placement_policy).__name__,
                "requested_replica_count": replica_count,
                "existing_ready_nodes": ready_nodes,
                "candidates": [],
                "selected_nodes": ready_nodes,
            }
            return ready_nodes

        snapshots = await asyncio.gather(
            *(node.placement_snapshot.remote(model_path) for node in self.nodes)
        )
        routed_loads = self.request_routing.node_loads()
        controller_loads = {
            snapshot["node_id"]: snapshot["request_load"]["total"]
            for snapshot in snapshots
        }
        ready_set = set(ready_nodes)
        candidates = [
            NodePlacementMetrics(
                node_id=snapshot["node_id"],
                active_requests=max(
                    snapshot["request_load"]["total"],
                    routed_loads.get(snapshot["node_id"], 0),
                ),
                free_cache_bytes=snapshot["model_cache"]["free_bytes"],
                cached_model_count=len(snapshot["model_cache"]["models"]),
                can_fit_without_eviction=snapshot["model_cache"][
                    "can_fit_without_eviction"
                ],
                required_cache_bytes=snapshot["model_cache"][
                    "required_cache_bytes"
                ],
                fit_error=snapshot["model_cache"].get("fit_error"),
                eviction_possible=snapshot["model_cache"]["eviction_possible"],
                estimated_eviction_count=snapshot["model_cache"][
                    "estimated_eviction_count"
                ],
                estimated_eviction_bytes=snapshot["model_cache"][
                    "estimated_eviction_bytes"
                ],
                estimated_evicted_models=(
                    tuple(snapshot["model_cache"]["estimated_evicted_models"])
                    if snapshot["model_cache"]["estimated_evicted_models"]
                    is not None
                    else None
                ),
            )
            for snapshot in snapshots
            if snapshot["node_id"] not in ready_set
        ]
        additional = self.model_placement_policy.select(
            candidates,
            replica_count - len(ready_nodes),
        )
        if len(additional) != replica_count - len(ready_nodes):
            raise ValueError("Not enough candidate nodes for requested replicas")
        self.last_model_placement_decision = {
            "mode": "automatic",
            "model": model.name,
            "policy": type(self.model_placement_policy).__name__,
            "requested_replica_count": replica_count,
            "existing_ready_nodes": ready_nodes,
            "candidates": [
                {
                    "node_id": candidate.node_id,
                    "active_requests": candidate.active_requests,
                    "controller_active_requests": controller_loads[
                        candidate.node_id
                    ],
                    "routed_requests": routed_loads.get(candidate.node_id, 0),
                    "free_cache_bytes": candidate.free_cache_bytes,
                    "cached_model_count": candidate.cached_model_count,
                    "can_fit_without_eviction": (
                        candidate.can_fit_without_eviction
                    ),
                    "required_cache_bytes": candidate.required_cache_bytes,
                    "fit_error": candidate.fit_error,
                    "eviction_possible": candidate.eviction_possible,
                    "estimated_eviction_count": (
                        candidate.estimated_eviction_count
                    ),
                    "estimated_eviction_bytes": candidate.estimated_eviction_bytes,
                    "estimated_evicted_models": (
                        list(candidate.estimated_evicted_models)
                        if candidate.estimated_evicted_models is not None
                        else None
                    ),
                }
                for candidate in candidates
            ],
            "selected_nodes": ready_nodes + list(additional),
        }
        return ready_nodes + list(additional)

    async def deploy_model_replicas(
        self,
        model: ModelSpec,
        model_path: str,
        max_model_len: int | None,
        *,
        node_ids: list[str] | None = None,
        replica_count: int = 1,
        force: bool = False,
    ) -> dict[str, Any]:
        """Serialize lifecycle changes for one model while keeping models independent."""
        lock = self._model_operation_locks.setdefault(model.name, asyncio.Lock())
        async with lock:
            return await self._deploy_model_replicas(
                model,
                model_path,
                max_model_len,
                node_ids=node_ids,
                replica_count=replica_count,
                force=force,
            )

    async def _deploy_model_replicas(
        self,
        model: ModelSpec,
        model_path: str,
        max_model_len: int | None,
        *,
        node_ids: list[str] | None = None,
        replica_count: int = 1,
        force: bool = False,
    ) -> dict[str, Any]:
        """Load a model on one or more nodes and update placement state."""
        targets = await self._resolve_deployment_nodes(
            model,
            model_path,
            node_ids,
            replica_count,
        )
        pending = [
            node_id
            for node_id in targets
            if force
            or self.model_placements.get_state(model.name, node_id)
            is not ModelReplicaState.READY
        ]
        already_ready = [node_id for node_id in targets if node_id not in pending]

        for node_id in pending:
            self._record_model_placement_stat(
                "deployment_attempts", model.name, node_id
            )

        for node_id in pending:
            self.model_placements.set_state(
                model.name, node_id, ModelReplicaState.LOADING
            )

        calls = [
            self.nodes_by_id[node_id].deploy_model.remote(
                model.name, model_path, max_model_len
            )
            for node_id in pending
        ]
        outcomes = await asyncio.gather(*calls, return_exceptions=True)

        ready = list(already_ready)
        failed: dict[str, str] = {}
        evicted: dict[str, list[str]] = {}
        for node_id, outcome in zip(pending, outcomes):
            if isinstance(outcome, BaseException):
                message = str(outcome) or type(outcome).__name__
                self.model_placements.set_state(
                    model.name,
                    node_id,
                    ModelReplicaState.FAILED,
                    error=message,
                )
                self._record_model_placement_stat(
                    "failed_deployments", model.name, node_id
                )
                failed[node_id] = message
                continue

            self.model_placements.set_state(
                model.name, node_id, ModelReplicaState.READY
            )
            self._record_model_placement_stat(
                "successful_deployments", model.name, node_id
            )
            ready.append(node_id)
            evicted_models = outcome if isinstance(outcome, list) else []
            evicted[node_id] = list(evicted_models)
            if evicted_models:
                self._record_model_placement_stat(
                    "evicted_models", model.name, node_id, len(evicted_models)
                )
            for evicted_model in evicted_models:
                self.model_placements.set_state(
                    evicted_model, node_id, ModelReplicaState.ABSENT
                )

        return {
            "target_nodes": targets,
            "ready": sorted(ready),
            "failed": dict(sorted(failed.items())),
            "evicted": dict(sorted(evicted.items())),
        }

    async def undeploy_model_replicas(
        self,
        model_name: str,
        *,
        node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Serialize unload against other lifecycle changes for the same model."""
        lock = self._model_operation_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            return await self._undeploy_model_replicas(
                model_name,
                node_ids=node_ids,
            )

    async def _undeploy_model_replicas(
        self,
        model_name: str,
        *,
        node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Unload selected replicas, or every known replica when unspecified."""
        if node_ids is None:
            targets = list(self.model_placements.nodes(model_name))
        else:
            if not node_ids:
                raise ValueError("node_ids must contain at least one node")
            targets = list(dict.fromkeys(node_ids))
            unknown = [
                node_id for node_id in targets if node_id not in self.nodes_by_id
            ]
            if unknown:
                raise ValueError(f"Unknown node_ids: {unknown}")

        with self._routing_lock:
            pending = [
                node_id
                for node_id in targets
                if self.model_placements.get_state(model_name, node_id)
                is not ModelReplicaState.ABSENT
            ]
            already_absent = [
                node_id for node_id in targets if node_id not in pending
            ]
            for node_id in pending:
                self.model_placements.set_state(
                    model_name, node_id, ModelReplicaState.DRAINING
                )
            reserved = {
                node_id: self.request_routing.reservation_count(
                    node_id, model_name
                )
                for node_id in pending
            }

        blocked = {node_id: count for node_id, count in reserved.items() if count}
        unloadable = [node_id for node_id in pending if node_id not in blocked]

        calls = [
            self.nodes_by_id[node_id].undeploy_model.remote(model_name)
            for node_id in unloadable
        ]
        outcomes = await asyncio.gather(*calls, return_exceptions=True)

        removed: list[str] = []
        failed: dict[str, str] = {
            node_id: (
                f"Model '{model_name}' has {count} routed request(s) "
                "that have not finished"
            )
            for node_id, count in blocked.items()
        }
        for node_id, outcome in zip(unloadable, outcomes):
            if isinstance(outcome, BaseException):
                message = str(outcome) or type(outcome).__name__
                failed[node_id] = message
                continue

            self.model_placements.set_state(
                model_name, node_id, ModelReplicaState.ABSENT
            )
            self._record_model_placement_stat(
                "successful_undeployments", model_name, node_id
            )
            removed.append(node_id)

        return {
            "target_nodes": targets,
            "removed": sorted(removed),
            "already_absent": sorted(already_absent),
            "failed": dict(sorted(failed.items())),
        }

    def serve(self, requests: list[Request]) -> list[list[StepOutput]]:
        """
        Submit the list of requests to be served.
        """

        async def serve_one(req: Request):
            node = self.reserve_node(req.model, req.request_id)
            try:
                return await node.serve.remote(req)
            finally:
                self.release_node_reservation(req.request_id)

        async def serve_all(requests: list[Request]):
            coros = [serve_one(req) for req in requests]
            return await asyncio.gather(*coros, return_exceptions=True)

        return asyncio.run(serve_all(requests))

    def replay(
        self,
        result_file: str,
        num_models: int,
        arrival_rate: float,
        inlen_scale: float = 1.0,
        outlen_scale: float = 1.0,
        duration: int = 120,
        workload: list[Request] | None = None,
    ):
        from aegaeon.workload import syn_workload

        ensure_outfile(result_file)
        reqs: list[Request] = []
        if workload is not None:
            reqs = workload
        else:
            generated_workload = syn_workload(
                model_num=num_models,
                arrival_rate=arrival_rate,
                inlen_scale=inlen_scale,
                outlen_scale=outlen_scale,
                duration=duration,
            )

            req_cnt = Counter()
            for (
                model,
                arrival_time,
                prompt_token_ids,
                decode_tokens,
            ) in generated_workload:
                reqs.append(
                    Request(
                        model,
                        arrival_time,
                        next(req_cnt),
                        prompt_token_ids=prompt_token_ids,
                        decode_tokens=decode_tokens,
                    )
                )

        async def serve_one(req: Request):
            await asyncio.sleep(req.arrival_time)
            req.arrival_time = time.time()
            node = self.reserve_node(req.model, req.request_id)
            try:
                return await node.serve.remote(req)
            finally:
                self.release_node_reservation(req.request_id)

        async def start_trace(latency: int):
            await asyncio.sleep(latency)
            handles = [node.start_trace.remote() for node in self.nodes]
            await asyncio.gather(*handles)

        async def stop_trace(latency: int):
            await asyncio.sleep(latency)
            handles = [node.stop_trace.remote() for node in self.nodes]
            await asyncio.gather(*handles)

        async def serve_all(requests: list[Request]):
            coros = [serve_one(req) for req in requests]

            result = await asyncio.gather(*coros, return_exceptions=True)
            return result[: len(requests)]

        # Execute the workload
        outputs = asyncio.run(serve_all(reqs))

        # Extract metrics
        metrics = {}
        for i, output in enumerate(outputs):
            if isinstance(output, list) and len(output) == reqs[i].decode_tokens:
                ttft, qos, per_token = compute_request_metrics(reqs[i], output)
                latencies = compute_request_latencies(reqs[i], output)
                metrics[reqs[i].request_id] = {
                    "ttft": ttft,
                    "qos": qos,
                    "per_token": per_token,
                    "output": "".join(item.new_token for item in output),
                    **latencies,
                }
            else:
                metrics[reqs[i].request_id] = {
                    "ttft": TTFT_SLO,
                    "qos": 0,
                    "per_token": [],
                }

        # Save metrics
        with open(result_file, "w") as f:
            json.dump(metrics, f)

    def reset(self):
        """
        Reset the service cluster.
        """
        ray.get([node.reset.remote() for node in self.nodes])

    # TODO: a FastAPI app like sllm


class Controller:
    """
    Manager of the control plane on a GPU node.
    """

    ### Singleton components ###
    _node_id: str = None
    _config_path: str = None
    _block_manager: BlockManager = None
    _quick_cache: QuickCache = None
    _cpu_cache_size: int = None
    _cpu_cache: torch.Tensor = None
    _viztracer: VizTracer = None

    _prefill_dispatcher: PrefillDispatcher | None = None
    _prefill_engines: dict[int, PrefillEngine] = {}
    _decode_dispatcher: DecodeDispatcher | None = None
    _decode_engines: dict[int, DecodeEngine] = {}

    _simple_dispatcher = None
    _simple_engines: dict[int, SimpleEngine] = {}

    # Mapping: request_id -> [StepOutput*]
    # - created when request is submitted
    # - cleared when request is finished
    _request_outputs: dict[int, list[StepOutput]] = {}

    # Terminal failures are delivered even when no StepOutput was produced.
    _request_errors: dict[int, RequestCapacityError] = {}

    # Authoritative request-to-model mapping for node-level routing telemetry.
    # Unlike scheduler queue inspection, this spans queued and executing work
    # without counting a request more than once.
    _request_models: dict[int, str] = {}

    # Models that have been cached (model_path filled)
    _cached_models: list = None

    # Explicitly forwarded to child Worker actors. Ray runtime environments
    # are attached to actors, so child actors must not rely on inheriting the
    # Controller process environment implicitly.
    _worker_env_vars: dict[str, str] = {}

    # Small, in-memory telemetry buffer used by the API demo.  Events stay in
    # the controller process, so exposing them does not add work to the GPU
    # workers or the token path.
    _engine_event_sequence: int = 0
    _engine_events: list[dict[str, Any]] = []

    # Synchronization for auto defragmentation.
    _defrag_event: Optional["asyncio.Event"] = None

    @staticmethod
    def _on_engine_switch(
        engine_id: int,
        stage: str,
        previous_model: ModelSpec | None,
        next_model: ModelSpec,
        prefetch_model: ModelSpec | None = None,
    ) -> None:
        """Record a scheduler-issued model switch for runtime observers."""
        global _ctrl

        Controller._engine_event_sequence += 1
        event = {
            "sequence": Controller._engine_event_sequence,
            "type": "model_switch",
            "timestamp": time.time(),
            "node_id": _ctrl._node_id,
            "engine_id": engine_id,
            "stage": stage,
            "from_model": str(previous_model) if previous_model else None,
            "to_model": str(next_model),
            "prefetch_model": str(prefetch_model) if prefetch_model else None,
        }
        Controller._engine_events.append(event)
        if len(Controller._engine_events) > 512:
            del Controller._engine_events[:-512]

    @staticmethod
    def _init_placement_group(
        node_id: str,
        num_engines: int,
        tensor_parallel_size: int,
    ) -> tuple[PlacementGroup, list[list[int]]]:
        """
        Create placement groups for all engines and workers on the node.
        """
        tp_size = tensor_parallel_size
        num_gpus = torch.cuda.device_count()
        required_gpus = num_engines * tp_size
        if num_gpus < required_gpus:
            raise ValueError(
                f"{num_engines} engine(s) with TP={tp_size} require "
                f"{required_gpus} visible GPUs, but only {num_gpus} are available"
            )
        device_ids_list = [
            list(range(i * tp_size, (i + 1) * tp_size)) for i in range(num_engines)
        ]
        pg = ray.util.placement_group(
            [{"CPU": _get_worker_num_cpus() * tp_size, node_id: 0.01}]
            * num_engines,  # engine_id i -> bundle i
            strategy="STRICT_PACK",  # ensures all workers and stage engines are on the same node
        )
        ray.get(pg.ready(), timeout=1000)
        return (pg, device_ids_list)

    @staticmethod
    def _init_cpu_cache(
        cudart: CudaRTLibrary,
        cpu_num_slabs: int,
        cpu_slab_size_bytes: int,
    ) -> torch.Tensor:
        if not os.path.isfile(AEGAEON_CPU_CACHE_FILENAME):
            logger.info(
                f"Creating the shared CPU cache from scratch at {AEGAEON_CPU_CACHE_FILENAME}. "
                "This might take some time."
            )
        else:
            logger.info(f"Found shared CPU cache at {AEGAEON_CPU_CACHE_FILENAME}")

        tensor = torch.from_file(
            AEGAEON_CPU_CACHE_FILENAME,
            shared=True,
            size=cpu_num_slabs * cpu_slab_size_bytes // 2,
            dtype=torch.float16,
        )
        # cudart.cudaHostRegister(ctypes.c_void_p(tensor.data_ptr()), tensor.nbytes)
        return tensor

    @staticmethod
    def _on_new_step_output(
        request_id: int,
        step_output: StepOutput,
    ) -> None:
        """Called when a stage engine generates a new step output."""
        Controller._check_initialized()
        outputs = _ctrl._request_outputs.get(request_id)
        if outputs is not None:
            outputs.append(step_output)

    @staticmethod
    def _on_new_lifetime_event(
        request_id: int,
        event: LifetimeEvent,
        dont_add_if_dup: bool = False,
    ) -> None:
        """Called when a stage engine generates a new step output."""

    @staticmethod
    def _on_requests_finished(finished_reqs: list[Request]) -> None:
        """Called when a stage engine finishes a batch of requests."""
        Controller._check_initialized()
        for request in finished_reqs:
            if request.failure is not None and request.request_id in _ctrl._request_outputs:
                _ctrl._request_errors[request.request_id] = request.failure
            _ctrl._request_models.pop(request.request_id, None)
        allocated_requests = [
            request
            for request in finished_reqs
            if request.request_id in _ctrl._block_manager.block_tables
        ]
        _ctrl._block_manager.free_blocks_batched(allocated_requests)

    @staticmethod
    def _check_initialized():
        global _ctrl
        if _ctrl._node_id is None:
            raise ValueError("Controller uninitialized. Call Controller.init() first.")

    @staticmethod
    async def init(
        config: NodeConfig,
    ) -> dict[str, Any]:
        """
        Starts the control plane event loops and data plane workers on a node.
        """
        global _ctrl
        if _ctrl._node_id is not None:
            raise ValueError(
                f"Controller.init() has already been called on {_ctrl._node_id}"
            )

        node_id = config.node_id
        num_prefill_engines = config.num_prefill_engines
        num_decode_engines = config.num_decode_engines
        num_engines = config.num_engines
        cpu_num_slabs = config.cpu_num_slabs
        cpu_slab_size_bytes = config.cpu_slab_size_bytes
        Controller._engine_event_sequence = 0
        Controller._engine_events = []
        logger.info(f"({node_id}) Initializing Controller")

        if config.config_path:
            Controller._config_path = config.config_path
            content = os.environ.get("AEGAEON_CONFIG_CONTENT")
            if content:
                from aegaeon.models import load_config_from_content

                load_config_from_content(content)
            else:
                from aegaeon.models import set_model_registry

                set_model_registry(Controller._config_path)

        is_simple = num_engines > 0
        if is_simple:
            assert (
                num_prefill_engines == 0 and num_decode_engines == 0
            ), "num_engines is mutually exclusive with num_prefill_engines/num_decode_engines"
            total_engines = num_engines
        else:
            assert (
                num_engines == 0
            ), "num_prefill_engines/num_decode_engines is mutually exclusive with num_engines"
            assert num_prefill_engines >= 0 and num_decode_engines >= 0
            total_engines = num_prefill_engines + num_decode_engines

        # Set visible devices
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # get rid of whatever ray sets
        if "AEGAEON_CUDA_VISIBLE_DEVICES" in os.environ:
            # respect custom settings
            devices = os.environ["AEGAEON_CUDA_VISIBLE_DEVICES"]
            logger.info(f"({node_id}) Found AEGAEON_CUDA_VISIBLE_DEVICES={devices}")
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        assert torch.cuda.is_available(), "Aegaeon is only available on CUDA devices"

        # Initialize cluster resources
        logger.info(f"({node_id}) Initializing block manager")
        block_manager = BlockManager(cpu_num_slabs, cpu_slab_size_bytes)
        logger.info(
            f"({node_id}) Initializing pinned cpu cache ({cpu_num_slabs} x {cpu_slab_size_bytes/(1024**3):.1f}GB)"
        )
        cpu_cache = Controller._init_cpu_cache(
            block_manager.cudart, cpu_num_slabs, cpu_slab_size_bytes
        )

        logger.info(f"({node_id}) Initializing QuickCache")

        # Populate registry instances with model_path from the driver.
        # config.cached_models was already populated by the lifespan.
        from aegaeon.model_registry import get_registry as _reg

        path_map = {m.name: m.model_path for m in config.cached_models if m.model_path}
        for spec in _reg():
            if not spec.model_path and spec.name in path_map:
                object.__setattr__(spec, "model_path", path_map[spec.name])
        # Resolve only the models selected for this node. Resolving the whole
        # registry here caused unused models with no explicit path to download.
        for model in config.cached_models:
            model.path()

        if config.model_cache_size > 0:
            cache_size = config.model_cache_size
        elif config.cached_models:
            from aegaeon.model_registry import get_registry

            cache_size = get_registry().estimate_cache_size_bytes(config.cached_models)
            logger.info(
                f"({node_id}) Auto-estimated model cache size: {cache_size / (1024**3):.1f} GB"
            )
        else:
            cache_size = 20 * (1024**3)  # default 20 GB for dynamic-only mode
            logger.info(
                f"({node_id}) No models pre-cached; using default model cache size: {cache_size / (1024**3):.1f} GB"
            )

        quick_cache = QuickCache(
            block_manager.cudart,
            AEGAEON_MODEL_CACHE_FILENAME,
            cache_size,
            model_cache_snapshot=None,
        )
        for model in config.cached_models:
            quick_cache.cache_model(model.name, model.path())
        model_cache_snapshot = quick_cache.snapshot()

        from aegaeon.config import register_model_config

        for model in config.cached_models:
            register_model_config(model)

        Controller._cached_models = config.cached_models

        logger.info(f"({node_id}) Initializing placement group")
        placement_group, device_ids_list = Controller._init_placement_group(
            node_id, total_engines, config.tensor_parallel_size
        )

        # Initialize the control plane components
        logger.info(f"({node_id}) Creating components")
        parallel_config = ParallelConfig(
            tensor_parallel_size=config.tensor_parallel_size
        )
        loader_config = QuickLoaderConfig(
            model_cache_filename=AEGAEON_MODEL_CACHE_FILENAME,
            model_cache_size=cache_size,
            model_cache_snapshot=model_cache_snapshot,
            arena_fraction=float(os.getenv("AEGAEON_ARENA_FRACTION", "0.875")),
        )

        if is_simple:
            simple_engines = {}
            for i in range(num_engines):
                simple_engines[i] = SimpleEngine(
                    i,
                    parallel_config,
                    loader_config,
                    device_ids_list[i],
                    placement_group,
                    Controller._on_new_step_output,
                    Controller._on_new_lifetime_event,
                    Controller._on_requests_finished,
                )
            prefill_engines = {}
            decode_engines = {}
            prefill_dispatcher = None
            decode_dispatcher = None
            from aegaeon.simple_dispatcher import SimpleDispatcher

            simple_dispatcher = SimpleDispatcher()
        else:
            prefill_engines = {}
            for i in range(num_prefill_engines):
                prefill_engines[i] = PrefillEngine(
                    i,
                    parallel_config,
                    loader_config,
                    device_ids_list[i],
                    placement_group,
                    Controller._on_new_step_output,
                    Controller._on_new_lifetime_event,
                    Controller._on_requests_finished,
                    policy=config.prefill_sched_policy,
                )
            prefill_dispatcher = None
            if prefill_engines:
                prefill_dispatcher = PrefillDispatcher(
                    policy=config.prefill_disp_policy
                )

            decode_engines = {}
            for i in range(
                num_prefill_engines, num_prefill_engines + num_decode_engines
            ):
                decode_engines[i] = DecodeEngine(
                    i,
                    parallel_config,
                    loader_config,
                    device_ids_list[i],
                    placement_group,
                    Controller._on_new_step_output,
                    Controller._on_new_lifetime_event,
                    Controller._on_requests_finished,
                )
            decode_dispatcher = None
            if decode_engines:
                decode_dispatcher = DecodeDispatcher(
                    work_stealing=config.decode_work_stealing,
                    work_stealing_min_batch_gap=(
                        config.decode_work_stealing_min_batch_gap
                    ),
                )
            simple_engines = {}
            simple_dispatcher = None

        # Install the controller
        logger.info(f"({node_id}) Installing components")

        _ctrl._node_id = node_id
        _ctrl._block_manager = block_manager
        _ctrl._quick_cache = quick_cache
        _ctrl._cpu_cache_filename = AEGAEON_CPU_CACHE_FILENAME
        _ctrl._cpu_cache_size = cpu_num_slabs * cpu_slab_size_bytes
        _ctrl._cpu_cache = cpu_cache
        _ctrl._viztracer = VizTracer(
            tracer_entries=1000000, min_duration=200, log_async=False
        )
        _ctrl._prefill_dispatcher = prefill_dispatcher
        _ctrl._prefill_engines = prefill_engines
        _ctrl._decode_dispatcher = decode_dispatcher
        _ctrl._decode_engines = decode_engines
        _ctrl._simple_dispatcher = simple_dispatcher
        _ctrl._simple_engines = simple_engines
        _ctrl._worker_env_vars = {
            key: os.environ[key]
            for key in _RUNTIME_ENV_KEYS
            if key in os.environ
        }
        _ctrl._request_outputs = {}
        _ctrl._request_errors = {}
        _ctrl._request_models = {}
        _ctrl._defrag_event = asyncio.Event()
        _ctrl._defrag_event.set()

        # Initialize stage engines and workers
        logger.info(f"({node_id}) Initializing stage engines")

        init_coros = []
        if is_simple:
            init_coros.extend(
                map(lambda engine: engine.initialize(), _ctrl._simple_engines.values())
            )
            if _ctrl._simple_dispatcher is not None:
                init_coros.append(_ctrl._simple_dispatcher.initialize())
        else:
            dispatchers = (
                [_ctrl._prefill_dispatcher]
                if _ctrl._prefill_dispatcher is not None
                else []
            ) + (
                [_ctrl._decode_dispatcher]
                if _ctrl._decode_dispatcher is not None
                else []
            )
            init_coros.extend(
                map(lambda engine: engine.initialize(), _ctrl._prefill_engines.values())
            )
            init_coros.extend(
                map(lambda engine: engine.initialize(), _ctrl._decode_engines.values())
            )
            init_coros.extend(
                map(lambda dispatcher: dispatcher.initialize(), dispatchers)
            )

        await asyncio.gather(*init_coros)
        return {
            "node_id": node_id,
            "cached_models": sorted(quick_cache.cached_model_names.values()),
        }

    # Dynamic model deployment
    @staticmethod
    def _count_active_requests(model_name: str) -> int:
        """Count requests from submission until model execution completes."""
        return sum(
            1
            for request_model in _ctrl._request_models.values()
            if request_model == model_name
        )

    @staticmethod
    def _evict_inactive_models(model_meta: ModelMeta) -> list[str]:
        """Evict inactive models until we have space for the new model.
        Returns names of evicted models. This does not guarantee success."""
        global _ctrl
        evicted = []
        for model_path, handle in list(_ctrl._quick_cache.cached_models.items()):
            cached_name = _ctrl._quick_cache.cached_model_names[model_path]
            if Controller._count_active_requests(cached_name) > 0:
                continue
            _ctrl._quick_cache.uncache_model(cached_name, model_path)
            evicted.append(cached_name)
            if _ctrl._quick_cache.cpu_allocator.can_allocate_model(model_meta):
                return evicted
        return evicted

    @staticmethod
    async def defrag():
        """To defragment the model cache, we must wait for all active
        switches to complete before relocating the model weights."""
        global _ctrl
        _ctrl._defrag_event.clear()
        logger.info("Defrag in progress")
        try:
            while True:
                switching = False
                for engine in itertools.chain(
                    _ctrl.prefill_engines().values(),
                    _ctrl.decode_engines().values(),
                    _ctrl.simple_engines().values(),
                ):
                    switch_futures = getattr(engine, "switch_futures", [])
                    if switch_futures:
                        _, pending = ray.wait(
                            switch_futures,
                            num_returns=len(switch_futures),
                            timeout=0,
                        )
                        if pending:
                            switching = True
                if not switching:
                    break
                await asyncio.sleep(0.1)

            # Now it is safe to defrag with gate closed
            updated_handles = _ctrl._quick_cache.defrag()
            for model_path, handle in updated_handles.items():
                Controller.remote_call_all_workers_async(
                    "add_model_content", model_path, handle
                )
        finally:  # deadlock prevention
            _ctrl._defrag_event.set()
            logger.info("Defrag completed")

    @staticmethod
    async def deploy_model(
        model_name: str, model_path: str, max_model_len: int | None = None
    ):
        global _ctrl
        from aegaeon.model_registry import ModelSpec, get_registry

        spec = ModelSpec(
            name=model_name, model_path=model_path, max_model_len=max_model_len
        )
        spec = get_registry().register_dynamic(spec)  # update id

        if model_path in _ctrl._quick_cache.cached_models:
            if get_model_config(spec).max_model_len != spec.max_model_len:
                register_model_config(spec)
                # Force switch on restart so new max_model_len takes effect
                for engine in itertools.chain(
                    _ctrl.simple_engines().values(),
                    _ctrl.prefill_engines().values(),
                    _ctrl.decode_engines().values(),
                ):
                    if engine.model_config is not None and engine.model_config.model.name == model_name:
                        engine.model_config = None
                        engine.switch_futures = []
            return

        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            from vllm.tokenizers import get_tokenizer

            from aegaeon.loader.handle import ModelHandle
            from aegaeon.loader.meta import ModelContent

            dummy_content = ModelContent(model_path, [], None)
            dummy_handle = ModelHandle(
                _ctrl._quick_cache.model_cache_start, dummy_content
            )
            _ctrl._quick_cache.cached_models[model_path] = dummy_handle
            _ctrl._quick_cache.cached_model_names[model_path] = model_name
            _ctrl._quick_cache.cached_tokenizers[model_path] = get_tokenizer(
                model_path, trust_remote_code=True
            )
            from aegaeon.config import register_model_config
            register_model_config(spec)
            return []

        model_meta = ModelMeta.from_model_path(model_path)
        alloc = _ctrl._quick_cache.cpu_allocator
        evicted = []
        if not alloc.can_allocate_model(model_meta):
            # Evict inactive models
            evicted = Controller._evict_inactive_models(model_meta)
            if evicted:
                logger.info(f"Auto-evicted: {evicted}")
            if not alloc.can_allocate_model(model_meta):
                await _ctrl.defrag()
                if not alloc.can_allocate_model(model_meta):
                    free_gb = alloc.get_free_size() / (1024**3)
                    needed_gb = model_meta.get_storage_size() / (1024**3)
                    raise MemoryError(
                        f"Insufficient CPU cache after eviction and defragmentation. "
                        f"Free: {free_gb:.1f} GB, needed: ~{needed_gb:.1f} GB."
                    )

        _ctrl._quick_cache.cache_model(model_name, model_path)
        from aegaeon.config import register_model_config

        register_model_config(spec)

        # This used to be done in QuickLoader init
        handle = _ctrl._quick_cache.cached_models[model_path]
        Controller.remote_call_all_workers_async(
            "add_model_content", model_path, handle
        )

        return evicted

    @staticmethod
    async def undeploy_model(model_name: str):
        global _ctrl
        from aegaeon.model_registry import get_registry

        registry = get_registry()
        spec = registry.get_by_name(model_name)
        if spec is None:
            raise ValueError(f"Model '{model_name}' is not deployed.")

        # Refuse if model has in-flight requests
        active_count = Controller._count_active_requests(model_name)
        if active_count > 0:
            raise ValueError(
                f"Model '{model_name}' has {active_count} active request(s). "
                f"Wait for them to finish or cancel them first."
            )

        # Remove at Controller level
        from aegaeon.config import unregister_model_config

        registry.unregister_dynamic(model_name)
        unregister_model_config(model_name)
        # Force switch on restart
        for engine in itertools.chain(
            _ctrl.simple_engines().values(),
            _ctrl.prefill_engines().values(),
            _ctrl.decode_engines().values(),
        ):
            if (
                engine.model_config is not None
                and engine.model_config.model.name == model_name
            ):
                engine.model_config = None
                engine.switch_futures = []

        model_path = spec.model_path
        if model_path:
            _ctrl._quick_cache.uncache_model(model_name, model_path)
            # Remove at worker level
            Controller.remote_call_all_workers_async("delete_model_content", model_path)

    @staticmethod
    async def start_event_loops():
        Controller._check_initialized()

        if _ctrl._simple_dispatcher is not None:
            # Runs forever
            try:
                await asyncio.gather(
                    _ctrl._block_manager.start_event_loop(),
                    *map(
                        lambda engine: engine.start_event_loop(),
                        _ctrl._simple_engines.values(),
                    ),
                    _ctrl._simple_dispatcher.start_event_loop(),
                )
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                current = asyncio.current_task()
                pending = [task for task in asyncio.all_tasks() if task is not current]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise
        else:
            dispatchers = (
                [_ctrl._prefill_dispatcher]
                if _ctrl._prefill_dispatcher is not None
                else []
            ) + (
                [_ctrl._decode_dispatcher]
                if _ctrl._decode_dispatcher is not None
                else []
            )

            # Runs forever
            try:
                await asyncio.gather(
                    # Component loops
                    _ctrl._block_manager.start_event_loop(),
                    *map(
                        lambda engine: engine.start_event_loop(),
                        _ctrl._prefill_engines.values(),
                    ),
                    *map(
                        lambda engine: engine.start_event_loop(),
                        _ctrl._decode_engines.values(),
                    ),
                    *map(lambda dispatcher: dispatcher.start_event_loop(), dispatchers),
                )

            except Exception as e:
                # ..something wrong in the control plane
                traceback.print_exception(type(e), e, e.__traceback__)
                current = asyncio.current_task()
                pending = [task for task in asyncio.all_tasks() if task is not current]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise

    @staticmethod
    async def submit(request: Request) -> None:
        """Submit a request without waiting for all generated tokens."""
        Controller._check_initialized()
        if request.request_id in _ctrl._request_outputs:
            raise ValueError(f"Duplicate request id {request.request_id}")

        _ctrl._request_outputs[request.request_id] = []
        _ctrl._request_models[request.request_id] = request.model.name
        Controller._on_new_lifetime_event(
            request.request_id, LifetimeEvent(LifetimeEventType.Issued)
        )

        try:
            if _ctrl._simple_dispatcher is not None:
                _ctrl._simple_dispatcher.add(request)
            else:
                if _ctrl._prefill_dispatcher is None:
                    raise ValueError(
                        f"Submitting a new request to {_ctrl._node_id} which has no "
                        "prefill engine configured."
                    )
                _ctrl._prefill_dispatcher.add(request)
        except BaseException:
            _ctrl._request_outputs.pop(request.request_id, None)
            _ctrl._request_errors.pop(request.request_id, None)
            _ctrl._request_models.pop(request.request_id, None)
            raise

    @staticmethod
    async def wait_for_step_outputs(
        request_id: int,
        offset: int,
    ) -> tuple[list[StepOutput], bool]:
        """Long-poll until one or more outputs are available after ``offset``."""
        SLEEP_FOR_STEP_OUTPUTS = 0.005

        Controller._check_initialized()
        while True:
            outputs = _ctrl._request_outputs.get(request_id)
            if outputs is None:
                raise KeyError(f"Unknown request id {request_id}")
            error = _ctrl._request_errors.get(request_id)
            if error is not None:
                raise error
            if offset < 0 or offset > len(outputs):
                raise ValueError(
                    f"Invalid output offset {offset} for request {request_id}; "
                    f"available outputs: {len(outputs)}"
                )
            if len(outputs) > offset:
                new_outputs = list(outputs[offset:])
                return new_outputs, bool(new_outputs[-1].is_finished)
            await asyncio.sleep(SLEEP_FOR_STEP_OUTPUTS)

    @staticmethod
    async def release_request(request_id: int) -> None:
        """Release a completed request's buffered outputs."""
        Controller._check_initialized()
        outputs = _ctrl._request_outputs.get(request_id)
        if outputs is None:
            return
        if request_id not in _ctrl._request_errors and (
            not outputs or not outputs[-1].is_finished
        ):
            raise ValueError(f"Request {request_id} has not finished")
        _ctrl._request_outputs.pop(request_id, None)
        _ctrl._request_errors.pop(request_id, None)
        _ctrl._request_models.pop(request_id, None)

    @staticmethod
    async def release_request_when_finished(request_id: int) -> None:
        """Eventually release output state after an HTTP client disconnects."""
        SLEEP_FOR_STEP_OUTPUTS = 0.005

        Controller._check_initialized()
        while True:
            outputs = _ctrl._request_outputs.get(request_id)
            if outputs is None:
                return
            if request_id in _ctrl._request_errors or (
                outputs and outputs[-1].is_finished
            ):
                await Controller.release_request(request_id)
                return
            await asyncio.sleep(SLEEP_FOR_STEP_OUTPUTS)

    @staticmethod
    async def runtime_snapshot(after_sequence: int = 0) -> dict[str, Any]:
        """Return serializable engine state plus switch events after a cursor."""
        Controller._check_initialized()

        engines = []
        for engine in itertools.chain(
            _ctrl._prefill_engines.values(),
            _ctrl._decode_engines.values(),
            _ctrl._simple_engines.values(),
        ):
            model = None
            if engine.model_config is not None:
                model = str(engine.model_config.model)

            switching = False
            switch_futures = getattr(engine, "switch_futures", [])
            if switch_futures:
                _, pending = ray.wait(
                    switch_futures,
                    num_returns=len(switch_futures),
                    timeout=0,
                )
                switching = bool(pending)
                if not switching:
                    engine.switch_futures = []

            active_requests = 0
            for batch in getattr(engine, "batches_in_pipeline", []):
                active_requests += len(getattr(batch, "requests", []))

            engines.append(
                {
                    "node_id": _ctrl._node_id,
                    "engine_id": engine.engine_id,
                    "stage": str(engine.stage),
                    "model": model,
                    "status": (
                        "switching" if switching else ("active" if model else "idle")
                    ),
                    "switch_started_at": (
                        engine.switch_started_at if switching else None
                    ),
                    "device_ids": list(engine.device_ids),
                    "active_requests": active_requests,
                }
            )

        events = [
            dict(event)
            for event in Controller._engine_events
            if event["sequence"] > after_sequence
        ]
        work_stealing = None
        decode_load = None
        if _ctrl._decode_dispatcher is not None:
            work_stealing = {
                "enabled": _ctrl._decode_dispatcher.work_stealing,
                "min_batch_gap": (
                    _ctrl._decode_dispatcher.work_stealing_min_batch_gap
                ),
                **_ctrl._decode_dispatcher.get_work_stealing_stats(),
            }
            decode_load = _ctrl._decode_dispatcher.get_decode_load_stats()

        allocator = _ctrl._quick_cache.cpu_allocator
        cached_models = sorted(_ctrl._quick_cache.cached_model_names.values())
        requests_by_model: dict[str, int] = {}
        for model_name in _ctrl._request_models.values():
            requests_by_model[model_name] = requests_by_model.get(model_name, 0) + 1

        return {
            "node_id": _ctrl._node_id,
            "sequence": Controller._engine_event_sequence,
            "timestamp": time.time(),
            "engines": engines,
            "events": events,
            "work_stealing": work_stealing,
            "decode_load": decode_load,
            "model_cache": {
                "capacity_bytes": allocator.get_size(),
                "allocated_bytes": allocator.get_allocated_size(),
                "free_bytes": allocator.get_free_size(),
                "models": cached_models,
            },
            "request_load": {
                "total": len(_ctrl._request_models),
                "by_model": dict(sorted(requests_by_model.items())),
            },
        }

    @staticmethod
    async def placement_snapshot(model_path: str) -> dict[str, Any]:
        """Return live load plus an allocator-accurate model fit check."""
        global _ctrl
        snapshot = await Controller.runtime_snapshot(0)
        fit_error = None
        required_cache_bytes = None
        can_fit_without_eviction = False
        eviction_possible = False
        estimated_evicted_models = None
        estimated_eviction_count = None
        estimated_eviction_bytes = None
        try:
            from aegaeon.loader.meta import ModelMeta

            model_meta = ModelMeta.from_model_path(model_path)
            slice_sizes = model_meta.get_slice_sizes()
            if not slice_sizes:
                raise ValueError(f"No safetensors found under '{model_path}'")
            required_cache_bytes = model_meta.get_storage_size()
            can_fit_without_eviction = (
                _ctrl._quick_cache.cpu_allocator.can_allocate_model(model_meta)
            )
            if can_fit_without_eviction:
                eviction_possible = True
                estimated_evicted_models = []
                estimated_eviction_count = 0
                estimated_eviction_bytes = 0
            else:
                from aegaeon.loader.allocator import MemoryTable
                from aegaeon.loader.meta import SliceInfo

                allocator = _ctrl._quick_cache.cpu_allocator
                simulated_table = MemoryTable(allocator.memory_table.size)
                for allocated_slice in allocator.memory_table.table:
                    simulated_table.append(
                        SliceInfo.with_addr_size(
                            allocated_slice.start, allocated_slice.get_size()
                        )
                    )
                estimated_evicted_models = []
                estimated_eviction_bytes = 0
                for cached_path, handle in (
                    _ctrl._quick_cache.cached_models.items()
                ):
                    cached_name = _ctrl._quick_cache.cached_model_names[cached_path]
                    if Controller._count_active_requests(cached_name) > 0:
                        continue
                    model_bytes = sum(
                        tensors_handle.slice_info.get_size()
                        for sharding_handle in handle.sharding_handles
                        for tensors_handle in sharding_handle.tensors_handles
                    )
                    estimated_evicted_models.append(cached_name)
                    estimated_eviction_bytes += model_bytes
                    for sharding_handle in handle.sharding_handles:
                        for tensors_handle in sharding_handle.tensors_handles:
                            allocated_slice = tensors_handle.slice_info
                            simulated_table.delete(
                                SliceInfo.with_addr_size(
                                    allocated_slice.start,
                                    allocated_slice.get_size(),
                                )
                            )
                    if simulated_table.can_allocate(slice_sizes):
                        eviction_possible = True
                        break
                estimated_eviction_count = len(estimated_evicted_models)
                if (
                    not eviction_possible
                    and simulated_table.get_free_size() >= required_cache_bytes
                ):
                    # Controller defragments after exhausting inactive models.
                    eviction_possible = True
        except Exception as exc:
            fit_error = str(exc) or type(exc).__name__

        snapshot["model_cache"].update(
            {
                "can_fit_without_eviction": can_fit_without_eviction,
                "required_cache_bytes": required_cache_bytes,
                "fit_error": fit_error,
                "eviction_possible": eviction_possible,
                "estimated_evicted_models": estimated_evicted_models,
                "estimated_eviction_count": estimated_eviction_count,
                "estimated_eviction_bytes": estimated_eviction_bytes,
            }
        )
        return snapshot

    @staticmethod
    async def serve(request: Request) -> list[StepOutput]:
        """Submit and serve a request, returning all tokens at completion."""
        await Controller.submit(request)
        outputs: list[StepOutput] = []
        offset = 0

        try:
            while True:
                new_outputs, is_finished = await Controller.wait_for_step_outputs(
                    request.request_id, offset
                )
                outputs.extend(new_outputs)
                offset += len(new_outputs)
                if is_finished:
                    break
            await Controller.release_request(request.request_id)
            return outputs
        except BaseException:
            # A failed or cancelled consumer must not retain token buffers.
            # The completion callback retains responsibility for active KV blocks
            # and routing state if the engine is still executing this request.
            _ctrl._request_outputs.pop(request.request_id, None)
            _ctrl._request_errors.pop(request.request_id, None)
            raise

    async def reset():
        """Reset the control plane and workers."""
        Controller._check_initialized()

        global _ctrl
        _ctrl._block_manager.reset()
        # TODO
        if _ctrl._prefill_dispatcher is not None:
            _ctrl._prefill_dispatcher.reset()
        if _ctrl._decode_dispatcher is not None:
            _ctrl._decode_dispatcher.reset()
        for engine in _ctrl._prefill_engines.values():
            engine.reset()
        for engine in _ctrl._decode_engines.values():
            engine.reset()
        for engine in _ctrl._simple_engines.values():
            engine.reset()

        _ctrl._request_outputs.clear()
        _ctrl._request_errors.clear()
        _ctrl._request_models.clear()

    async def start_trace(self):
        global _ctrl
        Controller._check_initialized()
        _ctrl._viztracer.start()

    async def stop_trace(self):
        global _ctrl
        Controller._check_initialized()
        _ctrl._viztracer.stop()
        _ctrl._viztracer.save("trace-control.json")

    @staticmethod
    def remote_call_all_workers_async(func_name: str, *args, **kwargs) -> list:
        """
        Call func_name asynchronously on all workers across all engines; return the futures immediately.
        """
        global _ctrl
        Controller._check_initialized()

        handlers = []
        for engine in itertools.chain(
            Controller.prefill_engines().values(),
            Controller.decode_engines().values(),
            Controller.simple_engines().values(),
        ):
            handlers.extend(
                engine._remote_call_all_workers_async(func_name, *args, **kwargs)
            )
        return handlers

    @staticmethod
    def node_id() -> str:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._node_id

    @staticmethod
    def block_manager() -> BlockManager:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._block_manager

    @staticmethod
    def get_tokenizer(name: str):
        global _ctrl
        Controller._check_initialized()
        return _ctrl._quick_cache.get_tokenizer(name)

    @staticmethod
    def cpu_cache_filename() -> str:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._cpu_cache_filename

    @staticmethod
    def cpu_cache_size() -> int:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._cpu_cache_size

    @staticmethod
    def cached_models():
        global _ctrl
        Controller._check_initialized()
        return _ctrl._cached_models

    @staticmethod
    def config_path() -> str:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._config_path or ""

    @staticmethod
    def worker_runtime_env(
        enable_foundry: bool = False,
    ) -> dict[str, dict[str, str]]:
        """Return a detached Ray runtime_env for this node's Worker actors."""
        global _ctrl
        Controller._check_initialized()
        return {
            "env_vars": _foundry_worker_env(
                _ctrl._worker_env_vars,
                cuda_graph_config=get_server_config().cuda_graph,
                enable_foundry=enable_foundry,
            )
        }

    @staticmethod
    def prefill_dispatcher() -> PrefillDispatcher | None:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._prefill_dispatcher

    @staticmethod
    def decode_dispatcher() -> DecodeDispatcher | None:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._decode_dispatcher

    @staticmethod
    def prefill_engines() -> dict[int, PrefillEngine]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._prefill_engines

    @staticmethod
    def decode_engines() -> dict[int, DecodeEngine]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._decode_engines

    @staticmethod
    def simple_dispatcher():
        global _ctrl
        Controller._check_initialized()
        return _ctrl._simple_dispatcher

    @staticmethod
    def simple_engines() -> dict[int, "SimpleEngine"]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._simple_engines

    @staticmethod
    def defrag_event() -> "asyncio.Event":
        global _ctrl
        Controller._check_initialized()
        return _ctrl._defrag_event


_ctrl: Controller = Controller()
