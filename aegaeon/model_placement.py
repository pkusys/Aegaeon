"""In-memory model replica placement state.

The placement table is intentionally independent from Ray and QuickCache.  It
records control-plane knowledge about model replicas; later routing and model
deployment policies can consume this state without depending on cache internals.
"""

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple


class ModelNotReadyError(RuntimeError):
    """Raised when a known model has no replica that can accept requests."""


class RequestRoutingPolicy(Protocol):
    """Interface for selecting one READY replica for a new request."""

    def select(
        self,
        ready_nodes: Sequence[str],
        node_loads: Mapping[str, int],
        routing_key: int,
    ) -> str: ...


class LeastOutstandingRequestsPolicy:
    """Select the least-loaded replica with deterministic rotating ties."""

    def select(
        self,
        ready_nodes: Sequence[str],
        node_loads: Mapping[str, int],
        routing_key: int,
    ) -> str:
        if not ready_nodes:
            raise ValueError("ready_nodes must contain at least one node")
        nodes = tuple(ready_nodes)
        start = routing_key % len(nodes)
        return min(
            enumerate(nodes),
            key=lambda item: (
                node_loads.get(item[1], 0),
                (item[0] - start) % len(nodes),
            ),
        )[1]


class RoundRobinRoutingPolicy:
    """Select a READY replica deterministically without consulting load."""

    def select(
        self,
        ready_nodes: Sequence[str],
        node_loads: Mapping[str, int],
        routing_key: int,
    ) -> str:
        if not ready_nodes:
            raise ValueError("ready_nodes must contain at least one node")
        return ready_nodes[routing_key % len(ready_nodes)]


def create_request_routing_policy(name: str) -> RequestRoutingPolicy:
    """Construct a configured request routing policy by stable name."""
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "least-outstanding":
        return LeastOutstandingRequestsPolicy()
    if normalized == "round-robin":
        return RoundRobinRoutingPolicy()
    raise ValueError(f"Unknown request routing policy: {name!r}")


class RequestRoutingTable:
    """Track locally routed requests until their remote execution finishes.

    The table lets the API process balance new requests without querying every
    Controller over Ray for each arrival.  Request IDs make reservations
    idempotent, while a lock keeps synchronous API callers safe.
    """

    def __init__(
        self,
        policy: Optional[RequestRoutingPolicy] = None,
    ) -> None:
        self.policy = policy or LeastOutstandingRequestsPolicy()
        self._assignments: Dict[int, Tuple[str, Optional[str]]] = {}
        self._node_loads: Dict[str, int] = {}
        self._assignment_totals: Dict[str, int] = {}
        self._model_assignment_totals: Dict[str, Dict[str, int]] = {}
        self._lock = Lock()

    def reserve(
        self,
        ready_nodes: Sequence[str],
        routing_key: int,
        model_name: Optional[str] = None,
    ) -> str:
        """Reserve the least-loaded READY node for one request."""
        if not ready_nodes:
            raise ValueError("ready_nodes must contain at least one node")

        nodes = tuple(ready_nodes)
        with self._lock:
            existing = self._assignments.get(routing_key)
            if existing is not None:
                node_id, existing_model = existing
                if existing_model != model_name:
                    raise ValueError(
                        f"Request {routing_key} is already reserved for "
                        f"model {existing_model!r}"
                    )
                return node_id

            node_id = self.policy.select(
                nodes,
                self._node_loads,
                routing_key,
            )
            if node_id not in nodes:
                raise ValueError(
                    f"Routing policy selected non-READY node '{node_id}'"
                )
            self._assignments[routing_key] = (node_id, model_name)
            self._node_loads[node_id] = self._node_loads.get(node_id, 0) + 1
            self._assignment_totals[node_id] = (
                self._assignment_totals.get(node_id, 0) + 1
            )
            if model_name is not None:
                model_totals = self._model_assignment_totals.setdefault(
                    model_name, {}
                )
                model_totals[node_id] = model_totals.get(node_id, 0) + 1
            return node_id

    def release(self, routing_key: int) -> Optional[str]:
        """Release a request reservation and return its node, if present."""
        with self._lock:
            reservation = self._assignments.pop(routing_key, None)
            if reservation is None:
                return None
            node_id, _model_name = reservation

            remaining = self._node_loads[node_id] - 1
            if remaining:
                self._node_loads[node_id] = remaining
            else:
                self._node_loads.pop(node_id)
            return node_id

    def node_loads(self) -> Dict[str, int]:
        """Return a copy of current request reservations by node."""
        with self._lock:
            return dict(self._node_loads)

    def assignment_totals(self) -> Dict[str, int]:
        """Return cumulative first-time request assignments by node."""
        with self._lock:
            return dict(self._assignment_totals)

    def model_assignment_totals(self) -> Dict[str, Dict[str, int]]:
        """Return cumulative first-time assignments grouped by model and node."""
        with self._lock:
            return {
                model_name: dict(node_totals)
                for model_name, node_totals in self._model_assignment_totals.items()
            }

    def reservation_count(self, node_id: str, model_name: str) -> int:
        """Return reservations that protect one model replica from unload."""
        with self._lock:
            return sum(
                reserved_node == node_id and reserved_model == model_name
                for reserved_node, reserved_model in self._assignments.values()
            )


@dataclass(frozen=True)
class NodePlacementMetrics:
    """Control-plane metrics used when choosing a model deployment node."""

    node_id: str
    active_requests: int
    free_cache_bytes: int
    cached_model_count: int
    can_fit_without_eviction: bool = True
    required_cache_bytes: Optional[int] = None
    fit_error: Optional[str] = None
    eviction_possible: bool = True
    estimated_eviction_count: Optional[int] = 0
    estimated_eviction_bytes: Optional[int] = 0
    estimated_evicted_models: Optional[Tuple[str, ...]] = ()


class ModelPlacementPolicy(Protocol):
    """Interface for automatic model deployment policies."""

    def select(
        self,
        candidates: Sequence[NodePlacementMetrics],
        count: int,
    ) -> Tuple[str, ...]: ...


class LeastLoadedPlacementPolicy:
    """Prefer eviction-free placement, then load and cache headroom."""

    def select(
        self,
        candidates: Sequence[NodePlacementMetrics],
        count: int,
    ) -> Tuple[str, ...]:
        if count < 0:
            raise ValueError("count must not be negative")
        ranked = sorted(
            candidates,
            key=lambda node: (
                not node.can_fit_without_eviction,
                not node.eviction_possible,
                (
                    node.estimated_eviction_bytes
                    if node.estimated_eviction_bytes is not None
                    else float("inf")
                ),
                (
                    node.estimated_eviction_count
                    if node.estimated_eviction_count is not None
                    else float("inf")
                ),
                node.active_requests,
                -node.free_cache_bytes,
                node.cached_model_count,
                node.node_id,
            ),
        )
        return tuple(node.node_id for node in ranked[:count])


class RoundRobinPlacementPolicy:
    """Select candidate nodes cyclically without consulting placement metrics."""

    def __init__(self) -> None:
        self._next_index = 0
        self._lock = Lock()

    def select(
        self,
        candidates: Sequence[NodePlacementMetrics],
        count: int,
    ) -> Tuple[str, ...]:
        if count < 0:
            raise ValueError("count must not be negative")
        ordered = sorted(candidates, key=lambda node: node.node_id)
        if not ordered or count == 0:
            return ()

        selected_count = min(count, len(ordered))
        with self._lock:
            start = self._next_index % len(ordered)
            selected = tuple(
                ordered[(start + offset) % len(ordered)].node_id
                for offset in range(selected_count)
            )
            self._next_index = (start + selected_count) % len(ordered)
        return selected


def create_model_placement_policy(name: str) -> ModelPlacementPolicy:
    """Construct a configured model placement policy by stable name."""
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "least-loaded":
        return LeastLoadedPlacementPolicy()
    if normalized == "round-robin":
        return RoundRobinPlacementPolicy()
    raise ValueError(f"Unknown model placement policy: {name!r}")


class ModelReplicaState(str, Enum):
    """Lifecycle state of one model replica on one physical node."""

    ABSENT = "absent"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelReplicaPlacement:
    """Current placement state for a model on a node."""

    model_name: str
    node_id: str
    state: ModelReplicaState
    error: Optional[str] = None


class ModelPlacementTable:
    """Track model replicas across nodes.

    ``ABSENT`` is represented by the lack of an entry.  Setting a replica to
    ``ABSENT`` therefore removes it from the table.  Query results are sorted
    to keep routing fallbacks, tests, and runtime snapshots deterministic.
    """

    def __init__(self) -> None:
        self._placements: Dict[str, Dict[str, ModelReplicaPlacement]] = {}

    @classmethod
    def from_ready_models(
        cls,
        models_by_node: Mapping[str, Iterable[str]],
    ) -> "ModelPlacementTable":
        """Build a table from node-local caches observed at service startup."""
        table = cls()
        for node_id, model_names in models_by_node.items():
            for model_name in model_names:
                table.set_state(model_name, node_id, ModelReplicaState.READY)
        return table

    @staticmethod
    def _validate_key(value: str, key_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key_name} must be a non-empty string")

    def set_state(
        self,
        model_name: str,
        node_id: str,
        state: ModelReplicaState,
        *,
        error: Optional[str] = None,
    ) -> None:
        """Record the latest known state of a model replica."""
        self._validate_key(model_name, "model_name")
        self._validate_key(node_id, "node_id")
        if not isinstance(state, ModelReplicaState):
            raise TypeError("state must be a ModelReplicaState")
        if state is ModelReplicaState.FAILED and not error:
            raise ValueError("FAILED replicas must include an error")
        if state is not ModelReplicaState.FAILED and error is not None:
            raise ValueError("error is only valid for FAILED replicas")

        if state is ModelReplicaState.ABSENT:
            replicas = self._placements.get(model_name)
            if replicas is None:
                return
            replicas.pop(node_id, None)
            if not replicas:
                self._placements.pop(model_name, None)
            return

        self._placements.setdefault(model_name, {})[node_id] = ModelReplicaPlacement(
            model_name=model_name,
            node_id=node_id,
            state=state,
            error=error,
        )

    def get_state(self, model_name: str, node_id: str) -> ModelReplicaState:
        """Return ``ABSENT`` when no placement is recorded."""
        placement = self._placements.get(model_name, {}).get(node_id)
        if placement is None:
            return ModelReplicaState.ABSENT
        return placement.state

    def nodes(
        self,
        model_name: str,
        state: Optional[ModelReplicaState] = None,
    ) -> Tuple[str, ...]:
        """Return nodes holding a model, optionally filtered by state."""
        replicas = self._placements.get(model_name, {}).values()
        return tuple(
            sorted(
                replica.node_id
                for replica in replicas
                if state is None or replica.state is state
            )
        )

    def ready_nodes(self, model_name: str) -> Tuple[str, ...]:
        """Return nodes that may currently receive requests for a model."""
        return self.nodes(model_name, ModelReplicaState.READY)

    def models_on_node(
        self,
        node_id: str,
        state: Optional[ModelReplicaState] = None,
    ) -> Tuple[str, ...]:
        """Return models recorded on a node, optionally filtered by state."""
        return tuple(
            sorted(
                model_name
                for model_name, replicas in self._placements.items()
                if node_id in replicas
                and (state is None or replicas[node_id].state is state)
            )
        )

    def snapshot(self) -> Dict[str, list]:
        """Return a detached, JSON-serializable placement snapshot."""
        return {
            model_name: [
                {
                    "node_id": placement.node_id,
                    "state": placement.state.value,
                    "error": placement.error,
                }
                for placement in sorted(
                    replicas.values(), key=lambda placement: placement.node_id
                )
            ]
            for model_name, replicas in sorted(self._placements.items())
        }
