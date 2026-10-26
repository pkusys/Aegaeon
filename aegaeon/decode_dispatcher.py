import asyncio
import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Literal, Optional, TYPE_CHECKING

from aegaeon.model_registry import ModelSpec
from aegaeon.config import ModelConfig, get_model_config
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.block_manager import BlockManager, BLOCK_SIZE
from aegaeon.utils import TTFT_SLO, TPOT_SLO, estimate_switch_time

if TYPE_CHECKING:
    from aegaeon.decode_scheduler import (
        DecodeScheduler,
        DecodeSchedulerLoad,
        StealableDecodeBatch,
    )

logger = init_logger(__name__)

# Sleep for this many seconds when there is no request to dispatch,
# or when no scheduler can be dispatched to.
SLEEP_WHEN_DISPATCH_PENDS = 0.003

# Sleep for this many seconds in each event loop, useful for debugging.
SLEEP_IN_EACH_EVENT_LOOP = 0


def coefficient_of_variation(values: Iterable[float]) -> float:
    """Return population standard deviation divided by the arithmetic mean."""
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


class WorkStealingRejectionReason(str, Enum):
    DISABLED = "disabled"
    DONOR_NOT_AT_BOUNDARY = "donor_not_at_boundary"
    INTERNAL_ERROR = "internal_error"
    STALE_PLAN = "stale_plan"
    RECEIVER_NOT_IDLE = "receiver_not_idle"
    NO_RECEIVER = "no_receiver"
    NO_DONOR = "no_donor"
    INSUFFICIENT_LOAD_GAP = "insufficient_load_gap"
    NO_ELIGIBLE_BATCH = "no_eligible_batch"


@dataclass
class WorkStealingStats:
    attempts: int = 0
    successes: int = 0
    rejections: Counter[str] = field(default_factory=Counter)

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_success(self) -> None:
        self.successes += 1

    def record_rejection(self, reason: WorkStealingRejectionReason) -> None:
        self.rejections[reason.value] += 1

    def snapshot(self) -> dict:
        """Return a detached, serialization-friendly metrics snapshot."""
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "rejections": dict(self.rejections),
        }


@dataclass(frozen=True)
class WorkStealingPlan:
    receiver: "DecodeScheduler"
    donor: "DecodeScheduler"
    candidate: "StealableDecodeBatch"
    receiver_load: "DecodeSchedulerLoad"
    donor_load: "DecodeSchedulerLoad"
    model_affinity: bool


@dataclass(frozen=True)
class WorkStealingDecision:
    plan: Optional[WorkStealingPlan] = None
    rejection_reason: Optional[WorkStealingRejectionReason] = None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.rejection_reason is None):
            raise ValueError("decision must contain exactly one result")

    @property
    def accepted(self) -> bool:
        return self.plan is not None


def select_work_steal(
    receiver: "DecodeScheduler",
    schedulers: Iterable["DecodeScheduler"],
    min_batch_gap: int,
) -> WorkStealingDecision:
    """Select a deterministic work-stealing plan without mutating schedulers."""
    if min_batch_gap < 2:
        raise ValueError("min_batch_gap must be at least 2")

    receiver_load = receiver.get_load_snapshot()
    if not receiver_load.is_idle:
        return WorkStealingDecision(
            rejection_reason=WorkStealingRejectionReason.RECEIVER_NOT_IDLE
        )

    donor_views = [
        (scheduler, scheduler.get_load_snapshot())
        for scheduler in schedulers
        if scheduler is not receiver
    ]
    donors_with_work = [
        (scheduler, load) for scheduler, load in donor_views if load.batch_count > 0
    ]
    if not donors_with_work:
        return WorkStealingDecision(
            rejection_reason=WorkStealingRejectionReason.NO_DONOR
        )

    eligible_donors = [
        (scheduler, load)
        for scheduler, load in donors_with_work
        if load.batch_count - receiver_load.batch_count >= min_batch_gap
    ]
    if not eligible_donors:
        return WorkStealingDecision(
            rejection_reason=WorkStealingRejectionReason.INSUFFICIENT_LOAD_GAP
        )

    ranked_candidates = []
    for donor, donor_load in eligible_donors:
        for candidate in donor.get_stealable_batches():
            model_affinity = (
                receiver_load.resident_model is not None
                and candidate.batch.model == receiver_load.resident_model
            )
            rank = (
                0 if model_affinity else 1,
                -donor_load.batch_count,
                donor_load.engine_id,
                candidate.queue_index,
            )
            ranked_candidates.append(
                (
                    rank,
                    WorkStealingPlan(
                        receiver=receiver,
                        donor=donor,
                        candidate=candidate,
                        receiver_load=receiver_load,
                        donor_load=donor_load,
                        model_affinity=model_affinity,
                    ),
                )
            )

    if not ranked_candidates:
        return WorkStealingDecision(
            rejection_reason=WorkStealingRejectionReason.NO_ELIGIBLE_BATCH
        )

    _, plan = min(ranked_candidates, key=lambda ranked: ranked[0])
    return WorkStealingDecision(plan=plan)


class DecodeDispatcher:
    """
    Dispatcher of decode requests to individual schedulers in decode engines.

    ## Event loops
    - Dispatch
        - Pop a request for dispatching
        - Find a decode scheduler to dispatch to

    Another API is provided for re-dispatching of batches.
    """

    def __init__(
        self,
        policy: Literal["fcfs", "sjf"] = "fcfs",
        work_stealing: bool = False,
        work_stealing_min_batch_gap: int = 2,
    ):
        if work_stealing_min_batch_gap < 2:
            raise ValueError("work_stealing_min_batch_gap must be at least 2")

        self.policy = policy
        self.node_id: str = None
        self.work_stealing = work_stealing
        self.work_stealing_min_batch_gap = work_stealing_min_batch_gap
        self._work_stealing_stats = WorkStealingStats()

        # Mapping: model_type -> [request*]
        self.model_queues: Dict[ModelSpec, list] = defaultdict(list)

        match policy:
            case "fcfs":
                self._add = self._add_fcfs
            case "sjf":
                self._add = self._add_sjf
            case _:
                raise ValueError(f"Unsupported decode dispatcher policy {policy}")

    def __repr__(self):
        return f"{self.node_id}|decode dispatcher"

    def get_work_stealing_stats(self) -> dict:
        return self._work_stealing_stats.snapshot()

    def get_decode_load_stats(self) -> dict:
        """Return per-engine cumulative load and cross-engine imbalance."""
        engines = sorted(
            (scheduler.get_runtime_stats() for scheduler in self.schedulers),
            key=lambda stats: stats["engine_id"],
        )
        cv_metrics = (
            "output_tokens",
            "active_seconds",
            "decode_steps",
            "turns_completed",
            "model_switches",
            "current_batch_count",
        )
        return {
            "engines": engines,
            "cv": {
                metric: coefficient_of_variation(
                    engine[metric] for engine in engines
                )
                for metric in cv_metrics
            },
        }

    def plan_work_steal(self, receiver: "DecodeScheduler") -> WorkStealingDecision:
        if not self.work_stealing:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.DISABLED
            )
        return select_work_steal(
            receiver,
            self.schedulers,
            min_batch_gap=self.work_stealing_min_batch_gap,
        )

    def apply_work_steal(self, plan: WorkStealingPlan) -> WorkStealingDecision:
        """Atomically transfer a revalidated batch at a donor round boundary."""
        if not self.work_stealing:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.DISABLED
            )

        receiver = plan.receiver
        donor = plan.donor
        if receiver is donor or not all(
            any(registered is scheduler for registered in self.schedulers)
            for scheduler in (receiver, donor)
        ):
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.STALE_PLAN
            )
        if not donor.at_work_stealing_boundary:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.DONOR_NOT_AT_BOUNDARY
            )

        receiver_load = receiver.get_load_snapshot()
        if not receiver_load.is_idle:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.RECEIVER_NOT_IDLE
            )

        donor_load = donor.get_load_snapshot()
        if (
            donor_load.batch_count - receiver_load.batch_count
            < self.work_stealing_min_batch_gap
        ):
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.INSUFFICIENT_LOAD_GAP
            )

        candidate = plan.candidate
        if not 0 < candidate.queue_index < len(donor.decode_batches):
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.STALE_PLAN
            )
        _, current_batch = donor.decode_batches[candidate.queue_index]
        if (
            current_batch is not candidate.batch
            or len(current_batch) != candidate.request_count
            or current_batch.get_num_total_tokens() != candidate.total_tokens
        ):
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.STALE_PLAN
            )

        owners = [
            (scheduler, queue_index)
            for scheduler in self.schedulers
            for queue_index, (_, batch) in enumerate(scheduler.decode_batches)
            if batch is candidate.batch
        ]
        if owners != [(donor, candidate.queue_index)]:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.STALE_PLAN
            )

        still_eligible = any(
            current.queue_index == candidate.queue_index
            and current.batch is candidate.batch
            for current in donor.get_stealable_batches()
        )
        if not still_eligible:
            return WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.NO_ELIGIBLE_BATCH
            )

        del donor.decode_batches[candidate.queue_index]
        receiver.decode_batches.append((0, candidate.batch))
        donor.record_work_steal_donated()
        receiver.record_work_steal_received()
        return WorkStealingDecision(plan=plan)

    def _select_work_steal_from(
        self, donor: "DecodeScheduler"
    ) -> WorkStealingDecision:
        """Select and apply at most one transfer without recording metrics."""
        receivers = sorted(
            (
                scheduler
                for scheduler in self.schedulers
                if scheduler is not donor and scheduler.get_load_snapshot().is_idle
            ),
            key=lambda scheduler: scheduler.engine.engine_id,
        )
        if not receivers:
            decision = WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.NO_RECEIVER
            )
        else:
            accepted = []
            rejected = []
            for receiver in receivers:
                decision = select_work_steal(
                    receiver,
                    (receiver, donor),
                    min_batch_gap=self.work_stealing_min_batch_gap,
                )
                if decision.accepted:
                    accepted.append(decision)
                else:
                    rejected.append(decision)

            if accepted:
                decision = min(
                    accepted,
                    key=lambda candidate: (
                        0 if candidate.plan.model_affinity else 1,
                        candidate.plan.receiver_load.engine_id,
                        candidate.plan.candidate.queue_index,
                    ),
                )
                decision = self.apply_work_steal(decision.plan)
            else:
                decision = rejected[0]

        return decision

    def try_work_steal_from(
        self, donor: "DecodeScheduler"
    ) -> WorkStealingDecision:
        """Try one transfer from ``donor`` and record the boundary outcome."""
        self._work_stealing_stats.record_attempt()
        try:
            decision = self._select_work_steal_from(donor)
        except Exception:
            decision = WorkStealingDecision(
                rejection_reason=WorkStealingRejectionReason.INTERNAL_ERROR
            )
            logger.exception(
                "work_steal event=error donor_engine=%s",
                donor.engine.engine_id,
            )

        if decision.accepted:
            self._work_stealing_stats.record_success()
            plan = decision.plan
            logger.info(
                "work_steal event=success donor_engine=%s receiver_engine=%s "
                "model=%s requests=%s tokens=%s affinity=%s",
                plan.donor_load.engine_id,
                plan.receiver_load.engine_id,
                plan.candidate.batch.model,
                plan.candidate.request_count,
                plan.candidate.total_tokens,
                plan.model_affinity,
            )
        else:
            self._work_stealing_stats.record_rejection(decision.rejection_reason)
            logger.info(
                "work_steal event=rejected donor_engine=%s reason=%s",
                donor.engine.engine_id,
                decision.rejection_reason.value,
            )
        return decision

    async def initialize(self):
        """
        Register decode schedulers as dispatch targets.
        """
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.schedulers = [
            engine.scheduler for engine in Controller.decode_engines().values()
        ]
        if self.work_stealing:
            for scheduler in self.schedulers:
                scheduler.set_work_stealing_callback(self.try_work_steal_from)

    def add(self, batch: BatchedRequests):
        """
        Add the given batch to the dispatcher queues, waiting to be dispatched.
        """
        for request in batch.requests:
            assert not request.is_finished
            self._add(request)

    def _add_fcfs(self, request: Request):
        self.model_queues[request.model].append(request)

    def _add_sjf(self, request: Request):
        heapq.heappush(
            self.model_queues[request.model], (request.get_total_len(), request)
        )

    def dispatch(self) -> bool:
        """
        Dispatch requests in the dispatcher queue to decode schedulers,
        until no requests are left.
        """

        dispatched = False
        for queue in self.model_queues.values():
            for i in range(len(queue)):
                dispatched = True
                request = queue[i] if isinstance(queue[i], Request) else queue[i][1]
                pick_scheduler: "DecodeScheduler" = None
                for scheduler in self.schedulers:
                    if scheduler.try_accept(request):
                        # ..accepted
                        pick_scheduler = scheduler
                        break
                    else:
                        # ..not accepted by an existing batch; find a scheduler
                        # with the lowest load
                        if pick_scheduler is None or len(
                            scheduler.decode_batches
                        ) < len(pick_scheduler.decode_batches):
                            pick_scheduler = scheduler
                else:
                    pick_scheduler.decode_batches.append(
                        (0, BatchedRequests([request]))
                    )
                logger.info(
                    f"({self}) request {request.request_id} -> ({pick_scheduler.engine})"
                )
            queue.clear()

        return dispatched

    # TODO: API for re-dispatch of a batch

    async def start_event_loop(self):
        async def dispatch_loop():
            while True:
                if not self.dispatch():
                    await asyncio.sleep(SLEEP_WHEN_DISPATCH_PENDS)
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await dispatch_loop()
