import time
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)
from collections import deque

from aegaeon.model_registry import ModelSpec
from aegaeon.config import get_model_config
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.estimator import cache_estimators, make_estimator, DecodeEstimator
from aegaeon.block_manager import BLOCK_LOCATION_CPU, BLOCK_SIZE
from aegaeon.scheduler_capacity import fit_decode_batch
from aegaeon.utils import TPOT_SLO, estimate_switch_time, prod

if TYPE_CHECKING:
    from aegaeon.stage_engine import DecodeEngine

logger = init_logger(__name__)


@dataclass(frozen=True)
class DecodeSchedulerLoad:
    engine_id: int
    resident_model: Optional[ModelSpec]
    batch_count: int
    request_count: int
    total_tokens: int

    @property
    def is_idle(self) -> bool:
        return self.batch_count == 0


@dataclass
class DecodeSchedulerRuntimeStats:
    rounds_completed: int = 0
    turns_completed: int = 0
    decode_steps: int = 0
    output_tokens: int = 0
    active_seconds: float = 0.0
    model_switches: int = 0
    work_steals_donated: int = 0
    work_steals_received: int = 0

    def snapshot(self) -> dict:
        """Return a detached, serialization-friendly metrics snapshot."""
        return {
            "rounds_completed": self.rounds_completed,
            "turns_completed": self.turns_completed,
            "decode_steps": self.decode_steps,
            "output_tokens": self.output_tokens,
            "active_seconds": self.active_seconds,
            "model_switches": self.model_switches,
            "work_steals_donated": self.work_steals_donated,
            "work_steals_received": self.work_steals_received,
        }


@dataclass(frozen=True)
class StealableDecodeBatch:
    queue_index: int
    quota: float
    batch: BatchedRequests
    request_count: int
    total_tokens: int


class DecodeScheduler:
    """
    Scheduler for the decode stage.

    NOTE: PP is not supported for this stage.
    """

    def __init__(
        self,
        engine: "DecodeEngine",
        max_batch_quota: float = 4.0,
        batch_lookahead: int = 16 * BLOCK_SIZE,
    ):
        self.engine = engine
        self.max_batch_quota = max_batch_quota
        self.batch_lookahead = batch_lookahead

        # Round and turn
        self.round: int = 0
        self.turn: int = 0
        self._turn_start: Optional[float] = None
        self._turn_request_moved: bool = False
        self._at_work_stealing_boundary: bool = False
        self._work_stealing_callback: Optional[
            Callable[["DecodeScheduler"], None]
        ] = None
        self._runtime_stats = DecodeSchedulerRuntimeStats()

        # Batches for the current round
        self.decode_batches: deque[Tuple[float, BatchedRequests]] = deque()
        # Admit arrivals without changing a batch already handed to workers.
        # Merge them at the next decode step, while preserving its time quota.
        self._pending_admissions: dict[BatchedRequests, List[Request]] = {}

        # Internal quota states for logging
        self.n: List[float] = []
        self.c0: float = 0
        self.alpha: float = 0

        # Execution loop
        self.loop: AsyncIterable[Tuple[BatchedRequests, Optional[ModelSpec]]] = (
            self._run()
        )

    def get_load_snapshot(self) -> DecodeSchedulerLoad:
        """Return a detached summary without changing scheduling state."""
        batches = tuple(batch for _, batch in self.decode_batches)
        model_config = self.engine.model_config
        return DecodeSchedulerLoad(
            engine_id=self.engine.engine_id,
            resident_model=(model_config.model if model_config is not None else None),
            batch_count=len(batches),
            request_count=sum(
                len(batch) + len(self._pending_admissions.get(batch, ()))
                for batch in batches
            ),
            total_tokens=sum(
                batch.get_num_total_tokens()
                + sum(
                    request.get_total_len()
                    for request in self._pending_admissions.get(batch, ())
                )
                for batch in batches
            ),
        )

    def record_decode_step(self, output_tokens: int, active_seconds: float) -> None:
        """Record one completed worker invocation without affecting scheduling."""
        if output_tokens < 0 or active_seconds < 0:
            raise ValueError("decode runtime metrics must be non-negative")
        self._runtime_stats.decode_steps += 1
        self._runtime_stats.output_tokens += output_tokens
        self._runtime_stats.active_seconds += active_seconds

    def record_work_steal_donated(self) -> None:
        self._runtime_stats.work_steals_donated += 1

    def record_work_steal_received(self) -> None:
        self._runtime_stats.work_steals_received += 1

    def get_runtime_stats(self) -> dict:
        """Return cumulative execution metrics plus the current queue load."""
        load = self.get_load_snapshot()
        return {
            "engine_id": load.engine_id,
            "resident_model": (
                str(load.resident_model) if load.resident_model is not None else None
            ),
            "current_batch_count": load.batch_count,
            "current_request_count": load.request_count,
            "current_total_tokens": load.total_tokens,
            **self._runtime_stats.snapshot(),
        }

    @property
    def at_work_stealing_boundary(self) -> bool:
        return self._at_work_stealing_boundary

    def set_work_stealing_callback(
        self, callback: Callable[["DecodeScheduler"], None]
    ) -> None:
        self._work_stealing_callback = callback

    @contextmanager
    def work_stealing_boundary(self) -> Iterator[None]:
        """Mark the synchronous end-of-round ownership-transfer boundary."""
        if self._at_work_stealing_boundary:
            raise RuntimeError("already at work-stealing boundary")
        self._at_work_stealing_boundary = True
        try:
            yield
        finally:
            self._at_work_stealing_boundary = False

    def get_stealable_batches(self) -> Tuple[StealableDecodeBatch, ...]:
        """Return safe non-head batch candidates without mutating the queue."""
        candidates = []
        for queue_index, (quota, batch) in enumerate(self.decode_batches):
            if (
                queue_index == 0
                or len(batch) == 0
                or batch.start_time is not None
                or batch in self._pending_admissions
            ):
                continue
            if not self.engine.block_manager.are_all_requests_at_location(
                batch.requests, BLOCK_LOCATION_CPU
            ):
                continue
            candidates.append(
                StealableDecodeBatch(
                    queue_index=queue_index,
                    quota=quota,
                    batch=batch,
                    request_count=len(batch),
                    total_tokens=batch.get_num_total_tokens(),
                )
            )
        return tuple(candidates)

    def _get_block_needed(self, length: int):
        return (length + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _batch_fits(self, batch: BatchedRequests, req: Request) -> bool:
        # TODO: temporary fix
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != req.model
        ):
            return True

        requests = [
            *batch.requests,
            *self._pending_admissions.get(batch, ()),
            req,
        ]
        return (
            sum(
                self.engine.block_manager.get_num_blocks_needed_for_request(
                    request,
                    self.engine.engine_id,
                    additional_tokens=self.batch_lookahead,
                )
                for request in requests
            )
            # TODO: this is buggy; must pre-compute max_blocks for all batches
            <= self.engine.block_manager.get_gpu_num_max_blocks_from_model(
                self.engine.engine_id, req.model
            )
        )

    def _batch_fits_strict(self, batch: BatchedRequests) -> bool:
        if not batch.requests:
            return True
        model = batch.requests[0].model
        return sum(
            self.engine.block_manager.get_num_blocks_needed_for_request(
                request,
                self.engine.engine_id,
                additional_tokens=0,
            )
            for request in batch.requests
        ) <= self.engine.block_manager.get_gpu_num_max_blocks_from_model(
            self.engine.engine_id, model
        )

    def try_accept(self, request: Request) -> bool:
        """Try accepting a new decode request to an existing batch."""
        for queue_index, (_, batch) in enumerate(self.decode_batches):
            if queue_index != 0 and batch.start_time is not None:
                continue
            if batch.model == request.model and self._batch_fits(batch, request):
                if queue_index == 0:
                    self._pending_admissions.setdefault(batch, []).append(request)
                else:
                    batch.add_request(request)
                return True
        return False

    def _drain_pending_admissions(self) -> None:
        """Join accepted arrivals only after the previous worker step finishes."""
        if not self._pending_admissions:
            return
        queued_batches = {batch for _, batch in self.decode_batches}
        for batch, requests in tuple(self._pending_admissions.items()):
            if batch.start_time is not None:
                continue
            self._pending_admissions.pop(batch)
            for request in requests:
                batch.add_request(request)
            # Keep ownership even if an empty head was retired at a boundary.
            if batch not in queued_batches:
                self.decode_batches.append((0, batch))
                queued_batches.add(batch)

    def pop_finished_requests(self) -> List[Request]:
        """Pop finished requests from the current batch."""
        return self.decode_batches[0][1].pop_finished_requests()

    async def get_next_batch(self) -> BatchedRequests:
        """Get the decode batch for the next step.

        Optionally return prefetch model for the next turn; turn and round may be
        incremented."""

        # Do this before resuming the generator: it may retire an empty head
        # or finish its turn as soon as it resumes after the previous step.
        self._drain_pending_admissions()
        decision: Tuple[BatchedRequests, Optional[ModelSpec]] = await anext(self.loop)
        batch, prefetch_model = decision
        if len(batch) == 0:
            return BatchedRequests()

        # Issue engine switch if needed
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != batch.model
        ):
            # No request is logically on a decode engine's GPU at this
            # time (`_run` will issue move-outs at end of turns).
            self._runtime_stats.model_switches += 1
            await self.engine.switch(
                get_model_config(batch.model),
                prefetch_model_config=get_model_config(prefetch_model),
            )

        # Check that the batch still fits; this might not be true when
        # a batch grows too much within its quota.
        if not self._batch_fits_strict(batch):
            # XXX: ..this should be rare
            logger.info(
                f"({self.engine}) batch {[r.request_id for r in batch.requests]} truncated"
            )
            await fit_decode_batch(self, batch)

        if not batch.requests:
            return batch

        # We have the batch; issue move-in operations
        try:
            request_moved = await self.engine.block_manager.move_requests(
                batch.requests, self.engine.engine_id
            )
        except ResourceWarning as exc:
            await self.engine.fail_requests(
                list(batch.requests), f"Decode swap-in allocation failed: {exc}"
            )
            batch.pop_finished_requests()
            return BatchedRequests()
        self._turn_request_moved = self._turn_request_moved or request_moved

        return batch

    async def _run(self) -> AsyncIterable[Tuple[BatchedRequests, Optional[ModelSpec]]]:
        """Main execution loop."""
        while True:
            # Start of the round
            self._enter_round()

            if len(self.decode_batches) == 0:
                # ..no requests; wait
                yield (BatchedRequests(), None)
                continue

            self.print_status()
            n_turn = len(self.decode_batches)
            while self.turn < n_turn and len(self.decode_batches) > 0:
                # ..pick a batch

                n_step = 0
                start = time.time()
                while True:
                    quota, batch = self.decode_batches[0]
                    if len(batch) == 0:
                        # ..all finished
                        break

                    if n_step == 0:
                        logger.info(
                            f"({self.engine}) <round {self.round}|{self.turn}> {[r.request_id for r in batch.requests]} {quota}"
                        )

                    # ..determine prefetch
                    prefetch_model = None
                    for i in range(1, n_turn - self.turn):
                        if i >= len(self.decode_batches):
                            break
                        if self.decode_batches[i][1].model != batch.model:
                            prefetch_model = self.decode_batches[i][1].model
                            break
                    yield (batch, prefetch_model)

                    n_step += 1
                    # ..by now the batch is already stepped once; starts counting quota
                    if self._turn_start is None:
                        self._turn_start = time.time()
                    if self._turn_start + quota < time.time():
                        # ..out of quota
                        break

                # ..turn ends
                # XXX: wait for all move-ins before we end this turn.
                # This enables the block manager to clear all move-in events at a
                # model switch.
                end = time.time()
                logger.info(
                    f"({self.engine}) <round {self.round}|{self.turn}> {n_step} steps; e2e {end-start:.2f}s; expected {quota:.2f}s; first-step {self._turn_start-start:.2f}s"
                )
                if self._turn_request_moved:
                    await asyncio.wait(
                        self.engine._remote_call_all_workers_async(
                            "wait_for_all_move_in"
                        )
                    )
                if len(self.decode_batches) > 1:
                    # XXX: optimization; no need to move out requests if
                    # this is the only batch.
                    try:
                        await self.engine.block_manager.move_requests(
                            batch.requests, BLOCK_LOCATION_CPU
                        )  # TODO: re-dispatch
                    except ResourceWarning as exc:
                        await self.engine.fail_requests(
                            list(batch.requests), f"Decode swap-out allocation failed: {exc}"
                        )
                        batch.pop_finished_requests()
                # Arrivals may have been staged while waiting for move-in or
                # swap-out above. Preserve them before rotating/removing head.
                self._drain_pending_admissions()
                self.turn += 1
                self._turn_start = None
                self._turn_request_moved = False
                if len(batch) > 0:
                    self.decode_batches.rotate(-1)
                else:
                    self.decode_batches.popleft()
                self._runtime_stats.turns_completed += 1

            self._runtime_stats.rounds_completed += 1

            if (
                self._work_stealing_callback is not None
                and len(self.decode_batches) >= 2
            ):
                try:
                    with self.work_stealing_boundary():
                        self._work_stealing_callback(self)
                except Exception:
                    logger.exception(
                        "(%s) work-stealing callback failed; continuing scheduling",
                        self.engine,
                    )

    def _enter_round(self):
        """Top of a round; form the decode batches for the round."""
        self._drain_pending_admissions()
        if len(self.decode_batches) == 0:
            return

        # Form decode batches
        decode_reqs: List[Request] = sum(
            (batch.requests for _, batch in self.decode_batches), []
        )
        decode_batches: List[BatchedRequests] = []
        decode_reqs.sort()  # Sort in FCFS order

        for req in decode_reqs:
            for batch in decode_batches:
                if batch.model == req.model and self._batch_fits(batch, req):
                    batch.add_request(req)
                    break
            else:
                decode_batches.append(BatchedRequests([req]))

        # Ensure a fair order
        decode_batches.sort(key=lambda batch: id(batch.model))

        # Apply updates
        self.decode_batches.clear()
        self.decode_batches.extend((0, batch) for batch in decode_batches)
        self._assign_quota()
        self.round += 1
        self.turn = 0

    def _assign_quota(self):
        """Assign quota for the current decode batches."""
        if len(self.decode_batches) == 0:
            return

        # ..compute alpha: how well can we satisfy the SLOs?
        d = []  # decode latency for one token
        models: set[ModelSpec] = set()  # unique set of models
        for _, batch in self.decode_batches:
            if batch.model is None:
                continue
            est = make_estimator(
                DecodeEstimator, model=batch.model, device=self.engine.device_type
            )
            d.append(
                est.predict(-1, -1, [req.get_total_len() for req in batch.requests])
            )
            models.add(batch.model)
        n = [TPOT_SLO / d[i] for i in range(len(d))]  # tpot / d[i]
        self.n = n

        n_prod = prod(n)
        n_sum = sum(n_prod / ni for ni in n)

        c0 = sum(
            estimate_switch_time(
                model, self.engine.device_type, self.engine.parallel_config
            )
            for model in models
        )  # model switching overhead
        self.c0 = c0

        alpha = max(
            c0 / min(n) / self.max_batch_quota + sum(1 / ni for ni in n),
            0.5,
        )
        self.alpha = alpha

        # ..from which we obtain the quota that each batch should run for
        for i in range(len(d)):
            quota = c0 * (n_prod / n[i]) / (n_prod * alpha - n_sum)
            assert quota > 0
            self.decode_batches[i] = (quota, self.decode_batches[i][1])

    def print_status(self):
        batches = " |> ".join(
            [
                f"{[request.request_id for request in batch.requests]} {quota:.2f}"
                for quota, batch in self.decode_batches
            ]
        )
        logger.info(
            f"({self.engine}) batches: {batches} "
            f"(n={self.n}, c0={self.c0}, alpha={self.alpha})"
        )
