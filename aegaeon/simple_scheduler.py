import time
import asyncio
import itertools
import math
from typing import List, Optional, Tuple, AsyncIterable, TYPE_CHECKING
from collections import defaultdict, deque

from aegaeon.model_registry import ModelSpec
from aegaeon.config import get_model_config, BLOCK_SIZE
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.estimator import make_estimator, DecodeEstimator
from aegaeon.block_manager import BLOCK_LOCATION_CPU
from aegaeon.scheduler_capacity import fit_decode_batch, request_capacity_error
from aegaeon.utils import TPOT_SLO, estimate_switch_time

if TYPE_CHECKING:
    from aegaeon.stage_engine import SimpleEngine

logger = init_logger(__name__)

SLEEP_WHEN_NO_REQUEST = 0.003


def calculate_decode_quotas(
    predicted_step_times: List[float],
    switch_time: float,
    max_batch_quota: float,
) -> Tuple[List[float], List[float], float]:
    """Return numerically stable per-model decode quotas."""
    fallback = [max_batch_quota] * len(predicted_step_times)
    if (
        not predicted_step_times
        or not math.isfinite(switch_time)
        or switch_time <= 0
        or any(
            not math.isfinite(value) or value <= 0
            for value in predicted_step_times
        )
    ):
        return fallback, [], 0.0

    n = [TPOT_SLO / value for value in predicted_step_times]
    if any(not math.isfinite(value) or value <= 0 for value in n):
        return fallback, n, 0.0

    inverse_sum = sum(1 / value for value in n)
    alpha = max(
        switch_time / min(n) / max_batch_quota + inverse_sum,
        0.5,
    )
    slack = alpha - inverse_sum
    if not math.isfinite(slack) or slack <= 0:
        return fallback, n, alpha

    quotas = [
        min(max_batch_quota, switch_time / value / slack)
        for value in n
    ]
    if any(not math.isfinite(value) or value <= 0 for value in quotas):
        return fallback, n, alpha
    return quotas, n, alpha


class SimpleScheduler:
    """Scheduler for simple mode (no PD separation).

    Uses a two-phase round/turn model:
      1. Prefill phase: process all new requests model-by-model.
         After each prefill step, unfinished requests are swapped to CPU.
      2. Decode phase: round/turn cooperative scheduling with SLO quotas.
         Each turn swaps the batch from CPU to GPU, runs within its time
         quota, then swaps back to CPU at turn end.

    New prefills arriving during a round are deferred to the next round.
    """

    def __init__(
        self,
        engine: "SimpleEngine",
        max_batch_quota: float = 4.0,
        batch_lookahead: int = 16 * BLOCK_SIZE,
    ):
        self.engine = engine
        self.max_batch_quota = max_batch_quota
        self.batch_lookahead = batch_lookahead

        self.waiting_queue: dict[ModelSpec, list[Request]] = defaultdict(list)
        self.active_requests: dict[ModelSpec, list[Request]] = defaultdict(list)

        self.round: int = 0
        self.turn: int = 0
        self._turn_start: Optional[float] = None
        self._turn_request_moved: bool = False

        self.decode_batches: deque[Tuple[float, BatchedRequests]] = deque()

        self.n: List[float] = []
        self.c0: float = 0
        self.alpha: float = 0

        self.loop: AsyncIterable[Tuple[BatchedRequests, Optional[ModelSpec]]] = (
            self._run()
        )

    def _get_block_needed(self, length: int):
        return (length + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _batch_fits(self, batch: BatchedRequests, req: Request) -> bool:
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != req.model
        ):
            return True

        requests = [*batch.requests, req]
        return sum(
            self.engine.block_manager.get_num_blocks_needed_for_request(
                request,
                self.engine.engine_id,
                additional_tokens=self.batch_lookahead,
            )
            for request in requests
        ) <= self.engine.block_manager.get_gpu_num_max_blocks_from_model(
            self.engine.engine_id, req.model
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

    def add_request(self, request: Request):
        self.waiting_queue[request.model].append(request)

    def pop_finished_requests(self) -> List[Request]:
        finished = []
        if self.decode_batches:
            finished = self.decode_batches[0][1].pop_finished_requests()
        return finished

    async def get_next_batch(self) -> BatchedRequests:
        decision: Tuple[BatchedRequests, Optional[ModelSpec]] = await anext(self.loop)
        batch, prefetch_model = decision
        if len(batch) == 0:
            return BatchedRequests()

        # Model switch
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != batch.model
        ):
            await self.engine.switch(
                get_model_config(batch.model),
                prefetch_model_config=get_model_config(prefetch_model),
            )

        # Prefill is already sized below, before yielding it to the engine.
        if batch.requests[0].is_prefill_stage():
            return batch

        # Batch overflow truncation
        if not self._batch_fits_strict(batch):
            logger.info(
                f"({self.engine}) batch {[r.request_id for r in batch.requests]} truncated"
            )
            await fit_decode_batch(self, batch)

        if not batch.requests:
            return batch

        # Swap-in from CPU (only for decode phase; prefill phase is handled
        # by SimpleEngine._step allocating blocks directly)
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
        while True:
            # --- Prefill phase ---
            models_with_prefills = [m for m, reqs in self.waiting_queue.items() if reqs]
            if models_with_prefills:
                models_with_prefills.sort(key=str)

                # Free the pool for new prefills, including those for the
                # resident model. Existing decodes must not consume its budget.
                current_model = (
                    self.engine.model_config.model if self.engine.model_config else None
                )
                if current_model is not None:
                    all_live = [
                        r
                        for reqs in itertools.chain(
                            self.active_requests.values(),
                            (batch.requests for _, batch in self.decode_batches),
                        )
                        for r in reqs
                        if not r.is_finished
                    ]
                    if all_live:
                        try:
                            await self.engine.block_manager.move_requests(
                                all_live, BLOCK_LOCATION_CPU
                            )
                        except ResourceWarning as exc:
                            await self.engine.fail_requests(
                                all_live, f"Prefill swap-out allocation failed: {exc}"
                            )

                for model in models_with_prefills:
                    reqs = self.waiting_queue[model][:]
                    self.waiting_queue[model].clear()

                    if (
                        self.engine.model_config is None
                        or self.engine.model_config.model != model
                    ):
                        await self.engine.switch(get_model_config(model))

                    for req in reqs:
                        if req.prompt is not None:
                            req.prompt_token_ids = self.engine.tokenizer.encode(
                                req.prompt
                            )
                            req.prompt = None

                    capacity = self.engine.block_manager.get_gpu_num_max_blocks_from_model(
                        self.engine.engine_id, model
                    )
                    chunks = []
                    chunk, used = [], 0
                    for req in reqs:
                        error = request_capacity_error(self.engine, req, "prefill")
                        if error is not None:
                            await self.engine.fail_requests([req], error)
                            continue
                        needed = self.engine.block_manager.get_num_blocks_needed_for_request(
                            req, self.engine.engine_id
                        )
                        if chunk and used + needed > capacity:
                            chunks.append(chunk)
                            chunk, used = [], 0
                        chunk.append(req)
                        used += needed
                    if chunk:
                        chunks.append(chunk)

                    for chunk in chunks:
                        prefill_batch = BatchedRequests(chunk)
                        logger.info(
                            f"({self.engine}) prefill {[r.request_id for r in chunk]}"
                        )
                        yield (prefill_batch, None)

                        live = [r for r in chunk if not r.is_finished]
                        if live:
                            try:
                                await self.engine.block_manager.move_requests(
                                    live, BLOCK_LOCATION_CPU
                                )
                            except ResourceWarning as exc:
                                await self.engine.fail_requests(
                                    live, f"Prefill swap-out allocation failed: {exc}"
                                )
                            else:
                                self.active_requests[model].extend(live)

                # Prefill phase end: swap all live requests to CPU.
                all_active = [
                    r
                    for reqs in itertools.chain(
                        self.active_requests.values(),
                        (batch.requests for _, batch in self.decode_batches),
                    )
                    for r in reqs
                    if not r.is_finished
                ]
                if all_active:
                    try:
                        await self.engine.block_manager.move_requests(
                            all_active, BLOCK_LOCATION_CPU
                        )
                    except ResourceWarning as exc:
                        await self.engine.fail_requests(
                            all_active, f"Prefill swap-out allocation failed: {exc}"
                        )

                continue

            # --- Decode phase ---
            self._enter_round()

            if len(self.decode_batches) == 0:
                yield (BatchedRequests(), None)
                await asyncio.sleep(SLEEP_WHEN_NO_REQUEST)
                continue

            self.print_status()
            n_turn = len(self.decode_batches)
            while self.turn < n_turn and len(self.decode_batches) > 0:
                n_step = 0
                start = time.time()
                while True:
                    quota, batch = self.decode_batches[0]
                    if len(batch) == 0:
                        break

                    if n_step == 0:
                        logger.info(
                            f"({self.engine}) <round {self.round}|{self.turn}> "
                            f"{[r.request_id for r in batch.requests]} {quota:.2f}"
                        )

                    prefetch_model = None
                    for i in range(1, n_turn - self.turn):
                        if i >= len(self.decode_batches):
                            break
                        if self.decode_batches[i][1].model != batch.model:
                            prefetch_model = self.decode_batches[i][1].model
                            break
                    yield (batch, prefetch_model)

                    n_step += 1
                    if self._turn_start is None:
                        self._turn_start = time.time()
                    if self._turn_start + quota < time.time():
                        break

                end = time.time()
                logger.debug(
                    f"({self.engine}) <round {self.round}|{self.turn}> "
                    f"{n_step} steps; e2e {end-start:.2f}s; expected {quota:.2f}s; "
                    f"first-step {self._turn_start-start:.2f}s"
                )
                if self._turn_request_moved:
                    await asyncio.wait(
                        self.engine._remote_call_all_workers_async(
                            "wait_for_all_move_in"
                        )
                    )
                # Swap-out only if there are other batches (same as PD DecodeScheduler).
                # Single-model decode: blocks stay on GPU across steps and rounds.
                if len(self.decode_batches) > 1:
                    try:
                        await self.engine.block_manager.move_requests(
                            batch.requests, BLOCK_LOCATION_CPU
                        )
                    except ResourceWarning as exc:
                        await self.engine.fail_requests(
                            list(batch.requests), f"Decode swap-out allocation failed: {exc}"
                        )
                        batch.pop_finished_requests()
                self.turn += 1
                self._turn_start = None
                self._turn_request_moved = False
                if len(batch) > 0:
                    self.decode_batches.rotate(-1)
                else:
                    self.decode_batches.popleft()

    def _enter_round(self):
        # Merge active_requests (new prefills since last round) with
        # existing decode_batches (still-live requests from previous round).
        decode_by_model: dict[ModelSpec, list[Request]] = defaultdict(list)

        for _, batch in self.decode_batches:
            live = [r for r in batch.requests if not r.is_finished]
            if live:
                decode_by_model[batch.model].extend(live)

        for model, reqs in self.active_requests.items():
            live = [r for r in reqs if not r.is_finished]
            if live:
                decode_by_model[model].extend(live)

        decode_batches: List[BatchedRequests] = []
        for model in sorted(decode_by_model.keys(), key=str):
            decode_batches.append(BatchedRequests(decode_by_model[model]))

        self.decode_batches.clear()
        self.decode_batches.extend((0, batch) for batch in decode_batches)
        self._assign_quota()
        self.round += 1
        self.turn = 0
        self.active_requests.clear()

    def _assign_quota(self):
        if len(self.decode_batches) == 0:
            return

        d = []
        models: set[ModelSpec] = set()
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
        c0 = sum(
            estimate_switch_time(
                model, self.engine.device_type, self.engine.parallel_config
            )
            for model in models
        )
        self.c0 = c0
        quotas, n, alpha = calculate_decode_quotas(d, c0, self.max_batch_quota)
        self.n = n
        self.alpha = alpha

        for i, quota in enumerate(quotas):
            self.decode_batches[i] = (quota, self.decode_batches[i][1])

    def print_status(self):
        batches = " |> ".join(
            [
                f"{[request.request_id for request in batch.requests]} {quota:.2f}"
                for quota, batch in self.decode_batches
            ]
        )
        logger.debug(
            f"({self.engine}) batches: {batches} "
            f"(n={self.n}, c0={self.c0}, alpha={self.alpha})"
        )
