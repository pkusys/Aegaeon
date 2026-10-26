import copy
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from aegaeon.config import get_model_config
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.block_manager import BLOCK_LOCATION_CPU, BLOCK_SIZE
from aegaeon.scheduler_capacity import request_capacity_error

if TYPE_CHECKING:
    from aegaeon.stage_engine import PrefillEngine

logger = init_logger(__name__)


class PrefillScheduler(ABC):
    """
    Scheduler for the prefill stage.
    """

    # The corresponding prefill engine
    engine: "PrefillEngine"

    # The number of on-the-fly (i.e. processing) request blocks
    # Adds when calling get_next_batch_and_pop()
    # Subtracts when calling on_finish_requests()
    num_on_fly_request_block: int = 0

    # Queue for waiting requests, grouped by prefill dispatcher as hints
    waiting_queue: List[List[Request]]

    # Max size of a group in waiting_queue
    max_group_size: int

    # # Estimated time(stamp) for completing all current requests in the waiting_queue.
    # completion_time: float = 0

    def _get_block_needed(self, length: int):
        return (length + BLOCK_SIZE - 1) // BLOCK_SIZE

    def add_request(self, request: Request) -> None:
        """
        Add a request to the scheduler.
        """
        for group in self.waiting_queue:
            if group[0].model == request.model and len(group) < self.max_group_size:
                group.append(request)
                break
        else:
            self.waiting_queue.append([request])

    # def _estimate_completion_time(self, request: Request) -> float:
    #     """
    #     Estimate the new completion time, assuming `request` is added.
    #     """
    #     prev_model: Optional[ModelType] = None
    #     if self.waiting_queue:
    #         prev_model = self.waiting_queue[-1].model
    #     elif self.engine.model_config is not None:
    #         prev_model = self.engine.model_config.model

    #     switch_time = 0
    #     if request.model != prev_model:
    #         switch_time += estimate_switch_time(request.model,
    #                                             self.engine.device_type,
    #                                             self.engine.parallel_config)

    #     est = make_estimator(PrefillEstimator, request.model, self.engine.device_type)
    #     execution_time = est.predict(request.get_input_len(), 1)

    #     if request.arrival_time > self.completion_time:
    #         return request.arrival_time + switch_time + execution_time
    #     else:
    #         return self.completion_time + switch_time + execution_time

    @abstractmethod
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get the next batch for the prefill stage and pop them from the wait queue.
        """
        raise NotImplementedError()

    async def on_finish_requests(self, batch: BatchedRequests):
        # At the end of a prefill batch, swap out unfinished requests,
        # and decrease the on_fly counter.
        try:
            await self.engine.block_manager.move_requests(
                [request for request in batch.requests if not request.is_finished],
                BLOCK_LOCATION_CPU,
            )
        except ResourceWarning as exc:
            await self.engine.fail_requests(
                [request for request in batch.requests if not request.is_finished],
                f"Prefill swap-out allocation failed: {exc}",
            )
        finally:
            self.num_on_fly_request_block -= sum(
                [self._get_block_needed(req.get_input_len()) for req in batch.requests]
            )

    @abstractmethod
    def print_status(self) -> None:
        """
        Print the status of the scheduler.
        """
        raise NotImplementedError()


class PrefillStageUniBatchScheduler(PrefillScheduler):
    """
    A prefill scheduler that always form batches with one request.
    """

    def __init__(
        self,
        engine: "PrefillEngine",
        max_group_size: int = 8,
    ):
        self.engine = engine
        self.max_group_size = max_group_size
        self.waiting_queue = []

    async def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get the next batch for the prefill stage and pop them from the wait_queue.
        """
        if len(self.waiting_queue) == 0:
            return BatchedRequests()

        # The previous uni-batch has moved out. Still check a single request's
        # physical demand after switch and tokenization; it can exceed the pool.
        if len(self.waiting_queue[0]) == 1:
            request = self.waiting_queue.pop(0)[0]
        else:
            request = self.waiting_queue[0].pop(0)

        # Issue engine switch if needed
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != request.model
        ):
            # No request is logically on a prefill engine's GPU at this
            # time (`on_finish_requests` will issue move-outs).
            assert self.num_on_fly_request_block == 0

            # Determine prefetch
            prefetch_model = None
            for group in self.waiting_queue:
                if group[0].model != request.model:
                    prefetch_model = group[0].model
                    break
            await self.engine.switch(
                get_model_config(request.model),
                prefetch_model_config=get_model_config(prefetch_model),
            )

        # HACK: get the correct prompt_token_ids here using tokenizer,
        # which is not possible before. However, various other decisions
        # already depended on the request's input token length.
        if request.prompt is not None:
            request.prompt_token_ids = self.engine.tokenizer.encode(request.prompt)
            request.prompt = None

        error = request_capacity_error(self.engine, request, "prefill")
        if error is not None:
            await self.engine.fail_requests([request], error)
            return BatchedRequests()

        next_batch = BatchedRequests([request])
        self.num_on_fly_request_block += self._get_block_needed(request.get_input_len())

        return next_batch

    def print_status(self):
        if len(self.waiting_queue) > 0:
            logger.info(
                f"({self.engine}) waiting: {[request.request_id for group in self.waiting_queue for request in group]}, "
                f"{self.num_on_fly_request_block} blocks occupied by on-the-fly requests"
            )


def get_prefill_stage_scheduler(engine: "PrefillEngine") -> PrefillScheduler:
    match engine.policy:
        case "uni":
            return PrefillStageUniBatchScheduler(engine)
        case _:
            raise NotImplementedError(f"Unsupported prefill policy {engine.policy}.")
