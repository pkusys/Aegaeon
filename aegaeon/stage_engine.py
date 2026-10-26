import os
import time
import copy
from typing import Callable, Optional, List
from abc import ABC, abstractmethod
import asyncio

from enum import Enum
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from aegaeon.logger import init_logger
from aegaeon.config import (
    ModelConfig,
    ParallelConfig,
    QuickLoaderConfig,
    BLOCK_SIZE,
    get_vllm_config,
    get_server_config,
)
from aegaeon.request import (
    Request,
    RequestCapacityError,
    BatchedRequests,
)
from aegaeon.utils import GB, get_distributed_init_method, get_open_port
from aegaeon.device_registry import DeviceSpec
from aegaeon.lifetime import LifetimeEvent, LifetimeEventType
from aegaeon.worker import Worker
from aegaeon.prefill_scheduler import PrefillScheduler, get_prefill_stage_scheduler
from aegaeon.decode_scheduler import DecodeScheduler

logger = init_logger(__name__)


class Stage(Enum):
    """The two stages of LLM serving."""

    PREFILL = "prefill"
    DECODE = "decode"
    SIMPLE = "simple"

    def __str__(self) -> str:
        return self.value


# Sleep for this many seconds when there is no request in PrefillStageLLMEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_PREFILL_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodeStageLLMEngine.step()
SLEEP_WHEN_DECODE_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1


class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(
        self,
        request: Request,
        new_token: str,
        new_token_id: int,
        dispatch_time: float,
        issue_time: float,
        step_time: float,
        block_overhead: Optional[float] = None,
    ):
        self.request = request
        self.request_id = request.request_id
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

        self.dispatch_time = dispatch_time
        self.issue_time = issue_time
        self.step_time = step_time
        self.block_overhead = block_overhead

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )


class StageEngine(ABC):
    """
    StageEngine: An engine that runs either the prefill stage or the decode stage.

    This class is the base class for PrefillEngine and DecodeEngine.
    """

    @abstractmethod
    def _get_scheduler(self) -> PrefillScheduler | DecodeScheduler:
        raise NotImplementedError()

    def __init__(
        self,
        stage: Stage,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
    ):
        self.engine_id = engine_id
        assert engine_id >= 0, f"valid engine_id should be >= 0"
        self.stage = stage
        self.model_config: Optional[ModelConfig] = None  # initialized later
        self.tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        self.parallel_config = parallel_config
        self.loader_config = loader_config
        self.device_ids = device_ids
        self.placement_group = placement_group

        self.scheduler: PrefillScheduler | DecodeScheduler = self._get_scheduler()

        self.on_new_step_output_callback = on_new_step_output_callback
        self.on_new_lifetime_event_callback = on_new_lifetime_event_callback
        self.on_requests_finished_callback = on_requests_finished_callback

        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
        self.switch_futures = []
        self.switch_started_at: Optional[float] = None

    def __repr__(self):
        return f"{self.node_id}|{self.stage} engine #{self.engine_id}"

    def _remote_call_all_workers_async(self, func_name: str, *args, **kwargs):
        """
        Call func_name asynchronously on all workers; return the futures immediately.
        """
        handlers = []
        for stage in self.workers:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args, **kwargs))
        return handlers

    async def _wait_for_request_moves(self, requests: List[Request]) -> None:
        """Complete pending DMA before a rejected request's blocks are freed."""
        if not any(
            request.request_id in self.block_manager.block_tables
            or request.request_id in self.block_manager.latest_events
            for request in requests
        ):
            return
        from aegaeon.llm import Controller

        # CPU-resident blocks may still be receiving a swap-out from a
        # different stage engine. Wait on every worker on this controller,
        # rather than only this engine's workers.
        for method in ("wait_for_all_move_in", "wait_for_all_move_out"):
            await asyncio.gather(*Controller.remote_call_all_workers_async(method))

    async def fail_requests(
        self, requests: List[Request], reason: str | RequestCapacityError
    ) -> None:
        """Terminate rejected requests and notify the controller without tokens."""
        failed_requests = [request for request in requests if not request.is_finished]
        if not failed_requests:
            return
        # The reclaimer treats a finished request as proof that its previous
        # moves completed, so the barrier must precede both finished and free.
        await self._wait_for_request_moves(failed_requests)
        for request in failed_requests:
            request.fail(reason)
            logger.warning("(%s) %s", self, request.failure)
        self._remote_call_all_workers_async(
            "release_requests",
            [request.request_id for request in failed_requests],
        )
        self.on_requests_finished_callback(failed_requests)

    async def initialize(self):
        """Initialize workers.

        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between engines.
        """
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.block_manager = Controller.block_manager()

        logger.info(f"({self}) initializing workers")

        # For each pipeline stage, create tensor_parallel_size workers
        # ach worker will be assigned a GPU;
        # Workers are guaranteed to reside in a single node,
        # so the loopback address is sufficient.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )

        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []

            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j

                worker_id = i * self.parallel_config.tensor_parallel_size + j
                worker = Worker.options(
                    num_cpus=get_server_config().worker_num_cpus,
                    resources={self.node_id: 0.01},
                    runtime_env=Controller.worker_runtime_env(
                        enable_foundry=self.stage == Stage.DECODE
                    ),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=self.engine_id,
                    ),
                ).remote(
                    stage=self.stage,
                    node_id=self.node_id,
                    engine_id=self.engine_id,
                    worker_id=worker_id,
                    device_id=self.device_ids[worker_id],
                    parallel_config=tmp_parallel_config,
                    loader_config=self.loader_config,
                    cpu_cache_filename=Controller.cpu_cache_filename(),
                    cpu_cache_size=Controller.cpu_cache_size(),
                    config_path=Controller.config_path(),
                )
                workers.append(worker)
            self.workers.append(workers)

        # gather device type
        device_types = await asyncio.gather(
            *self._remote_call_all_workers_async("get_device_type")
        )
        unique_device_type = set(device_types)
        if len(unique_device_type) > 1:
            raise ValueError(
                f"Heterogeneous device types for engine #{self.engine_id}: {unique_device_type}"
            )
        self.device_type: DeviceSpec = device_types[0]
        self.block_manager._set_device(self.device_type)
        self.block_manager._set_parallel_config(self.parallel_config)

        # cache estimators
        from aegaeon.estimator import cache_estimators, DecodeEstimator

        cache_estimators(
            DecodeEstimator,
            Controller.cached_models(),
            self.device_type,
        )

        # initialize distributed groups
        handlers = []
        workers_1d = sum(self.workers, [])
        for rank, worker in enumerate(workers_1d):
            handlers.append(
                worker.initialize.remote(
                    distributed_init_method=distributed_init_method,
                    rank=rank,
                    local_rank=self.device_ids[rank],
                )
            )

        # This finishes worker initialization
        await asyncio.gather(*handlers)

    async def switch(
        self,
        model_config: ModelConfig,
        enable_quick_loader: bool = True,
        prefetch_model_config: Optional[ModelConfig] = None,
        prefetch_enable_quick_loader: bool = True,
    ):
        """
        Switch the engine and its workers to the given model config.

        Load the model, and also initialize GPU kv-cache.
        """
        if self.model_config and self.model_config.model == model_config.model:
            return

        from aegaeon.llm import Controller

        await Controller.defrag_event().wait()

        logger.info("({}) switching to {}".format(self, model_config.model))

        prefetch_scratch_bytes = 0
        if prefetch_model_config is not None and prefetch_enable_quick_loader:
            prefetch_scratch_bytes = self.loader_config.get_model_loading_scratch_size(
                prefetch_model_config.model.path()
            )
            if prefetch_scratch_bytes:
                logger.info(
                    "(%s) reserving %.2f GiB of QuickLoader scratch for %s",
                    self,
                    prefetch_scratch_bytes / GB,
                    prefetch_model_config.model,
                )

        try:
            cache_memory_bytes = model_config.get_available_cache_memory_bytes(
                self.parallel_config,
                self.device_type,
                prefetch_model_config=prefetch_model_config,
            )
            cache_memory_bytes -= prefetch_scratch_bytes
        except MemoryError:
            # A speculative prefetch must never prevent the scheduled model
            # itself from running.
            cache_memory_bytes = 0

        # QuickLoader needs one checkpoint shard as temporary GPU scratch while
        # constructing the prefetched model. Keep that space in addition to at
        # least 1 GiB of usable KV cache.
        if cache_memory_bytes < GB:
            if prefetch_model_config is not None:
                logger.warning(
                    "(%s) skipping prefetch of %s: only %.2f GiB would remain "
                    "for KV cache after reserving %.2f GiB of loader scratch",
                    self,
                    prefetch_model_config.model,
                    cache_memory_bytes / GB,
                    prefetch_scratch_bytes / GB,
                )
            prefetch_model_config = None
            cache_memory_bytes = model_config.get_available_cache_memory_bytes(
                self.parallel_config, self.device_type
            )
        # num_gpu_blocks = 128
        vllm_config = get_vllm_config(
            model_config=model_config,
            parallel_config=self.parallel_config,
        )
        pf_vllm_config = get_vllm_config(
            model_config=prefetch_model_config,
            parallel_config=self.parallel_config,
        )

        Controller._on_engine_switch(
            engine_id=self.engine_id,
            stage=str(self.stage),
            previous_model=self.model_config.model if self.model_config else None,
            next_model=model_config.model,
            prefetch_model=(
                prefetch_model_config.model if prefetch_model_config else None
            ),
        )
        success = self._remote_call_all_workers_async(
            "switch",
            cache_memory_bytes,
            model_config,
            vllm_config,
            enable_quick_loader=enable_quick_loader,
            prefetch_model_config=prefetch_model_config,
            prefetch_enable_quick_loader=prefetch_enable_quick_loader,
            prefetch_vllm_config=pf_vllm_config,
        )
        self.switch_futures = success
        self.switch_started_at = time.time()
        # ``switch`` runs inside an async Ray actor.  Blocking the actor event
        # loop with ``ray.get`` stalls dispatch and request bookkeeping for the
        # whole model-load interval (Ray also emits a warning for this).  Ray
        # ObjectRefs are awaitable, so keep the event loop responsive here.
        cache_registrations = await asyncio.gather(*success)
        num_gpu_blocks, block_shape, cache_groups = cache_registrations[0]
        if any(
            registration != cache_registrations[0]
            for registration in cache_registrations
        ):
            raise RuntimeError(
                f"TP ranks produced different cache layouts: {cache_registrations}"
            )

        self.model_config = model_config

        self.tokenizer = Controller.get_tokenizer(model_config.model.path())

        # Update block manager
        self.block_manager.register_gpu_space(
            self.engine_id,
            num_gpu_blocks,
            block_shape,
            cache_groups,
            self._remote_call_all_workers_async,
        )

    @abstractmethod
    async def start_event_loop(self):
        raise NotImplementedError()

    @abstractmethod
    async def print_status(self):
        raise NotImplementedError()


class PrefillEngine(StageEngine):
    """
    ## Event loops
    - Step
        - Call scheduler for the next batch (spin if none)
        - Invoke `step()` on workers
        - Await results
        - For unfinished requests:
            - Invoke swap-out on workers
            - Send the requests to DecodeDispatcher
        - For finished requests:
            - Free requests

    - Report
        - `print_status()`
    """

    def _get_scheduler(self) -> PrefillScheduler:
        return get_prefill_stage_scheduler(self)

    def __init__(
        self,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
        # The prefill scheduler policy
        policy: str = "uni",
    ):
        self.policy = policy
        super().__init__(
            Stage.PREFILL,
            engine_id,
            parallel_config,
            loader_config,
            device_ids,
            placement_group,
            on_new_step_output_callback,
            on_new_lifetime_event_callback,
            on_requests_finished_callback,
        )

        # All the batched requests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []

    async def _step(self):
        """
        Run prefill on the batch of requests chosen by the scheduler.

        NOTE: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # Get the next batch from scheduler
        dispatch_time = time.time()
        batched_requests = await self.scheduler.get_next_batch_and_pop()

        if len(batched_requests):
            try:
                await self.block_manager.allocate_blocks_batched(
                    batched_requests, self.engine_id
                )
            except ResourceWarning as exc:
                await self.fail_requests(
                    batched_requests.requests, f"prefill KV-cache allocation failed: {exc}"
                )
                await self.scheduler.on_finish_requests(batched_requests)
                batched_requests.pop_finished_requests()
                batched_requests = BatchedRequests()

        if len(batched_requests) == 0:
            # ..no new batch to serve; spin
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_PREFILL_NO_REQUEST)
        else:
            logger.debug(
                f"({self}) serving {[request.request_id for request in batched_requests.requests]} "
                f"({sum(request.get_input_len() for request in batched_requests.requests)} tokens)"
            )

            # ..log down the lifetime event
            for request in batched_requests.requests:
                self.on_new_lifetime_event_callback(
                    request.request_id, LifetimeEvent(LifetimeEventType.PrefillBegin)
                )

            # ..push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                [req.meta() for req in batched_requests.requests],
                [
                    self.block_manager.get_block_table(request.request_id).group_blocks
                    for request in batched_requests.requests
                ],
                # time_blocked=True, # TODO: configure this
            )
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # ..no request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                # ..wait for the results
                issue_time = time.time()
                maybe_generated_tokens_ids, block_times = (
                    await self.batches_ret_futures[0]
                )
                step_time = time.time()

                finished_batch = self.batches_in_pipeline[0]
                for request in finished_batch.requests:
                    if request.eos_token_id is None and self.tokenizer is not None:
                        request.eos_token_id = getattr(
                            self.tokenizer, "eos_token_id", None
                        )
                finished_batch.finish_one_iteration(maybe_generated_tokens_ids)

                # ..invoke scheduler callback (move out requests and clear states)
                try:
                    await self.scheduler.on_finish_requests(finished_batch)
                except ResourceWarning as exc:
                    await self.fail_requests(
                        finished_batch.requests,
                        f"prefill KV-cache transfer failed: {exc}",
                    )

                for i, (request, maybe_new_token_id) in enumerate(
                    zip(finished_batch.requests, maybe_generated_tokens_ids)
                ):
                    if maybe_new_token_id is not None and request.failure is None:
                        try:
                            token = self.tokenizer.decode(maybe_new_token_id)
                            # token = "<todo>"
                        except Exception as e:
                            logger.warning(
                                f"({self}) failed to decode token_id {maybe_new_token_id}; error: {e}"
                            )
                            token = "<nat>"
                        block_overhead = None if block_times is None else block_times[i]
                        step_output = StepOutput(
                            request,
                            token,
                            maybe_new_token_id,
                            dispatch_time,
                            issue_time,
                            step_time,
                            block_overhead=block_overhead,
                        )
                        self.on_new_lifetime_event_callback(
                            request.request_id,
                            LifetimeEvent(LifetimeEventType.PrefillEnd),
                        )
                        self.on_new_step_output_callback(
                            request.request_id,
                            step_output,
                        )

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

                # ..post processing
                decode_batch = BatchedRequests()
                finished_reqs: List[Request] = []
                for request in finished_batch.requests:
                    if not request.is_finished:
                        decode_batch.add_request(request)
                    else:
                        finished_reqs.append(request)
                self._remote_call_all_workers_async(
                    "release_requests",
                    [request.request_id for request in finished_batch.requests],
                )
                # ..free resources for finished requests
                self.on_requests_finished_callback(finished_reqs)

                # ..put decode requests into DecodeDispatcher
                if len(decode_batch) > 0:
                    from aegaeon.llm import Controller

                    dispatcher = Controller.decode_dispatcher()
                    if dispatcher is None:
                        raise NotImplementedError("cross-node dispatching")
                    logger.debug(
                        f"({self}) sending batch {[request.request_id for request in decode_batch.requests]} -> ({dispatcher})"
                    )
                    dispatcher.add(decode_batch)

    async def start_event_loop(self):
        async def prefill_step_loop():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        async def report_loop():
            while True:
                self.print_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(prefill_step_loop(), report_loop())

    def print_status(self):
        self.scheduler.print_status()


class DecodeEngine(StageEngine):
    """
    ## Event loops
    - Step
        - (top of round) Call Scheduler to group Requests into batches and assign time slices
        - (top of turn) Get the next batch and time slice
            - optionally, invoke model prefetch
            - optionally, re-dispatch waiting batches with DecodeDispatcher to other engines
        - (during turn)
            - Invoke `step()` on workers
            - Await results until time slice expires
    """

    def _get_scheduler(self) -> DecodeScheduler:
        return DecodeScheduler(self)

    def __init__(
        self,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
    ):
        super().__init__(
            Stage.DECODE,
            engine_id,
            parallel_config,
            loader_config,
            device_ids,
            placement_group,
            on_new_step_output_callback,
            on_new_lifetime_event_callback,
            on_requests_finished_callback,
        )

        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []

    async def _step(self) -> None:
        """
        Run one step of decode on the batch of requests chosen by the scheduler.

        NOTE: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # Get the next batch from scheduler
        #  - this may trigger move-in if some requests are not on GPU
        #  - this may also trigger move-out if the batch grows too much, or we're starting a new turn
        dispatch_time = time.time()
        batch = await self.scheduler.get_next_batch()

        if len(batch):
            try:
                await self.block_manager.allocate_blocks_batched(batch, self.engine_id)
            except ResourceWarning as exc:
                await self.fail_requests(
                    batch.requests, f"decode KV-cache allocation failed: {exc}"
                )
                batch.pop_finished_requests()
                batch = BatchedRequests()

        if len(batch) == 0:
            self.batches_in_pipeline.append(BatchedRequests())
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_DECODE_NO_REQUEST)
        else:
            # ..log down the lifetime event
            for request in batch.requests:
                self.on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.DecodeBegin),
                    True,
                )

            # ..push the batch into pipeline
            batch.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batch)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                [req.meta() for req in batch.requests],
                [
                    self.block_manager.get_block_table(request.request_id).group_blocks
                    for request in batch.requests
                ],
                # time_blocked=True, TODO: configure this
            )
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                # ..wait for the results
                issue_time = time.time()
                maybe_generated_tokens_ids, block_times = (
                    await self.batches_ret_futures[0]
                )
                step_time = time.time()

                self.scheduler.record_decode_step(
                    output_tokens=sum(
                        token_id is not None
                        for token_id in maybe_generated_tokens_ids
                    ),
                    active_seconds=step_time - dispatch_time,
                )

                finished_batch = self.batches_in_pipeline[0]
                for request in finished_batch.requests:
                    if request.eos_token_id is None and self.tokenizer is not None:
                        request.eos_token_id = getattr(
                            self.tokenizer, "eos_token_id", None
                        )
                finished_batch.finish_one_iteration(maybe_generated_tokens_ids)

                for i, (request, maybe_new_token_id) in enumerate(
                    zip(finished_batch.requests, maybe_generated_tokens_ids)
                ):
                    if maybe_new_token_id is not None:
                        try:
                            token = self.tokenizer.decode(maybe_new_token_id)
                            # token = "<todo>"
                        except Exception as e:
                            logger.warning(
                                f"({self}) failed to decode token_id {maybe_new_token_id}; error: {e}"
                            )
                            token = "<nat>"
                        block_overhead = None if block_times is None else block_times[i]
                        step_output = StepOutput(
                            request,
                            token,
                            maybe_new_token_id,
                            dispatch_time,
                            issue_time,
                            step_time,
                            block_overhead=block_overhead,
                        )
                        self.on_new_step_output_callback(
                            request.request_id,
                            step_output,
                        )
                        if request.is_finished:
                            self.on_new_lifetime_event_callback(
                                request.request_id,
                                LifetimeEvent(LifetimeEventType.DecodeEnd),
                            )

                # ..clear states for finished requests
                finished_reqs = self.scheduler.pop_finished_requests()
                if finished_reqs:
                    self._remote_call_all_workers_async(
                        "release_requests",
                        [request.request_id for request in finished_reqs],
                    )
                self.on_requests_finished_callback(finished_reqs)

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

    async def start_event_loop(self):
        async def decode_step_loop():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await asyncio.gather(decode_step_loop())

    def print_status(self):
        self.scheduler.print_status()


SLEEP_WHEN_SIMPLE_NO_REQUEST = 0.003


class SimpleEngine(StageEngine):
    """Engine for simple mode (no PD separation).

    A single engine pool handles both prefill and decode phases.
    Requests stay on the same engine throughout their lifecycle.
    """

    def _get_scheduler(self):
        from aegaeon.simple_scheduler import SimpleScheduler

        return SimpleScheduler(self)

    def __init__(
        self,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        on_requests_finished_callback: Callable[[List[Request]], None],
    ):
        super().__init__(
            Stage.SIMPLE,
            engine_id,
            parallel_config,
            loader_config,
            device_ids,
            placement_group,
            on_new_step_output_callback,
            on_new_lifetime_event_callback,
            on_requests_finished_callback,
        )
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []

    def __repr__(self):
        return f"{self.node_id}|simple engine #{self.engine_id}"

    async def initialize(self):
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.block_manager = Controller.block_manager()

        logger.info(f"({self}) initializing workers")

        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )

        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []

            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j

                worker_id = i * self.parallel_config.tensor_parallel_size + j
                worker = Worker.options(
                    num_cpus=get_server_config().worker_num_cpus,
                    resources={self.node_id: 0.01},
                    runtime_env=Controller.worker_runtime_env(
                        enable_foundry=False
                    ),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=self.engine_id,
                    ),
                ).remote(
                    stage=self.stage,
                    node_id=self.node_id,
                    engine_id=self.engine_id,
                    worker_id=worker_id,
                    device_id=self.device_ids[worker_id],
                    parallel_config=tmp_parallel_config,
                    loader_config=self.loader_config,
                    cpu_cache_filename=Controller.cpu_cache_filename(),
                    cpu_cache_size=Controller.cpu_cache_size(),
                    config_path=Controller.config_path(),
                )
                workers.append(worker)
            self.workers.append(workers)

        device_types = await asyncio.gather(
            *self._remote_call_all_workers_async("get_device_type")
        )
        unique_device_type = set(device_types)
        if len(unique_device_type) > 1:
            raise ValueError(
                f"Heterogeneous device types for engine #{self.engine_id}: {unique_device_type}"
            )
        self.device_type: DeviceSpec = device_types[0]
        self.block_manager._set_device(self.device_type)
        self.block_manager._set_parallel_config(self.parallel_config)

        from aegaeon.estimator import cache_estimators, DecodeEstimator

        cache_estimators(
            DecodeEstimator,
            Controller.cached_models(),
            self.device_type,
        )

        handlers = []
        workers_1d = sum(self.workers, [])
        for rank, worker in enumerate(workers_1d):
            handlers.append(
                worker.initialize.remote(
                    distributed_init_method=distributed_init_method,
                    rank=rank,
                    local_rank=self.device_ids[rank],
                )
            )

        await asyncio.wait(handlers)

    async def _step(self):
        dispatch_time = time.time()
        batched_requests = await self.scheduler.get_next_batch()

        if len(batched_requests):
            try:
                await self.block_manager.allocate_blocks_batched(
                    batched_requests, self.engine_id
                )
            except ResourceWarning as exc:
                await self.fail_requests(
                    batched_requests.requests, f"simple KV-cache allocation failed: {exc}"
                )
                batched_requests.pop_finished_requests()
                batched_requests = BatchedRequests()

        if len(batched_requests) == 0:
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_SIMPLE_NO_REQUEST)
        else:
            logger.debug(
                f"({self}) serving {[request.request_id for request in batched_requests.requests]} "
                f"({batched_requests.get_num_input_tokens()} tokens)"
            )

            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                [req.meta() for req in batched_requests.requests],
                [
                    self.block_manager.get_block_table(request.request_id).group_blocks
                    for request in batched_requests.requests
                ],
            )
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            if self.batches_ret_futures[0] is None:
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                issue_time = time.time()
                maybe_generated_tokens_ids, block_times = (
                    await self.batches_ret_futures[0]
                )
                step_time = time.time()

                finished_batch = self.batches_in_pipeline[0]
                for request in finished_batch.requests:
                    if request.eos_token_id is None and self.tokenizer is not None:
                        request.eos_token_id = getattr(
                            self.tokenizer, "eos_token_id", None
                        )
                finished_batch.finish_one_iteration(maybe_generated_tokens_ids)

                for i, (request, maybe_new_token_id) in enumerate(
                    zip(finished_batch.requests, maybe_generated_tokens_ids)
                ):
                    if maybe_new_token_id is not None:
                        try:
                            token = self.tokenizer.decode(maybe_new_token_id)
                        except Exception as e:
                            logger.warning(
                                f"({self}) failed to decode token_id {maybe_new_token_id}; error: {e}"
                            )
                            token = "<nat>"
                        block_overhead = None if block_times is None else block_times[i]
                        step_output = StepOutput(
                            request,
                            token,
                            maybe_new_token_id,
                            dispatch_time,
                            issue_time,
                            step_time,
                            block_overhead=block_overhead,
                        )
                        self.on_new_step_output_callback(
                            request.request_id,
                            step_output,
                        )
                        if request.is_finished:
                            self.on_new_lifetime_event_callback(
                                request.request_id,
                                LifetimeEvent(LifetimeEventType.DecodeEnd),
                            )

                finished_reqs = finished_batch.pop_finished_requests()
                if finished_reqs:
                    self._remote_call_all_workers_async(
                        "release_requests",
                        [request.request_id for request in finished_reqs],
                    )
                self.on_requests_finished_callback(finished_reqs)

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

    async def start_event_loop(self):
        async def simple_step_loop():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        async def report_loop():
            while True:
                self.scheduler.print_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(simple_step_loop(), report_loop())

    def print_status(self):
        self.scheduler.print_status()
