import simpy
from collections import deque
from . import *
from .request import Request, RequestState
from .estimator import PrefillEstimator, DecodeEstimator, make_estimator
from typing_extensions import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.basicConfig()

PF = 1
BS1 = 1

PCIE = 16

class Worker:
    """GPU workers for simulation.
    
    Workers and Scheduler work in unison to simulate request dispatching.
    To facilitate a large variety of scheduler implementations, we adopt the
    following framework (inspired by QLM):

    *   Requests that are not finished are pooled in a global buffer; each
        Worker has a separate queue that references the Requests
    
    *   Scheduler implementations purge finished / failed requests from the
        global buffer, and adjust the per-worker queues

    *   Workers remain in an event loop where one iteration (prefill or decode
        batch) is executed in each cycle. At the start of each cycle, a worker
        looks at the (perhaps new) front of the queue, and update states accordingly.
        - If the front is a series of requests that matches the current model
          of the worker, execute as many of them as one batch
        - If the front consists of requests that differs from the current model,
          a model swap is performed; the previous batch, if not finished, is
          implicitly evicted, achieving request reordering
    
    The core of this framework is how Scheduler arranges the Worker queues, which
    can be used to emulate many scheduling algorithms.

    A few further remarks:
    *   There is currently no way to explicitly limit the size of a worker batch
        if all requests at the front are of the same model. The scheduler can
        always assign exceeding requests to other workers if the batch size is
        too big; and in the case where no worker can handle them, the system is
        overloaded anyway.
    *   Re-arrangement of the queues has currently one LIMITATOON: the Requests
        that are being executed must only be moved back in the same queue, but
        not to a different queue. This eliminates inter-worker synchronization:
        otherwise a new worker would have to wait for a Request that is still
        being executed by another worker.
    """

    def __init__(
        self,
        env: 'simpy.Environment' = None,
        worker_id: int = None,
        device: DeviceType = DeviceType.a10,    
        swap_overhead_ms: int = 0,
        max_batch_size: int = 1024,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        assert worker_id is not None, "Worker ID must be set"
        self.env = env
        self.worker_id = worker_id
        self.device = device
        self.swap_overhead_ms = swap_overhead_ms
        self.max_batch_size = min(1024, max_batch_size)

        # Current model
        self.model: Optional[ModelType] = None

        # Worker queue for pending requests
        self.queue: 'deque[Request]' = deque()

        # Estimators
        self.prefill_est: Optional[PrefillEstimator] = None
        self.decode_est: Optional[DecodeEstimator] = None

        # Wakeup event  
        self._wakeup_event = env.event()

        # Metrics
        self.time_last_batch_ms: int = 0
        self.batch_size_overtime: list[Tuple[int, int]] = []

    def __str__(self):
        return (
            f'Worker('
            f'id={self.worker_id},'
            f'device={self.device},'
            f')'
        )
    
    __repr__ = __str__

    def wakeup(self):
        """Trigger the wakeup event of this worker"""
        self._wakeup_event.succeed()
        self._wakeup_event = self.env.event()

    def dump_state(self):
        queue = '\n\t'.join([str(req) for req in self.queue])
        print((
            f'Worker('
            f'id={self.worker_id},'
            f'device={self.device},'
            f'model={self.model},'
            f'queue={queue},'
            f')'
        ))

    def run(self):
        """Worker execution loop.
        
        Take Requests from the queue, potentially swap, and execute the batch
        for one iteration.
        """
        while True:
            self.do_purge()
            if self.queue:
                if self.queue[0].model == self.model:
                    # Form a new batch
                    prefill_batch, decode_batch = self.make_batch()
                    self.do_evict(prefill_batch, decode_batch)

                    batch = prefill_batch if prefill_batch else decode_batch
                    yield from self.do_batch(batch)
                else:
                    # Perform a swap for the front request
                    yield from self.do_swap(self.queue[0].model)
            else:
                # Sleep until there is something
                yield self._wakeup_event

    def make_batch(self) -> Tuple[list[Request], list[Request]]:
        """Pack requests at the front of the queue into a batch.
        
        The current behavior imitates the vLLM scheduler: requests are first prefilled
        FCFS until the KV cache is full; then decode batches are formed. This does imply
        that this part of the worker queue is implicitly re-ordered.

        This is to prioritize prefill jobs; otherwise the prefill jobs suffer from HOL
        blocking, and TTFT degrades massively.
        """
        prefill_batch: list[Request] = []
        decode_batch: list[Request] = []

        # Respect max_batch_size
        max_batch_size = min(self.max_batch_size, len(self.queue))

        # Pack requests into the batch, prioritizing prefill jobs
        accum_tokens = 0
        accum_batch_tokens = 0
        max_tokens = ModelType.max_num_tokens(self.model, device=self.device)
        max_batch_tokens = ModelType.max_num_tokens(self.model, device=self.device)

        for i in range(max_batch_size):
            req: 'Request' = self.queue[i]
            if req.model != self.model:
                break

            sched_tokens = req.current_context_len
            sched_batch_tokens = sched_tokens if req.is_prefill else 1
            if sched_tokens > max_batch_tokens:
                raise ValueError(
                    f'[ts={self.env.now}ms] Request length {sched_tokens} exceeds max-batch-tokens {max_batch_tokens}')
            
            # ..try and fit as many requests as possible
            if accum_tokens + sched_tokens > max_tokens \
                or accum_batch_tokens + sched_batch_tokens > max_batch_tokens:
                break

            accum_tokens += sched_tokens
            accum_batch_tokens += sched_batch_tokens
            assert req.model == self.model and not req.is_finished and not req.is_failed
            if req.is_prefill:
                prefill_batch.append(req)
            else:
                assert req.is_decode
                decode_batch.append(req)
        
        return (prefill_batch, decode_batch)

    def do_purge(self):
        """Purge finished / failed requests from the queue"""
        for _ in range(len(self.queue)):
            req: 'Request' = self.queue.popleft()
            if req.is_failed or req.is_finished:
                continue
            else:
                self.queue.append(req)
    
    def do_evict(
        self, 
        prefill_batch: list[Request],
        decode_batch: list[Request],
    ):
        """Evict previously executed requests if not in the current batch"""
        for req in self.queue:
            if req.state == RequestState.EXECUTING \
                and req not in prefill_batch \
                and req not in decode_batch:
                # print(f'[ts={self.env.now}ms] evicting {req}')
                req.evict()
    
    def do_batch(self, batch: list[Request]):
        """Process one batch"""

        assert self.model is not None
        assert all(self.model == req.model for req in batch)

        prefill_reqs: list[Request] = []
        decode_reqs: list[Request] = []
        for req in batch:
            req.sched()
            if req.is_prefill:
                prefill_reqs.append(req)
            else:
                decode_reqs.append(req)

        if prefill_reqs:
            batch = prefill_reqs
            # Get prefill latency w.r.t. the prefill requests
            prefill_len_list = [req.current_context_len for req in prefill_reqs]
            latency_ms = int(1000 * self.prefill_est.predict(-1, -1, prefill_len_list=prefill_len_list))
            if latency_ms < 0:
                print(self.model, prefill_len_list)
            logger.debug(f'<{self}> doing prefill batch ({self.model}) {[req.req_id for req in batch]} ({latency_ms})')
        else:
            # Get decode latency w.r.t. the decode requests
            decode_len_list = [req.current_context_len for req in decode_reqs]
            latency_ms = int(1000 * self.decode_est.predict(-1, -1, decode_len_list=decode_len_list))
            logger.debug(f'<{self}> doing decode batch ({self.model}) {[req.req_id for req in batch]} ({latency_ms})')

        yield self.env.timeout(latency_ms)
        assert all(not req.is_finished for req in batch), f'<{self}> {batch}'
        for req in batch:
            req.step()
            if req.is_finished:
                logger.debug(f'<{self}> req {req.req_id} is finished')

        self.time_last_batch_ms = int(self.env.now)
        return

    def do_swap(self, model: ModelType):
        """Swap the pending model onto the worker"""
        assert self.model != model
        logger.debug(f'<{self}> swapping {self.model} -> {model}')

        # TODO: cleanup overhead
        if self.model is not None:
            pass

        # All requests are evicted
        for req in self.queue:
            req.evict()

        # Set this beforehand to indicate that swapping is initiated
        self.model = model
        self.prefill_est = make_estimator(PrefillEstimator, self.model, self.device)
        self.decode_est = make_estimator(DecodeEstimator, self.model, self.device)

        # Wait for swapping
        # TODO: loading/init overhead
        C = 1
        latency = int(model.mem_size() / PCIE * 1000 * C) + self.swap_overhead_ms
        yield self.env.timeout(latency)
        # yield self.env.timeout(1)
        
        return

"""
Below are workers specifically for the Diswitch policy.

Unlike the base `Worker`, these workers can do their own scheduling
beyond reordering prefills. These workers also distinguish between 
prefill and decode roles; each role only accepts the corresponding
requests.

The worker-level scheduling functions as follows:
-   For the decode role, 
    -   Form batches that are as large as possible (as permitted by 
        KV-cache size limit).
    -   Decide the time slice for each batch, such that the TPOT SLO
        is maximally satisfied, while the time slice does not exceed
        some limit
    -   Execute these batches in a round-robin way (each batch gets
        one time slice as a "turn"; a "round" refers to performing one 
        "turn" each decode batch)
-   For the prefill role,
    -   Form batches that are as large as possible
    -   FCFS/SJF/EDF are both valid scheduling strategies. 
    -   TODO: QLM may be the best here?
"""

class PrefillWorker(Worker):

    # Scheduler; for invoking re-schedule when a prefill batch finished
    scheduler: Any

    max_group_sz: int = 0
    group_sz: list[int] = []

    def __init__(
        self,
        scheduler,
        env: 'simpy.Environment' = None,
        worker_id: int = None,
        device: DeviceType = DeviceType.a10,
        swap_overhead_ms: int = 0,
    ):
        super().__init__(env, worker_id=worker_id, device=device,
                         swap_overhead_ms=swap_overhead_ms)
        self.scheduler = scheduler

        # The queue is to be interpreted as a queue of batches
        self.queue: deque[list[Request]] = deque()

        self.prefetch_time_ms = 0

    def __str__(self):
        return (
            f'PrefillWorker('
            f'id={self.worker_id},'
            f'device={self.device},'
            f')'
        )
    
    def _batch_fits(self, batch: list[Request], req: Request) -> bool:
        return sum(r.current_context_len for r in batch) + req.current_context_len \
            <= ModelType.max_num_tokens(req.model, device=self.device)

    def run(self):
        """
        Worker execution loop.
        """
        
        while True:
            if not self.queue:
                # ..no requests; wait
                yield self._wakeup_event
                continue

            # Limit bs=1
            # TODO: make this nicer
            self.max_group_sz = max(self.max_group_sz, max(len(g) for g in self.queue))
            self.group_sz.append(self.max_group_sz)
            if BS1:
                if len(self.queue[0]) == 1:
                    prefill_batch = self.queue.popleft() 
                else:
                    prefill_batch = [self.queue[0].pop(0)]
            else:
                prefill_batch = self.queue.popleft() 

            # prefill_batch = self.queue.popleft()
            # ..do the batch
            model = prefill_batch[0].model
            if self.model != model:
                # swap if needed
                logger.debug(f'<{self}> swap {self.model} -> {model}')
                yield from self._do_swap(model)
            yield from self._do_prefill(prefill_batch)

            # ..invoke a re-schedule of the decodes
            self.scheduler.schedule()
            
            # while len(self.queue) == 0:
            #     # ..might as well decode a bit when no prefill is waiting
            #     decode_batch = list(filter(lambda req: not req.is_finished, prefill_batch))
            #     if len(decode_batch) == 0:
            #         break
            #     yield from self._do_decode(decode_batch)
            # else:
            #     # ..invoke a re-schedule of the decodes
            #     self.scheduler.schedule()
    
    def _do_swap(self, model: ModelType):
        assert model != self.model
        assert model is not None

        latency_ms = int(model.mem_size() / PCIE * 1000) + self.swap_overhead_ms
        if PF: latency_ms = max(10, latency_ms - self.prefetch_time_ms)
        yield self.env.timeout(latency_ms)
        self.prefetch_time_ms = 0

        self.model = model
        return 
    
    def _do_prefill(self, prefill_batch: list[Request]):
        logger.debug(f'<{self}> prefill batch {[req.req_id for req in prefill_batch]} starts')

        model = prefill_batch[0].model
        assert self.model == model
        for req in prefill_batch:
            req.sched()
        
        prefill_len_list = [req.current_context_len for req in prefill_batch]
        est = make_estimator(PrefillEstimator, model=model, device=self.device)
        latency_ms = int(1000 * est.predict(-1, -1, prefill_len_list=prefill_len_list))
        
        yield self.env.timeout(latency_ms)
        self.prefetch_time_ms += latency_ms
        for req in prefill_batch:
            req.step()

        logger.debug(f'<{self}> prefill batch {[req.req_id for req in prefill_batch]} ends')     
        return
    
    def _do_decode(self, decode_batch: list[Request]):
        model = decode_batch[0].model
        assert self.model == model

        est = make_estimator(DecodeEstimator, model=model, device=self.device)
        decode_len_list = [req.current_context_len for req in decode_batch]
        latency_ms = int(1000 * est.predict(-1, -1, decode_len_list=decode_len_list))
        yield self.env.timeout(latency_ms)
        for req in decode_batch:
            req.step()

            if req.is_finished:
                logger.info(f'<{self}> req {req.req_id} finished;')
        return

class DecodeWorker(Worker):

    # Scheduler; for invoking re-schedule when a decode batch is migrated
    scheduler: Any

    # Actual set of requests
    requests: set[Request]

    # Round and turn counter
    round: int 
    turn: int

    # Batches for the current round
    decode_batches: deque[Tuple[int, list[Request]]]

    n_batches = []
    alphas = []

    # Target TPOT
    tpot_slo_ms: int
    # Maximum time for a time slice
    max_time_slice_ms: int

    def __init__(
        self,
        scheduler,
        env: 'simpy.Environment' = None,
        worker_id: int = None,
        device: DeviceType = DeviceType.a10,
        tpot_slo_ms: int = 100,
        max_time_slice_ms: int = 4000,
        swap_overhead_ms: int = 0,
    ):
        super().__init__(env, worker_id=worker_id, device=device,
                         swap_overhead_ms=swap_overhead_ms)
        self.scheduler = scheduler

        self.round = 0
        self.turn = 0
        self.requests = set()
        self.decode_batches = deque()
        self.tpot_slo_ms = tpot_slo_ms
        self.max_time_slice_ms = max_time_slice_ms

        self.prefetch_time_ms = 0

    def __str__(self):
        return (
            f'DecodeWorker('
            f'id={self.worker_id},'
            f'device={self.device},'
            f')'
        )
    
    def run(self):
        """
        Worker execution loop.
        """
        
        while True:
            # Perform the round
            self._enter_round()

            if not self.decode_batches:
                # ..no requests; wait
                yield self._wakeup_event
                continue
            
            start = int(self.env.now)
            n_turn = len(self.decode_batches)
            while self.turn < n_turn:
                # Top of a turn
                
                # ..pick a batch
                # for i in range(n_turn - self.turn):
                #     _, batch = self.decode_batches[i]
                #     if batch[0].model == self.model:
                #         self.decode_batches[0], self.decode_batches[i] = \
                #         self.decode_batches[i], self.decode_batches[0]
                #         break
                decode_batch = self.decode_batches[0]
                
                # ..do the batch
                model = decode_batch[1][0].model
                if self.model != model:
                    # swap if needed
                    logger.debug(f'<{self}> (round {self.round}|{self.turn}) swap {self.model} -> {model}')
                    yield from self._do_swap(model)
                logger.debug(f'<{self}> (round {self.round}|{self.turn}) decode batch {decode_batch[0]}, {[(req.req_id, req.current_context_len) for req in decode_batch[1]]}')
                yield from self._do_decode()

                # Bottom of a turn
                self.turn += 1
                self.decode_batches.rotate(-1)
                # self._try_migrate()
                self._accept(to_batch=True)
            
            end = int(self.env.now)
            logger.debug(f'<{self}> (round {self.round}) round takes {end-start} in total')

    def _try_migrate(self):
        self.scheduler.policy._try_migrate_decode(
            self,
            [worker for worker in self.scheduler.workers if isinstance(worker, DecodeWorker) and worker != self]
        ) 

    def _do_swap(self, model: ModelType):
        assert model != self.model
        assert model is not None

        latency_ms = int(model.mem_size() / PCIE * 1000) + self.swap_overhead_ms
        if PF: latency_ms = max(10, latency_ms - self.prefetch_time_ms)
        yield self.env.timeout(latency_ms)
        self.prefetch_time_ms = 0
        
        self.model = model
        self._accept(to_batch=True)
        return

    def _do_decode(self):
        decode_batch = self.decode_batches[0][1]
        model = decode_batch[0].model
        assert self.model == model

        time_used_ms = 0
        decode_est = make_estimator(DecodeEstimator, model=model, device=self.device)
        prefill_est = make_estimator(PrefillEstimator, model=model, device=self.device)

        steps = 0
        
        ALPHA = 1
        num_tokens = sum(req.current_context_len for req in decode_batch)
        yield self.env.timeout(int(num_tokens * ALPHA * 0.5 / PCIE))

        while time_used_ms < self.decode_batches[0][0]:
            decode_len_list = [req.current_context_len for req in decode_batch if not req.is_finished]
            if len(decode_len_list) == 0:
                # early stop
                break
            steps += 1

            if prefill_len_list := [req.current_context_len for req in decode_batch if req.is_prefill]:
                latency_ms = int(1000 * prefill_est.predict(-1, -1, prefill_len_list=prefill_len_list))
            else:
                latency_ms = int(1000 * decode_est.predict(-1, -1, decode_len_list=decode_len_list))

            yield self.env.timeout(latency_ms)
            self.prefetch_time_ms += latency_ms

            for req in decode_batch:
                if req.is_finished:
                    continue
                req.step()

                if req.is_finished:
                    logger.info(f'<{self}> (round {self.round}|{self.turn}) req {req.req_id} finished;')
                    self.requests.remove(req)
            time_used_ms += latency_ms
            self._accept(to_batch=True)

        logger.debug(f'<{self}> (round {self.round}|{self.turn}) {steps} steps in decode batch')
        return

    def _batch_fits(self, batch: list[Request], req: Request) -> bool:
        return sum(r.current_context_len for r in batch if not r.is_finished) + req.current_context_len \
            <= ModelType.max_num_tokens(req.model, device=self.device)
    
    def _accept(self, to_batch: bool = False):
        """Accept new decode requests dispatched from the main scheduler"""
        # Add new requests
        accepted = len(self.queue) > 0
        while self.queue:
            req = self.queue.popleft()
            assert not req.is_finished

            # req.kvcache = False
            self.requests.add(req)

            if to_batch:
                for _, batch in self.decode_batches:
                    if batch[0].model == req.model and self._batch_fits(batch, req):
                        batch.append(req)
                        break
                else:
                    self.decode_batches.append((0, [req]))

    def _enter_round(self):
        """Top of a round; setup the decode batches for the round"""

        self._accept()

        # Form decode batches
        decode_reqs: list[Request] = list(self.requests)
        decode_batches: list[list[Request]] = []
        decode_reqs.sort(key=lambda req:req.time_arrival_ms)
        for req in decode_reqs:
            for batch in decode_batches:
                if batch[0].model == req.model and self._batch_fits(batch, req):
                    batch.append(req)
                    break
            else:
                decode_batches.append([req])

        decode_batches.sort(key=lambda batch: id(batch[0].model))
        self.decode_batches.clear()
        self.decode_batches.extend((0, batch) for batch in decode_batches)
        
        self._assign_time_slices()
        if self.decode_batches:
            self.round += 1
        self.turn = 0
    
    def _assign_time_slices(self):
        if len(self.decode_batches) == 0:
            return

        # ..compute alpha: how well can we satisfy the SLOs?
        d = []  # decode latency for one token
        for _, batch in self.decode_batches:
            est = make_estimator(DecodeEstimator, model=batch[0].model, device=self.device)
            d.append(est.predict(-1, -1, [req.current_context_len for req in batch]))
        n = [self.tpot_slo_ms / 1000 / d[i] for i in range(len(d))]    # tpot / d[i]
    
        n_prod = 1
        for i in range(len(n)):
            n_prod *= n[i]
        
        n_sum = 0
        for i in range(len(n)):
            n_sum += n_prod / n[i]

        C0 = self.swap_overhead_ms / 1000
        THETA = 1
        for model in set(batch[0].model for _, batch in self.decode_batches):
            # TODO: maybe correct this coefficient
            C0 += ModelType.mem_size(model) / PCIE * THETA
        alpha = max(
            C0 / min(n) / (self.max_time_slice_ms / 1000) + sum(1/n[i] for i in range(len(n))),
            0.5,
        )

        logger.debug(f'<{self}> n={n}, C0={C0}, alpha={alpha}')
        self.n_batches.append(len(self.decode_batches))
        self.alphas.append(int(10 * alpha) / 10)

        # ..from which we obtain the time slice that each batch should run for
        for i in range(len(d)):
            time_slice_ms = int(C0 * (n_prod / n[i]) / (n_prod * alpha - n_sum) * 1000)
            assert time_slice_ms > 0
            self.decode_batches[i] = (time_slice_ms, self.decode_batches[i][1])
