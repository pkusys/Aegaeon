from .request import Request, RequestState
from .estimator import *
from .worker import Worker, PrefillWorker, DecodeWorker
from . import ModelType
from abc import ABC, abstractmethod
from typing_extensions import Optional, Tuple, Any
from collections import deque
from queue import PriorityQueue

import simpy
import logging
import numpy as np
import itertools
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.basicConfig()

class Scheduler:

    def __init__(
        self,
        env: 'simpy.Environment',
        policy: 'Policy',
        workers: list[Worker],
        h2d: float, # TODO
    ):
        self.policy = policy
        self.env = env
        self.workers = workers

        # Global request queue
        self.requests: list[Request] = []
        
    
    def schedule(self):
        """Schedule requests in the global request queue"""
        
        # Purge finished / failed requests
        reqs: list[Request] = []
        for req in self.requests:
            if req.progress == 0 and (self.env.now - req.time_arrival_ms) > 15000:
                req.fail()
            if req.is_failed or req.is_finished:
                continue
            reqs.append(req)
        self.requests = reqs

        # Schedule using some policy
        schedule = self.policy.schedule(reqs, self.workers)

        # Sanity checks
        if not isinstance(self.policy, MixServePolicy):
            req_to_worker: dict[Request, Worker] = {}
            for worker in self.workers:
                for req in worker.queue:
                    req_to_worker[req] = worker
            for worker, reqs in schedule.items():
                for req in reqs:
                    if req.state == RequestState.EXECUTING and req_to_worker.get(req, worker) != worker:
                        raise ValueError(
                            f'{req} must not be in the queue of {worker} because it is currently being executed on {req_to_worker.get(req)}')
        
        # Submit the new plan
        for worker in self.workers:
            new_queue = schedule[worker]
            worker.queue = new_queue
            worker.wakeup()
    
    def _put_request(
        self, 
        at: int,
        req: Request,
    ):
        yield self.env.timeout(at)
        req.arrive()
        logger.debug(f'<Scheduler> req {req.req_id} arrives')
        self.requests.append(req)
        self.schedule()
    
    def put_workload(
        self,
        workload: list[Tuple[int, Request]],
    ):
        for at, req in workload:
            self.env.process(self._put_request(at, req))

class Policy(ABC):
    """Base class for scheduler policy"""

    @abstractmethod
    def schedule(
        self, 
        requests: list[Request],
        workers: list[Worker], 
    ) -> dict[Worker, deque[Request]]:
        """Map requests to workers"""
        ...


class FCFSPolicy(Policy):
    """
    Scheduling with FCFS.

    This is as simple as it gets. Requests are always sent to existing
    available workers; and when no worker is available, pick a worker
    by LRU for a swap.
    """

    def __str__(self):
        return f'FCFS'

    def schedule(
        self, 
        requests: list[Request],
        workers: list[Worker], 
    ) -> dict[Worker, deque[Request]]:

        schedule: dict[Worker, deque[Request]] = {}
        old_reqs = set()
        for worker in workers:
            schedule[worker] = deque(worker.queue)
            old_reqs.update(worker.queue)

        worker_by_model = {}
        for worker in workers:
            model = None if len(worker.queue) == 0 else worker.queue[-1]
            if model:
                worker_by_model[model] = worker
        
        for req in requests:
            if req in old_reqs:
                continue

            if worker := worker_by_model.get(req.model, None):
                schedule[worker].append(req)
            else:
                swap_worker = workers[0]
                swap_lru = workers[0].time_last_batch_ms
                for worker in workers:
                    if worker.time_last_batch_ms < swap_lru:
                        swap_worker = worker
                        swap_lru = worker.time_last_batch_ms
                schedule[swap_worker].append(req)
        
        return schedule

class V02Policy(Policy):
    """
    Scheduling policy for the v0.2 deployment.
    
    Parameter:
        - R = 3 (threshold for scale-up)
        - QS = 5 (maximum queue size for a single worker)
    
    For each request, scan all workers that have the matching model, and 
    pick the one with the biggest batch size (not exceeding B).

    If there are less than R available slots in total, schedule a scale-up.
    """
    R: int
    QS: int

    def __init__(
        self,
        R: int = 1,
        QS: int = 4,
    ):
        self.R = R
        self.QS = QS

    def __str__(self):
        return f'V02(QS={self.QS},R={self.R})'

    def schedule(
        self, 
        requests: list[Request],
        workers: list[Worker], 
    ) -> dict[Worker, deque[Request]]:
        """
        This policy is basically FCFS with some scale-up logic, without any
        eviction, reordering or rebalancing.

        To implement, simply assign workers to requests that have not been
        associated with a worker.
        """
        
        schedule: dict[Worker, deque[Request]] = {}
        old_reqs = set()
        for worker in workers:
            schedule[worker] = deque(worker.queue)
            old_reqs.update(worker.queue)

        states: dict[Worker, Tuple[ModelType, int]] = {}
        for worker in workers:
            model = worker.model if len(worker.queue) == 0 else worker.queue[-1].model
            count = 0
            for r in reversed(worker.queue):
                if r.model == model:
                    count += 1
                else:
                    break
            states[worker] = (model, count)
        
        for req in requests:
            if req in old_reqs:
                continue

            # Count slots
            slots = sum(max(0, self.QS-count) for model, count in states.values() if model == req.model)
            if slots < self.R:
                # Find a swap worker by LRU
                swap_worker = None
                swap_lru = 1000000000
                for worker, (model, _) in states.items():
                    if model == None:
                        swap_worker = worker
                        break
                    if model == req.model: continue
                    if worker.time_last_batch_ms < swap_lru:
                        swap_worker = worker
                        swap_lru = worker.time_last_batch_ms
                if swap_worker:
                    schedule[swap_worker].append(req)
                    continue
            
            # No swapping; send to an existing worker
            target_worker = None
            # target_slots = (self.QS, 1000000000)
            target_slots = 1000000000
            for worker, (model, count) in states.items():
                if model != req.model: continue
                # slots = (max(0, self.QS-count), count)
                slots = count
                if slots < target_slots:
                    target_worker = worker
                    target_slots = slots
            assert target_worker
            schedule[target_worker].append(req)
        
        return schedule
    
class SeLLMPolicy(V02Policy):
    """
    Scheduling policy for ServerlessLLM.
    """
    def __str__(self):
        return f'SeLLM(autoscale)'

class SeLLMPlusPolicy(V02Policy):
    """
    Scheduling policy for ServerlessLLM+.
    """
    def __str__(self):
        return f'SeLLM+(srtf)'

    def schedule(
        self, 
        requests: list[Request],
        workers: list[Worker], 
    ) -> dict[Worker, deque[Request]]:
        
        """
        This policy follows an assign-and-reorder approach. 
        It picks out new requests, assign them according to model affinity, then
        reorders the scheduling on each worker in an SRTF way.
        It also favors prefill jobs by counting their time as only the prefill time.
        """

        batched_schedule: dict[Worker, list[list[Request]]] = {}
        old_reqs = set()
        for worker in workers:
            sched: list[list[Request]] = []
            for req in worker.queue:
                for batch in sched:
                    if batch and batch[0].model == req.model and len(batch) < self.QS:
                        batch.append(req)
                        break
                else:
                    sched.append([req])
            batched_schedule[worker] = sched
            old_reqs.update(worker.queue)
        
        # Assign
        for req in requests:
            if req in old_reqs:
                continue

            min_worker = None
            min_load = 0
            for worker in workers:
                for batch in batched_schedule[worker]:
                    if batch and batch[0].model == req.model and len(batch) < self.QS:
                        batch.append(req)
                        break
                else:
                    if min_worker is None or min_load > len(batched_schedule[worker]):
                        min_worker = worker
                        min_load = len(batched_schedule[worker])
                    continue
                break
            else:
                assert min_worker is not None
                batched_schedule[min_worker].append([req])
        
        # Reorder
        for sched in batched_schedule.values():
            times = {}
            for batch in sched:
                # batch.sort(key=lambda req: req.output_len-req.progress)
                prefill_est = make_estimator(PrefillEstimator, model=batch[0].model, device=workers[0].device)
                decode_est = make_estimator(DecodeEstimator, model=batch[0].model, device=workers[0].device)

                prefill_reqs = [req for req in batch if req.is_prefill]
                if prefill_reqs:
                    time = prefill_est.predict(-1, -1, [req.input_len for req in prefill_reqs])
                else:
                    time = decode_est.predict(int((batch[0].current_context_len + batch[0].input_len + batch[0].output_len) / 2), 1) \
                        * (batch[0].input_len + batch[0].output_len - batch[0].current_context_len)
                times[batch[0]] = time
            sched.sort(key=lambda batch: times[batch[0]])

        schedule = {worker: deque(sum(sched, [])) for worker, sched in batched_schedule.items()}
        return schedule

class MixServePolicy(Policy):
    """
    Prototype implementation of scheduling with MixServe.

    Parameter: None
    
    TODO: description
    """

    prefill_policy: str
    decode_policy: str
    num_prefill_workers: int

    def __init__(
        self,
        prefill_policy: str = 'fcfs',
        decode_policy: str = 'avgload',
        num_prefill_workers: int = 1,
    ):
        assert prefill_policy in ('fcfs', 'sjf',), f'prefill policy {prefill_policy} not supported'
        assert decode_policy in ('avgload',), f'decode policy {decode_policy} not supported'
        assert num_prefill_workers > 0, 'at least one PrefillWorker is needed'
        
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy
        self.num_prefill_workers = num_prefill_workers
        
    
    def __str__(self):
        return f'MixLLM({self.num_prefill_workers}{self.prefill_policy}|{self.decode_policy})'
        
    def schedule(
        self, 
        requests: list[Request],
        workers: list[PrefillWorker | DecodeWorker],
    ) -> dict[Worker, deque[Request]]:
        assert all(type(worker) in (PrefillWorker, DecodeWorker) for worker in workers), 'MixServe only works in unison with PrefillWorkers and DecodeWorkers'
        schedule: dict[Worker, deque[Request]] = {worker: deque() for worker in workers}

        prefill_workers: list[PrefillWorker] = list(filter(lambda worker: type(worker) == PrefillWorker, workers))
        decode_workers: list[DecodeWorker] = list(filter(lambda worker: type(worker) == DecodeWorker, workers))
        assert prefill_workers and decode_workers, 'MixServe requires at least one worker for prefill and decode each'

        prefill_reqs: list[Request] = []
        new_decode_reqs: list[Request] = []
        for req in requests:
            if req.is_prefill:
                prefill_reqs.append(req)
            else:
                if all(req not in worker.requests for worker in decode_workers):
                    new_decode_reqs.append(req)

        # Schedule prefills
        if self.prefill_policy == 'fcfs':
            schedule |= self._schedule_prefill_fcfs(prefill_reqs, prefill_workers)
        elif self.prefill_policy == 'sjf':
            schedule |= self._schedule_prefill_sjf(prefill_reqs, prefill_workers)
        else:
            raise NotImplementedError
        
        # Schedule decodes
        if self.decode_policy == 'avgload':
            schedule |= self._schedule_decode_avgload(new_decode_reqs, decode_workers)
        else:
            raise NotImplementedError

        # print('======= schedule =======')
        # for worker, reqs in schedule.items():
        #     print(f'{worker.worker_id}: {[req.req_id for req in reqs]}')
        return schedule
    
    def _schedule_prefill_fcfs(
        self,
        prefill_reqs: list[Request],
        prefill_workers: list[PrefillWorker],
    ) -> dict[Worker, deque[list[Request]]]:
        
        # Keep the old schedule so far
        schedule: dict[PrefillWorker, deque[list[Request]]] = {worker: worker.queue for worker in prefill_workers}

        prefill_reqs.sort(key=lambda req: req.time_arrival_ms)
        if prefill_reqs and prefill_reqs[-1].time_arrival_ms == int(prefill_workers[0].env.now):
            # This is THE new prefill request
            req = prefill_reqs[-1]
            if any(req in itertools.chain(*worker.queue) for worker in prefill_workers):
                return schedule

            min_worker = None
            min_load = 10000000
            for worker, batches in schedule.items():
                for batch in worker.queue:
                    if batch[0].model == req.model and worker._batch_fits(batch, req):
                        # favor workers that can accept the request with existing batches..
                        batch.append(req)
                        break # (found)
                else:
                    # ..no such worker found; pick the least loaded one
                    load = len(batches)
                    if load < min_load:
                        min_load = load
                        min_worker = worker
                    continue
                break # (found)
            else:
                # (not found)
                schedule[min_worker].append([req])
                
        return schedule

    def _schedule_prefill_sjf(
        self,
        prefill_reqs: list[Request],
        prefill_workers: list[PrefillWorker],
    ) -> dict[Worker, deque[list[Request]]]:
        
        # Keep the old schedule so far
        schedule: dict[PrefillWorker, deque[list[Request]]] = {worker: worker.queue for worker in prefill_workers}
        
        def exec_time(req: Request):
            est = make_estimator(PrefillEstimator, req.model, prefill_workers[0].device)
            return est.predict(req.input_len, 1)

        prefill_reqs.sort(key=lambda req: req.time_arrival_ms)
        if prefill_reqs and prefill_reqs[-1].time_arrival_ms == int(prefill_workers[0].env.now):
            # This is THE new prefill request
            req = prefill_reqs[-1]
            if any(req in itertools.chain(*worker.queue) for worker in prefill_workers):
                return schedule

            min_worker = None
            min_load = 10000000
            for worker, batches in schedule.items():
                for batch in worker.queue:
                    if batch[0].model == req.model and worker._batch_fits(batch, req):
                        # favor workers that can accept the request with existing batches..
                        batch.append(req)
                        break # (found)
                else:
                    # ..no such worker found; pick the least loaded one
                    load = len(batches)
                    if load < min_load:
                        min_load = load
                        min_worker = worker
                    continue
                break # (found)
            else:
                # (not found)
                schedule[min_worker].append([req])
                
        return schedule
    
    def _schedule_decode_avgload(
        self,
        new_decode_reqs: list[Request],
        decode_workers: list[DecodeWorker],
    ) -> dict[Worker, deque[Request]]:
        schedule: dict[DecodeWorker, deque[Request]] = {worker: deque() for worker in decode_workers}

        decode_worker_batches: dict[DecodeWorker, list[list[Request]]] = {
            worker: [[req for req in batch] for _, batch in worker.decode_batches]
            for worker in decode_workers
        }

        for req in new_decode_reqs:
            min_worker = None
            min_load = 10000000
            for worker, batches in decode_worker_batches.items():
                for batch in batches:
                    if batch[0].model == req.model and worker._batch_fits(batch, req):
                        batch.append(req)
                        schedule[worker].append(req)
                        break
                else:
                    load = len(batches)
                    if load < min_load:
                        min_load = load
                        min_worker = worker
                    continue
                break
            else:
                decode_worker_batches[min_worker].append([req])
                schedule[min_worker].append(req)

        return schedule
    
    def _try_migrate_decode(
        self,
        src_decode_worker: DecodeWorker,
        decode_workers: list[DecodeWorker],
    ):
        schedule: dict[DecodeWorker, deque[Request]] = {worker: deque() for worker in decode_workers}
        decode_reqs = src_decode_worker.decode_batches[-1][1]

        decode_worker_batches: dict[DecodeWorker, list[list[Request]]] = {
            worker: [[req for req in batch] for _, batch in worker.decode_batches]
            for worker in decode_workers
        }

        # Migrate, if the batch could be distributed onto other workers
        # without adding a new batch
        for req in decode_reqs:
            for worker, batches in decode_worker_batches.items():
                for batch in batches:
                    if batch[0].model == req.model and worker._batch_fits(batch, req):
                        schedule[worker].append(req)
                        batch.append(req)
                        break
                else:
                    continue
                break
            else:
                return None
        
        src_decode_worker.decode_batches.pop()
        for worker, reqs in schedule.items():
            worker.queue.extend(reqs)
            worker._accept(to_batch=True)
    
