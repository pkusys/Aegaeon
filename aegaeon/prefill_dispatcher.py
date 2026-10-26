import asyncio
from collections import defaultdict
from typing import List, Dict, Literal, Optional, TYPE_CHECKING

from aegaeon.model_registry import ModelSpec
from aegaeon.logger import init_logger
from aegaeon.request import Request

if TYPE_CHECKING:
    from aegaeon.prefill_scheduler import PrefillScheduler

logger = init_logger(__name__)

# Sleep for this many seconds when there is no request to dispatch,
# or when no scheduler can be dispatched to.
SLEEP_WHEN_DISPATCH_PENDS = 0.003

# Sleep for this many seconds in each event loop, useful for debugging.
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print dispatcher status every this many seconds
PRINT_STATUS_INTERVAL = 1


class PrefillDispatcher:
    """
    Dispatcher of prefill requests to individual schedulers in prefill engines.

    ## Event loops
    - Dispatch
        - Pop a request for dispatching
        - Find a prefill scheduler to dispatch to
    """

    def __init__(
        self,
        policy: Literal["fcfs-avgload", "sjf-avgload"] = "fcfs-avgload",
    ):
        self.policy = policy
        self.node_id: str = None

        # Mapping: model_type -> [request*]
        self.model_queues: Dict[ModelSpec, List[Request]] = defaultdict(list)

        match policy:
            case "fcfs-avgload":
                self._add = self._add_fcfs
                self._pop = self._pop_fcfs
                self._dispatch = self._dispatch_avgload
            case "sjf-avgload":
                self._add = self._add_sjf
                self._pop = self._pop_sjf
                self._dispatch = self._dispatch_avgload
            case _:
                raise ValueError(f"Unsupported prefill dispatcher policy {policy}")

    def __repr__(self):
        return f"{self.node_id}|prefill dispatcher"

    async def initialize(self):
        """
        Register prefill schedulers as dispatch targets.
        """
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.schedulers = [
            engine.scheduler for engine in Controller.prefill_engines().values()
        ]

    def add(self, request: Request):
        """
        Add request to the dispatcher queues, waiting to be dispatched.
        """
        self._add(request)

    def _add_fcfs(self, request: Request):
        self.model_queues[request.model].append(request)

    def _add_sjf(self, request: Request):
        raise NotImplementedError()

    def _pop_fcfs(self) -> Optional[List[Request]]:
        pick_queue = None
        for queue in self.model_queues.values():
            if len(queue) == 0:
                continue
            if pick_queue is None or queue[0].arrival_time < pick_queue[0].arrival_time:
                pick_queue = queue

        return pick_queue

    def _pop_sjf(self) -> Optional[List[Request]]:
        raise NotImplementedError()

    async def dispatch(self):
        """
        Dispatch one request in the dispatcher queue to a prefill scheduler.

        If there is no request to dispatch or all prefill schedulers are full, pends.
        """
        queue = self._pop()
        if queue is None or len(queue) == 0:
            await asyncio.sleep(SLEEP_WHEN_DISPATCH_PENDS)
        else:
            request = queue[0]
            scheduler = self._dispatch(request)
            if scheduler is None:
                await asyncio.sleep(SLEEP_WHEN_DISPATCH_PENDS)
            else:
                queue.pop(0)
                logger.info(
                    f"({self}) request {request.request_id} -> ({scheduler.engine})"
                )
                scheduler.add_request(request)

    def _dispatch_avgload(self, request: Request) -> Optional["PrefillScheduler"]:
        pick_scheduler = None
        pick_load = 0

        for scheduler in self.schedulers:
            for group in scheduler.waiting_queue:
                if (
                    group[0].model == request.model
                    and len(group) < scheduler.max_group_size
                ):
                    return scheduler

            if pick_scheduler is None:
                pick_scheduler = scheduler
                pick_load = sum(len(group) for group in scheduler.waiting_queue)
            else:
                load = sum(len(group) for group in scheduler.waiting_queue)
                if load < pick_load:
                    pick_scheduler = scheduler
                    pick_load = load

        return pick_scheduler

    async def print_status(self):
        if any(len(queue) > 0 for queue in self.model_queues.values()):
            logger.info(
                f"({self.node_id}|prefill dispatcher) waiting: {[req.request_id for reqs in self.model_queues.values() for req in reqs]})"
            )
        await asyncio.sleep(PRINT_STATUS_INTERVAL)

    async def start_event_loop(self):
        async def dispatch_loop():
            while True:
                await self.dispatch()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        async def report_loop():
            while True:
                await self.print_status()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await asyncio.gather(dispatch_loop(), report_loop())
