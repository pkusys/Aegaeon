import asyncio
from collections import defaultdict
from typing import List, Dict, Optional, TYPE_CHECKING

from aegaeon.model_registry import ModelSpec
from aegaeon.logger import init_logger
from aegaeon.request import Request

if TYPE_CHECKING:
    from aegaeon.simple_scheduler import SimpleScheduler

logger = init_logger(__name__)

SLEEP_WHEN_DISPATCH_PENDS = 0.003
SLEEP_IN_EACH_EVENT_LOOP = 0
PRINT_STATUS_INTERVAL = 1


class SimpleDispatcher:
    """Dispatcher for simple mode.

    Routes incoming requests to SimpleEngine schedulers.
    Uses a single per-model FIFO queue and assigns requests
    to the least-loaded scheduler currently serving the same model.
    """

    def __init__(self):
        self.node_id: str = None
        self.model_queues: Dict[ModelSpec, List[Request]] = defaultdict(list)

    def __repr__(self):
        return f"{self.node_id}|simple dispatcher"

    async def initialize(self):
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.schedulers: List["SimpleScheduler"] = [
            engine.scheduler for engine in Controller.simple_engines().values()
        ]

    def add(self, request: Request):
        self.model_queues[request.model].append(request)

    async def dispatch(self):
        dispatched = False
        for queue in self.model_queues.values():
            if len(queue) == 0:
                continue
            request = queue[0]
            scheduler = self._pick_scheduler(request)
            if scheduler is not None:
                queue.pop(0)
                scheduler.add_request(request)
                dispatched = True
                logger.info(
                    f"({self}) request {request.request_id} -> ({scheduler.engine})"
                )

        if not dispatched:
            await asyncio.sleep(SLEEP_WHEN_DISPATCH_PENDS)

    def _pick_scheduler(self, request: Request) -> Optional["SimpleScheduler"]:
        pick_scheduler = None
        pick_load = float("inf")

        for scheduler in self.schedulers:
            waiting_load = sum(len(reqs) for reqs in scheduler.waiting_queue.values())
            active_load = sum(len(reqs) for reqs in scheduler.active_requests.values())
            total_load = waiting_load + active_load

            # Prefer a scheduler already serving this model
            if (
                scheduler.engine.model_config is not None
                and scheduler.engine.model_config.model == request.model
            ):
                return scheduler

            if total_load < pick_load:
                pick_scheduler = scheduler
                pick_load = total_load

        return pick_scheduler

    async def print_status(self):
        if any(len(queue) > 0 for queue in self.model_queues.values()):
            logger.info(
                f"({self.node_id}|simple dispatcher) waiting: "
                f"{[req.request_id for reqs in self.model_queues.values() for req in reqs]}"
            )

    async def start_event_loop(self):
        async def dispatch_loop():
            while True:
                await self.dispatch()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        async def report_loop():
            while True:
                await self.print_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(dispatch_loop(), report_loop())
