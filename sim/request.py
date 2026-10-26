from . import ModelType
from .estimator import *
from typing_extensions import Optional
from aenum import Enum
import simpy

class RequestState(Enum):
    FINISHED = 0
    PENDING = 1
    EXECUTING = 2
    FAILED = -1

class Request:
    """LLM requests for simulation."""

    def __init__(
        self,
        env: 'simpy.Environment' = None,
        req_id: int = None,
        model: ModelType = None,
        input_len: int = 512,
        output_len: int = 128,
    ):
        assert req_id is not None, "Request ID must be set"
        assert model is not None, "Request model must be specified"
        self.env = env
        self.req_id = req_id

        self.model = model
        self.input_len = input_len
        self.output_len = output_len
        self.metrics = {
            "time_arrival_ms": None,
            "time_first_sched_ms": None,
            "time_each_token_ms": [],
        }
        # progress: int in [0, output_len]
        self.progress: int = 0
        self.kvcache: bool = False
        self.state = RequestState.PENDING
    
    def __str__(self):
        return (
            f'Request('
            f'id={self.req_id},'
            f'model={self.model},'
            f'io=({self.input_len},{self.output_len}),'
            f'progress={self.progress},'
            f'kvcache={self.kvcache},'
            f'state={self.state},'
            f')'
        )
    
    __repr__ = __str__

    def __getstate__(self):
        return {
            'req_id': self.req_id,
            'model': str(self.model),
            'input_len': self.input_len,
            'output_len': self.output_len,
            'progress': self.progress,
            'kvcache': self.kvcache,
            'state': self.state,
            'metrics': self.metrics,
        }
    
    def __setstate__(self, state):
        self.env = None
        self.req_id = state['req_id']
        self.model = ModelType.from_str(state['model'][10:])
        self.input_len = state['input_len']
        self.output_len = state['output_len']
        self.progress = state['progress']
        self.kvcache = state['kvcache']
        self.state = state['state']
        self.metrics = state['metrics']

    def arrive(self):
        if self.metrics.get("time_arrival_ms", None) is None:
            self.metrics["time_arrival_ms"] = int(self.env.now)

    def sched(self):
        self.state = RequestState.EXECUTING
        if self.metrics["time_first_sched_ms"] is None:
            self.metrics["time_first_sched_ms"] = int(self.env.now)

    def fail(self):
        self.state = RequestState.FAILED
    
    def evict(self):
        # TODO: only evict by RECOMPUTATION is emulated now
        self.kvcache = False
        self.state = RequestState.PENDING

    def step(self):
        if self.state == RequestState.FAILED: return
        assert self.progress < self.output_len, f'{self}'

        self.progress += 1
        self.kvcache = True
        if self.progress == self.output_len:
            self.state = RequestState.FINISHED

        self.metrics["time_each_token_ms"].append(int(self.env.now))

    @property
    def current_context_len(self):
        return self.input_len + self.progress
    
    @property
    def is_finished(self):
        if finished := (self.state == RequestState.FINISHED):
            assert self.progress == self.output_len
        return finished
    
    @property
    def is_failed(self):
        return self.state == RequestState.FAILED

    @property
    def is_prefill(self):
        return not self.is_failed and not self.is_failed and self.kvcache == False
    
    @property
    def is_decode(self):
        return not self.is_failed and not self.is_failed and self.kvcache == True
    
    @property
    def time_arrival_ms(self) -> Optional[int]:
        return self.metrics.get("time_arrival_ms", None)

    @property
    def time_first_sched_ms(self) -> Optional[int]:
        return self.metrics.get("time_first_sched_ms", None)
    
    @property
    def time_first_token_ms(self) -> Optional[int]:
        if not self.metrics["time_each_token_ms"]:
            return None
        return self.metrics["time_each_token_ms"][0]
    
    @property
    def time_last_token_ms(self) -> Optional[int]:
        if not self.is_finished:
            return None
        return self.metrics["time_each_token_ms"][-1]
    
    @property
    def time_each_token_ms(self) -> list[int]:
        return self.metrics["time_each_token_ms"]
    
    @property
    def time_queue_ms(self) -> Optional[int]:
        if (time_first_sched_ms := self.time_first_sched_ms) is None:
            return None
        return time_first_sched_ms - self.time_arrival_ms
    
    @property
    def time_prefill_ms(self) -> Optional[int]:
        if (time_first_token_ms := self.time_first_token_ms) is None:
            return None
        return time_first_token_ms - self.time_first_sched_ms
    
    def get_time_each_token_percentile(self) -> dict[int, int]:
        if not self.is_finished:
            return dict()
        
        time_each_token_list = self.metrics["time_each_token_ms"]
        ts = [end - start for start, end in zip(time_each_token_list[:-1], time_each_token_list[1:])]
        ts.sort()

        if not ts:
            return dict()

        percentiles = [25, 50, 75, 90]
        return {p: ts[int(len(ts) * p / 100)] for p in percentiles}