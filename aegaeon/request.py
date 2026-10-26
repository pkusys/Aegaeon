"""Sampling parameters for text generation."""

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, TypeAlias, Tuple

from aegaeon.model_registry import ModelSpec

# (req_id, input_ids, seq_len, context_len, multimodal engine input)
RequestMeta: TypeAlias = Tuple[int, List[int], int, int, Optional[Dict[str, Any]]]


class RequestCapacityError(RuntimeError):
    """A request cannot be served within the available KV-cache capacity."""


class Request:
    """A request contains the user's prompt, generated tokens and related information.
    Args:
        arrival_time: the absolute or relative time when the request arrives.
        request_id: the unique identifier for the request.
        prompt_token_ids: the token ids of the prompt.
    """

    def __init__(
        self,
        model: ModelSpec,
        arrival_time: float,
        request_id: int,
        prompt_token_ids: List[int],
        decode_tokens: int,
        prompt: Optional[str] = None,
        ignore_eos: bool = False,
        eos_token_ids: Optional[Iterable[int]] = None,
        stop_token_ids: Optional[Iterable[int]] = None,
        min_tokens: int = 0,
        max_model_len: Optional[int] = None,
        multimodal_input: Optional[Dict[str, Any]] = None,
    ):
        # static states
        self.model = model
        self.arrival_time = arrival_time
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.prompt = prompt
        self.decode_tokens = decode_tokens
        self.ignore_eos = ignore_eos
        self.eos_token_ids = frozenset(int(value) for value in eos_token_ids or ())
        self.stop_token_ids = frozenset(int(value) for value in stop_token_ids or ())
        self.min_tokens = min_tokens
        self.max_model_len = max_model_len
        self.multimodal_input = multimodal_input

        # dynamic states
        self.generated_token_ids = []
        self.is_finished = False
        self.is_running = False
        self.failure: Optional[RequestCapacityError] = None
        self.eos_token_id: Optional[int] = None

    def __lt__(self, other):
        if isinstance(other, Request):
            return self.arrival_time < other.arrival_time
        raise NotImplementedError(f"cannot compare Request with {type(other)}")

    def fail(self, reason: str | RequestCapacityError) -> None:
        """Finish the request with an explicit, serializable capacity error."""
        if self.failure is None:
            self.failure = RequestCapacityError(
                f"Request {self.request_id} ({self.model.name}): {reason}"
            )
        self.is_finished = True
        self.is_running = False

    def _check_finish_condition(self):
        if self.get_output_len() >= self.decode_tokens:
            self.is_finished = True
        elif (
            self.max_model_len is not None
            and self.get_total_len() >= self.max_model_len
        ):
            self.is_finished = True
        # The candidate stop token itself does not count toward min_tokens.
        # This matches the API decoder, which checks the number of already-kept
        # output tokens before accepting a stop token.
        elif self.get_output_len() > self.min_tokens and self.generated_token_ids:
            token_id = self.generated_token_ids[-1]
            if token_id in self.stop_token_ids:
                self.is_finished = True
            elif not self.ignore_eos:
                if token_id in self.eos_token_ids or token_id == self.eos_token_id:
                    self.is_finished = True

    def add_generated_token(self, token_id: int):
        if self.get_output_len() > self.decode_tokens:
            raise ValueError(
                f"The generated tokens exceed the maximum output length {self.decode_tokens}\
                for request {self.request_id}."
            )
        self.generated_token_ids.append(token_id)
        self._check_finish_condition()

    def is_prefill_stage(self) -> bool:
        return len(self.generated_token_ids) == 0

    def get_input_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.generated_token_ids)

    def get_total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def get_input_tokens_ids(self) -> List[int]:
        """The token ids of the input tokens for the next iteration.
        For request in the prefill stage, this is equal to the prompt_token_ids.
        For request in the decode stage, this is equal to the newly generated token.
        """
        if self.is_prefill_stage():
            return self.prompt_token_ids
        else:
            return [self.generated_token_ids[-1]]

    def get_num_input_tokens(self) -> int:
        return len(self.get_input_tokens_ids())

    def get_first_new_token_index(self) -> int:
        """The index of the first newly generated tokens.
        In the decode phase, only the input tokens need to compute QKV. The index
        of the first token in the input tokens of next round is needed to do positional
        embedding and decode phase kernel correctly.
        """
        return (
            0
            if self.is_prefill_stage()
            else self.get_input_len() + self.get_output_len() - 1
        )

    def meta(self) -> RequestMeta:
        return (
            self.request_id,
            self.get_input_tokens_ids(),
            self.get_total_len(),
            self.get_first_new_token_index(),
            # A different decode engine may need this to reconstruct M-RoPE
            # state after prefill/decode disaggregation.
            self.multimodal_input,
        )

    def __str__(self) -> str:
        return (
            f"Request(arrival_time = {self.arrival_time}, "
            f"request_id={self.request_id}, "
            f"input_len={self.get_input_len()}, "
            f"output_len={self.get_output_len()}, "
            f"decode_tokens={self.decode_tokens}, "
            f"is_prefill_stage={self.is_prefill_stage()}, "
            f"is_finished={self.is_finished})"
        )


class BatchedRequests:
    def __init__(
        self,
        requests: Optional[List[Request]] = None,
    ) -> None:
        self.model: Optional[ModelSpec]
        if not requests:
            self.requests = []
            self.model = None
        else:
            self.requests = requests
            self.model = requests[0].model
        self.start_time = None

    def __len__(self):
        return len(self.requests)

    def __str__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def __repr__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def add_request(self, request: Request):
        assert (
            self.model is None or self.model == request.model
        ), f"request {request.request_id} model {request.model} and batch model {self.model} mismatches"
        self.model = request.model
        self.requests.append(request)

    def pop_finished_requests(self) -> List[Request]:
        finished_requests, unfinished_requests = [], []
        for request in self.requests:
            if request.is_finished:
                finished_requests.append(request)
            else:
                unfinished_requests.append(request)
        self.requests = unfinished_requests
        return finished_requests

    def start_one_iteration(self, start_time):
        """Update the start time of the batch before its execution of iteration."""
        assert self.start_time is None, "the batch has already started one iteration"
        self.start_time = start_time
        self.is_running = True

    def finish_one_iteration(
        self,
        maybe_generated_tokens_ids: List[Optional[int]],
    ):
        """Update the requests in the batch after it finishes one iteration
        Note: the order of generated tokens should align with self.requests.
        """
        assert self.start_time is not None, "the batch has not been started"
        for request, maybe_token_id in zip(self.requests, maybe_generated_tokens_ids):
            if maybe_token_id is not None:
                request.add_generated_token(maybe_token_id)
        self.start_time = None
        self.is_running = False

    #### General Getters
    def get_request_ids(self) -> List[int]:
        return [request.request_id for request in self.requests]

    def get_num_input_tokens(self) -> int:
        return sum([request.get_num_input_tokens() for request in self.requests])

    def get_num_total_tokens(self) -> int:
        return sum([request.get_total_len() for request in self.requests])

    #### Getters for the GPT operator parameters ####
    def get_input_tokens_batched(self) -> List[List[int]]:
        return [request.get_input_tokens_ids() for request in self.requests]

    def get_first_token_indexes(self) -> List[int]:
        return [request.get_first_new_token_index() for request in self.requests]

    def get_is_prefill_stage(self) -> List[bool]:
        return [request.is_prefill_stage() for request in self.requests]
