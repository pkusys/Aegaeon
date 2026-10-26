"""Capacity checks made after the worker registers the selected model's cache."""

from aegaeon.block_manager import BLOCK_LOCATION_CPU
from aegaeon.request import BatchedRequests


def request_capacity_error(engine, request, stage):
    """Return an actionable error for a request that cannot execute alone."""
    needed = engine.block_manager.get_num_blocks_needed_for_request(
        request, engine.engine_id
    )
    capacity = engine.block_manager.get_gpu_num_max_blocks_from_model(
        engine.engine_id, request.model
    )
    if needed > capacity:
        return (
            f"Request {request.request_id} cannot fit in {stage} on engine "
            f"#{engine.engine_id} for model {request.model}: "
            f"input_tokens={request.get_input_len()}, "
            f"output_tokens={request.get_output_len()}, "
            f"required_blocks={needed}, capacity_blocks={capacity}"
        )
    return None


async def fit_decode_batch(scheduler, batch):
    """Keep a runnable head, delay competing work, and terminate impossible work.

    Mutate the yielded batch in place: the scheduler generator holds this same
    object until the turn ends. Replacing it would leave stale request owners.
    Lookahead is a packing heuristic, not a requirement for executing one step.
    """
    engine = scheduler.engine
    capacity = engine.block_manager.get_gpu_num_max_blocks_from_model(
        engine.engine_id, batch.model
    )
    runnable, delayed = [], []
    used = 0
    for request in batch.requests:
        error = request_capacity_error(engine, request, "decode")
        if error is not None:
            await engine.fail_requests([request], error)
            continue
        needed = engine.block_manager.get_num_blocks_needed_for_request(
            request, engine.engine_id
        )
        if delayed or used + needed > capacity:
            delayed.append(request)
        else:
            runnable.append(request)
            used += needed

    # Remove failed requests before any await can expose them to dispatch.
    batch.requests = runnable
    if delayed:
        try:
            await engine.block_manager.move_requests(delayed, BLOCK_LOCATION_CPU)
        except ResourceWarning as exc:
            await engine.fail_requests(delayed, f"Decode swap-out allocation failed: {exc}")
        else:
            scheduler.decode_batches.insert(1, (0, BatchedRequests(delayed)))
            scheduler._assign_quota()

