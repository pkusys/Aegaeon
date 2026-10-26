import argparse
import asyncio
import json
import os
import signal
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
import ray
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field

from aegaeon.config import (
    NodeConfig,
    get_model_config,
    register_model_config,
    unregister_model_config,
)
from aegaeon.llm import LLMService
from aegaeon.logger import init_logger
from aegaeon.model_placement import ModelNotReadyError
from aegaeon.model_registry import ModelSpec
from aegaeon.resources import web_directory
from aegaeon.request import Request as AegaeonRequest
from aegaeon.request import RequestCapacityError
from aegaeon.tool_calls import (
    ParsedToolCalls,
    normalize_tool_call_messages,
    parse_tool_calls_with_vllm,
    resolve_tool_parser,
)

logger = init_logger(__name__)


_STREAM_CLEANUP_TASKS: set[asyncio.Task] = set()


@dataclass
class ServerContext:
    """Runtime state shared with API handlers through dependency injection."""

    llm_service: LLMService
    cached_models: list[ModelSpec]
    deployed_models: dict[str, ModelSpec]
    config_path: str = ""
    tensor_parallel_size: int = 1


def get_context(request: FastAPIRequest) -> ServerContext:
    """FastAPI dependency: expose the ServerContext built during lifespan."""
    return request.app.state.context


class DecodeRateEMA:
    """Track the exponential moving average of received decode tokens/second."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value: float | None = None
        self.last_observed_at: float | None = None

    def observe(self, token_count: int = 1) -> float | None:
        if token_count <= 0:
            return self.value
        now = time.monotonic()
        previous_at = self.last_observed_at
        self.last_observed_at = now
        if previous_at is None:
            return self.value

        elapsed = now - previous_at
        if elapsed <= 0:
            return self.value
        instantaneous_rate = token_count / elapsed
        if self.value is None:
            self.value = instantaneous_rate
        else:
            self.value = (
                self.alpha * instantaneous_rate + (1.0 - self.alpha) * self.value
            )
        return self.value


class AccessRateTracker:
    """Track per-model request arrivals and generated tokens in a sliding window.

    The console page shows these as "Model Access Rate".Rates are simple window averages: (counts in the last ''window'' secs) / ''window''.

    Rate limiting: when ``max_requests_per_second`` / ``max_tokens_per_second``
    are set, ``check`` rejects requests whose sliding-window rate exceeds the
    configured per-model cap (HTTP 429).

    All access happens on the asyncio event loop, guarded by ''_lock''."""

    def __init__(
        self,
        window_seconds: int = 60,
        max_requests_per_second: float | None = None,
        max_tokens_per_second: float | None = None,
    ) -> None:
        self.window = window_seconds
        self.max_requests_per_second = max_requests_per_second
        self.max_tokens_per_second = max_tokens_per_second
        self._lock = asyncio.Lock()
        self._arrivals: dict[str, list[float]] = {}
        self._tokens: dict[str, list[tuple[float, int]]] = {}

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        self._arrivals = {
            model: [ts for ts in stamps if ts >= cutoff]
            for model, stamps in self._arrivals.items()
        }
        self._arrivals = {
            model: stamps for model, stamps in self._arrivals.items() if stamps
        }
        self._tokens = {
            model: [(ts, count) for ts, count in pairs if ts >= cutoff]
            for model, pairs in self._tokens.items()
        }
        self._tokens = {model: pairs for model, pairs in self._tokens.items() if pairs}

    async def record_request(self, model: str) -> None:
        now = time.monotonic()
        async with self._lock:
            self._arrivals.setdefault(model, []).append(now)
            self._prune(now)

    async def record_tokens(self, model: str, count: int) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        async with self._lock:
            self._tokens.setdefault(model, []).append((now, count))
            self._prune(now)

    async def check(self, model: str) -> str | None:
        """Return an error message if the model exceeds its rate limit, else None.

        The pending request itself is counted into the request rate, so a
        request that would push the window over ``max_requests_per_second`` is
        rejected. Token rate is computed from already-generated tokens only.
        """
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            request_rate = (len(self._arrivals.get(model, [])) + 1) / self.window
            token_rate = (
                sum(count for _, count in self._tokens.get(model, [])) / self.window
            )
            if (
                self.max_requests_per_second is not None
                and request_rate > self.max_requests_per_second
            ):
                return (
                    f"model {model!r} over request rate limit: "
                    f"{request_rate:.3f} > {self.max_requests_per_second} req/s"
                )
            if (
                self.max_tokens_per_second is not None
                and token_rate > self.max_tokens_per_second
            ):
                return (
                    f"model {model!r} over token rate limit: "
                    f"{token_rate:.3f} > {self.max_tokens_per_second} tok/s"
                )
            return None

    async def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            metrics = []
            for model in sorted(set(self._arrivals) | set(self._tokens)):
                request_count = len(self._arrivals.get(model, []))
                token_count = sum(count for _, count in self._tokens.get(model, []))
                metrics.append(
                    {
                        "model": model,
                        "requests_per_second": round(request_count / self.window, 3),
                        "tokens_per_second": round(token_count / self.window, 3),
                        "requests": request_count,
                        "tokens": token_count,
                    }
                )
            return metrics


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


_ACCESS_RATE = AccessRateTracker(
    max_requests_per_second=_env_float("AEGAEON_MAX_REQUESTS_PER_SECOND"),
    max_tokens_per_second=_env_float("AEGAEON_MAX_TOKENS_PER_SECOND"),
)


class APIModel(BaseModel):
    """Reject fields that this compatibility layer cannot implement."""

    model_config = ConfigDict(extra="forbid")


class FunctionDefinition(APIModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    strict: Optional[bool] = None


class FunctionTool(APIModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedFunctionToolChoice(APIModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class NamedToolChoice(APIModel):
    type: Literal["function"] = "function"
    function: NamedFunctionToolChoice


class FunctionCall(APIModel):
    name: str
    # OpenAI sends a JSON string. vLLM's chat templates also accept an already
    # decoded object for assistant history, so keep both wire forms readable.
    arguments: Union[str, Dict[str, Any], List[Any]]


class ToolCall(APIModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(APIModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class StreamOptions(APIModel):
    include_usage: bool = False


class DeployRequest(BaseModel):
    """Used for deploy and undeploy requests."""

    model: str
    model_path: str | None = None
    max_model_len: int | None = None
    node_ids: list[str] | None = None
    replica_count: int = Field(default=1, ge=1)
    tool_parser: str | None = None


class GreedyGenerationRequest(APIModel):
    """Parameters shared by chat and legacy completion requests.

    Sampling controls are present so vLLM/OpenAI clients can send explicit
    greedy settings. Values that would require stochastic sampling are
    rejected in ``_validate_greedy_parameters``.
    """

    model: str
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: int = 1

    # Greedy-equivalent sampling controls.
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    top_k: int | None = 1
    min_p: float | None = 0.0
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0
    repetition_penalty: float | None = 1.0
    seed: int | None = None
    logit_bias: dict[str, float] | None = None
    use_beam_search: bool = False
    length_penalty: float = 1.0

    # Stopping, tokenization, and vLLM extension fields supported here.
    stop_token_ids: list[int] | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: int | None = None
    truncation_side: Literal["left", "right"] | None = None
    add_special_tokens: bool = True
    return_token_ids: bool = False
    priority: int = 0
    request_id: str | None = None
    user: str | None = None


class ChatCompletionRequest(GreedyGenerationRequest):
    messages: list[Message]
    max_completion_tokens: int | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    add_special_tokens: bool = False
    tools: Optional[List[FunctionTool]] = None
    tool_choice: Optional[
        Union[Literal["none", "auto", "required"], NamedToolChoice]
    ] = None
    parallel_tool_calls: Optional[bool] = None

    # Log probabilities are unavailable because the worker only returns ids.
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    mm_processor_kwargs: dict[str, Any] | None = None


class CompletionRequest(GreedyGenerationRequest):
    prompt: str | list[int]
    echo: bool = False
    logprobs: int | None = None


class TokenizeRequest(APIModel):
    model: str
    prompt: str
    add_special_tokens: bool = True


class DetokenizeRequest(APIModel):
    model: str
    tokens: list[int]
    skip_special_tokens: bool = True


class GenerationResult:
    def __init__(
        self,
        text: str,
        token_ids: list[int],
        prompt_token_ids: list[int],
        finish_reason: str,
        internal_id: int,
    ) -> None:
        self.text = text
        self.token_ids = token_ids
        self.prompt_token_ids = prompt_token_ids
        self.finish_reason = finish_reason
        self.internal_id = internal_id

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)

    def usage(self) -> dict[str, int]:
        prompt_tokens = len(self.prompt_token_ids)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": prompt_tokens + self.completion_tokens,
        }


class GenerationHandle:
    """A submitted generation's routing information."""

    def __init__(
        self,
        internal_id: int,
        node,
        request: AegaeonRequest,
        *,
        routing_reserved: bool = False,
        routing_service: LLMService | None = None,
    ) -> None:
        self.internal_id = internal_id
        self.node = node
        self.request = request
        self.routing_reserved = routing_reserved
        self.routing_service = routing_service

    def release_routing(self) -> None:
        """Release this handle's routing reservation at most once."""
        if not self.routing_reserved:
            return
        self.routing_reserved = False
        if self.routing_service is not None:
            self.routing_service.release_node_reservation(self.internal_id)


def _select_startup_model_names(
    cli_models: str | None,
    configured_models: tuple[str, ...] | None,
    available_models: list[str],
) -> list[str]:
    """Resolve startup models while preserving the legacy default-to-all behavior."""
    if cli_models is not None:
        selected = [name.strip() for name in cli_models.split(",") if name.strip()]
    elif configured_models is not None:
        selected = list(configured_models)
    else:
        selected = list(available_models)
    return list(dict.fromkeys(selected))


def create_service(args) -> ServerContext:
    """Build the Aegaeon cluster from startup arguments.

    Returns a :class:`ServerContext` that API handlers receive through the
    ``get_context`` dependency. Keeping this separate from ``lifespan`` makes
    the startup sequence reusable and testable.
    """
    config_path = ""
    if getattr(args, "model_config", None):
        from aegaeon.models import set_model_registry

        config_path = os.path.abspath(args.model_config)
        os.environ["AEGAEON_CONFIG_PATH"] = config_path

        # Keep internal config reading for backwards compatibility with the original script
        with open(config_path) as _f:
            os.environ["AEGAEON_CONFIG_CONTENT"] = _f.read()

        set_model_registry(config_path)
    else:
        from aegaeon.config import ServerConfig, set_server_config
        from aegaeon.model_registry import ModelSpecRegistry, set_registry

        set_registry(ModelSpecRegistry.default())
        set_server_config(ServerConfig())

        try:
            from aegaeon.device_registry import (
                DeviceRegistry,
            )
            from aegaeon.device_registry import (
                set_registry as set_device_reg,
            )

            set_device_reg(DeviceRegistry.default())
        except ImportError:
            pass

    from aegaeon.model_registry import get_registry

    registry = get_registry()

    try:
        from aegaeon.config import get_server_config

        startup_models = get_server_config().startup_models
    except RuntimeError:
        startup_models = None
    model_names = _select_startup_model_names(
        getattr(args, "models", None),
        startup_models,
        [spec.name for spec in registry],
    )

    # if not model_names:
    #    raise ValueError("System startup failed: No valid models specified")

    cached_models = []
    for name in model_names:
        spec = registry.get_by_name(name)
        if spec is None:
            raise ValueError(f"Model '{name}' not found in config")
        cached_models.append(spec)

    if not ray.is_initialized():
        ray_address = getattr(args, "ray_address", None) or os.environ.get(
            "AEGAEON_RAY_ADDRESS",
            os.environ.get("RAY_ADDRESS", "auto"),
        )
        if ray_address.lower() in {"", "local", "none"}:
            ray_address = None
        logger.info("Connecting Aegaeon to Ray cluster at %s", ray_address)
        ray_init_kwargs = {
            "address": ray_address,
            "ignore_reinit_error": True,
        }
        # Existing clusters do not inherit this process's environment in the
        # Controller/Worker actors; forward the arena-fraction knob explicitly.
        arena_fraction = os.environ.get("AEGAEON_ARENA_FRACTION")
        if arena_fraction:
            ray_init_kwargs["runtime_env"] = {
                "env_vars": {"AEGAEON_ARENA_FRACTION": arena_fraction}
            }
        if ray_address is None:
            ray_init_kwargs["resources"] = {"node_0": 1}
            ray_temp_dir = os.environ.get("AEGAEON_RAY_TEMP_DIR")
            if ray_temp_dir:
                ray_init_kwargs["_temp_dir"] = ray_temp_dir
        ray.init(**ray_init_kwargs)

    try:
        from aegaeon.config import get_server_config

        server_cfg = get_server_config()
        nnodes = getattr(server_cfg, "nnodes", 1)
        num_prefill_engines = getattr(
            server_cfg, "num_prefill_engines", args.num_prefill_engines
        )
        num_decode_engines = getattr(
            server_cfg, "num_decode_engines", args.num_decode_engines
        )
        num_engines = getattr(
            server_cfg, "num_engines", getattr(args, "num_engines", 0)
        )
        tensor_parallel_size = getattr(args, "tensor_parallel_size", None) or getattr(
            server_cfg, "tensor_parallel_size", 1
        )
        cpu_num_slabs = getattr(server_cfg, "cpu_num_slabs", args.cpu_num_slabs)
        cpu_slab_size_bytes = getattr(
            server_cfg, "cpu_slab_size_bytes", args.cpu_slab_size_gb * (1024**3)
        )
        model_cache_size = getattr(
            server_cfg, "model_cache_size_gb", args.model_cache_size_gb
        ) * (1024**3)
        decode_work_stealing = getattr(server_cfg, "decode_work_stealing", False)
        decode_work_stealing_min_batch_gap = getattr(
            server_cfg, "decode_work_stealing_min_batch_gap", 2
        )
        request_routing_policy_name = getattr(
            server_cfg, "request_routing_policy", "least-outstanding"
        )
        model_placement_policy_name = getattr(
            server_cfg, "model_placement_policy", "least-loaded"
        )
    except Exception:
        nnodes = 1
        num_prefill_engines = args.num_prefill_engines
        num_decode_engines = args.num_decode_engines
        num_engines = getattr(args, "num_engines", 0)
        tensor_parallel_size = getattr(args, "tensor_parallel_size", 1) or 1
        cpu_num_slabs = args.cpu_num_slabs
        cpu_slab_size_bytes = args.cpu_slab_size_gb * (1024**3)
        model_cache_size = args.model_cache_size_gb * (1024**3)
        decode_work_stealing = False
        decode_work_stealing_min_batch_gap = 2
        request_routing_policy_name = "least-outstanding"
        model_placement_policy_name = "least-loaded"

    if nnodes < 1:
        raise ValueError("server.nnodes must be at least 1")
    if tensor_parallel_size < 1:
        raise ValueError("server.tensor_parallel_size must be at least 1")
    if num_engines > 0:
        if num_prefill_engines != 0 or num_decode_engines != 0:
            raise ValueError(
                "server.num_engines (simple mode) is mutually exclusive with "
                "server.num_prefill_engines/server.num_decode_engines"
            )
    elif num_prefill_engines + num_decode_engines < 1:
        raise ValueError(
            "Configure at least one simple engine, or at least one prefill/decode engine"
        )

    ray_resources = ray.cluster_resources()
    missing_nodes = [
        f"node_{i}" for i in range(nnodes) if f"node_{i}" not in ray_resources
    ]
    if missing_nodes:
        raise ValueError(
            f"server.nnodes={nnodes}, but Ray does not advertise resources "
            f"{missing_nodes}. A machine with multiple GPUs is still one node; "
            "use nnodes: 1 and set num_engines to the GPU count."
        )

    cluster_config = [
        NodeConfig(
            node_id=f"node_{i}",
            num_prefill_engines=num_prefill_engines,
            num_decode_engines=num_decode_engines,
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            cpu_num_slabs=cpu_num_slabs,
            cpu_slab_size_bytes=cpu_slab_size_bytes,
            model_cache_size=model_cache_size,
            cached_models=cached_models,
            config_path=config_path,
            decode_work_stealing=decode_work_stealing,
            decode_work_stealing_min_batch_gap=decode_work_stealing_min_batch_gap,
        )
        for i in range(nnodes)
    ]
    # Ensure model_path is set on ModelSpec instances before they are
    # serialised to the Controller.  This avoids an extra download inside
    # Controller.init().
    for spec in cached_models:
        spec.path()

    from aegaeon.model_placement import (
        create_model_placement_policy,
        create_request_routing_policy,
    )

    model_placement_policy = create_model_placement_policy(
        model_placement_policy_name
    )
    request_routing_policy = create_request_routing_policy(
        request_routing_policy_name
    )

    return ServerContext(
        llm_service=LLMService(
            cluster_config,
            model_placement_policy=model_placement_policy,
            request_routing_policy=request_routing_policy,
        ),
        cached_models=cached_models,
        deployed_models={spec.name: spec for spec in cached_models},
        config_path=config_path,
        tensor_parallel_size=tensor_parallel_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ray's driver initialization replaces Uvicorn's SIGTERM handler.
    on_main_thread = threading.current_thread() is threading.main_thread()
    sigterm_handler = signal.getsignal(signal.SIGTERM) if on_main_thread else None
    try:
        try:
            app.state.context = create_service(app.state.engine_args)
        finally:
            if sigterm_handler is not None:
                signal.signal(signal.SIGTERM, sigterm_handler)
        yield
    finally:
        ray.shutdown()


app = FastAPI(title="Aegaeon OpenAI-Compatible API Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serverlessllm_url(path: str) -> str:
    """Resolve the private ServerlessLLM endpoint used by the browser proxy."""
    base = os.getenv("SERVERLESSLLM_URL", "http://127.0.0.1:8343").rstrip("/")
    return f"{base}{path}"


def _serverlessllm_proxy_error(exc: BaseException) -> JSONResponse:
    upstream = _serverlessllm_url("")
    return JSONResponse(
        status_code=502,
        content={
            "detail": {
                "message": (
                    f"Cannot reach ServerlessLLM at {upstream}: {exc}. "
                    "Set SERVERLESSLLM_URL to the address reachable from the "
                    "Aegaeon API process."
                ),
                "type": "serverlessllm_connection_error",
                "param": None,
                "code": "bad_gateway",
            }
        },
    )


@app.get("/v1/serverlessllm/models", include_in_schema=False)
async def proxy_serverlessllm_models():
    """Expose ServerlessLLM through Aegaeon's origin to avoid browser CORS."""
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                _serverlessllm_url("/v1/models"),
                headers={"Accept": "application/json"},
            ) as upstream:
                body = await upstream.read()
                return Response(
                    content=body,
                    status_code=upstream.status,
                    media_type=upstream.content_type or "application/json",
                )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return _serverlessllm_proxy_error(exc)


@app.post("/v1/serverlessllm/chat/completions", include_in_schema=False)
async def proxy_serverlessllm_chat_completions(request: FastAPIRequest):
    """Proxy complete and SSE chat responses without buffering the upstream."""
    body = await request.body()
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        payload = {}
    streaming = bool(payload.get("stream"))
    timeout = aiohttp.ClientTimeout(
        total=None if streaming else 900,
        connect=15,
        sock_connect=15,
        sock_read=None if streaming else 900,
    )
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        upstream = await session.post(
            _serverlessllm_url("/v1/chat/completions"),
            data=body,
            headers={
                "Content-Type": request.headers.get("content-type", "application/json"),
                "Accept": request.headers.get(
                    "accept",
                    "text/event-stream" if streaming else "application/json",
                ),
            },
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        await session.close()
        return _serverlessllm_proxy_error(exc)

    if not streaming or upstream.status >= 400:
        try:
            response_body = await upstream.read()
            return Response(
                content=response_body,
                status_code=upstream.status,
                media_type=upstream.content_type or "application/json",
            )
        finally:
            upstream.release()
            await session.close()

    async def iterator():
        decode_rate = DecodeRateEMA()
        buffer = b""
        try:
            async for chunk in upstream.content.iter_any():
                buffer += chunk
                while True:
                    boundaries = [
                        (position, separator)
                        for separator in (b"\r\n\r\n", b"\n\n")
                        if (position := buffer.find(separator)) >= 0
                    ]
                    if not boundaries:
                        break
                    boundary, separator = min(boundaries, key=lambda item: item[0])

                    block = buffer[:boundary]
                    buffer = buffer[boundary + len(separator) :]
                    lines = block.splitlines()
                    data_lines = [
                        line[5:].lstrip() for line in lines if line.startswith(b"data:")
                    ]
                    if not data_lines:
                        yield block + separator
                        continue

                    data = b"\n".join(data_lines)
                    if data == b"[DONE]":
                        yield block + separator
                        continue
                    try:
                        event = json.loads(data)
                        choice = (event.get("choices") or [{}])[0]
                        content = (
                            (choice.get("delta") or {}).get("content")
                            or choice.get("text")
                            or ""
                        )
                        if content:
                            rate = decode_rate.observe()
                            if rate is not None:
                                event["decode_tokens_per_second"] = rate
                        yield _sse(event).encode("utf-8")
                    except (AttributeError, TypeError, ValueError, IndexError):
                        yield block + separator
            if buffer:
                yield buffer
        finally:
            upstream.release()
            await session.close()

    return StreamingResponse(
        iterator(),
        status_code=upstream.status,
        media_type=upstream.content_type or "text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@lru_cache(maxsize=128)
def _get_tokenizer(model_path: str):
    from aegaeon.utils import get_tokenizer

    return get_tokenizer(model_path)


def _error(parameter: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": message,
            "type": "invalid_request_error",
            "param": parameter,
            "code": None,
        },
    )


def _resolve_model(ctx: ServerContext, model_name: str):
    spec = ctx.deployed_models.get(model_name)
    if spec is not None:
        return spec
    raise _error("model", f"Model '{model_name}' is not deployed.")


def _tokenizer_for(spec):
    try:
        return _get_tokenizer(spec.path())
    except Exception as exc:
        raise _error("model", f"Failed to load tokenizer for '{spec.name}': {exc}")


def _effective_max_tokens(request: GreedyGenerationRequest, default: int) -> int:
    max_tokens = request.max_tokens
    if isinstance(request, ChatCompletionRequest):
        max_completion_tokens = request.max_completion_tokens
        if max_tokens is not None and max_completion_tokens is not None:
            if max_tokens != max_completion_tokens:
                raise _error(
                    "max_completion_tokens",
                    "max_tokens and max_completion_tokens must match when both are set",
                )
        if max_completion_tokens is not None:
            max_tokens = max_completion_tokens

    max_tokens = default if max_tokens is None else max_tokens
    if max_tokens < 1:
        raise _error("max_tokens", "max_tokens must be at least 1")
    return max_tokens


def _validate_greedy_parameters(
    request: GreedyGenerationRequest, max_tokens: int
) -> None:
    checks = [
        ("n", request.n == 1, "Aegaeon currently supports only n=1"),
        (
            "temperature",
            request.temperature in (None, 0, 0.0),
            "the current Aegaeon worker is greedy; temperature must be 0",
        ),
        (
            "top_p",
            request.top_p in (None, 1, 1.0),
            "the current Aegaeon worker is greedy; top_p must be 1",
        ),
        (
            "top_k",
            request.top_k in (None, 1),
            "the current Aegaeon worker is greedy; top_k must be 1",
        ),
        (
            "min_p",
            request.min_p in (None, 0, 0.0),
            "min_p sampling is not supported; min_p must be 0",
        ),
        (
            "frequency_penalty",
            request.frequency_penalty in (None, 0, 0.0),
            "frequency_penalty is not supported",
        ),
        (
            "presence_penalty",
            request.presence_penalty in (None, 0, 0.0),
            "presence_penalty is not supported",
        ),
        (
            "repetition_penalty",
            request.repetition_penalty in (None, 1, 1.0),
            "repetition_penalty is not supported",
        ),
        ("logit_bias", not request.logit_bias, "logit_bias is not supported"),
        (
            "use_beam_search",
            not request.use_beam_search,
            "beam search is not supported",
        ),
        (
            "length_penalty",
            request.length_penalty == 1.0,
            "length_penalty is not supported",
        ),
        ("priority", request.priority == 0, "priority scheduling is not supported"),
        (
            "min_tokens",
            0 <= request.min_tokens <= max_tokens,
            "min_tokens must be between 0 and max_tokens",
        ),
        (
            "truncate_prompt_tokens",
            request.truncate_prompt_tokens is None
            or request.truncate_prompt_tokens == -1
            or request.truncate_prompt_tokens > 0,
            "truncate_prompt_tokens must be -1 or a positive integer",
        ),
        (
            "stream_options",
            request.stream or request.stream_options is None,
            "stream_options can only be used when stream=true",
        ),
        (
            "stop_token_ids",
            request.stop_token_ids is None
            or all(token_id >= 0 for token_id in request.stop_token_ids),
            "stop_token_ids must contain only non-negative integers",
        ),
    ]
    for parameter, valid, message in checks:
        if not valid:
            raise _error(parameter, message)

    if isinstance(request, ChatCompletionRequest):
        if not request.messages:
            raise _error("messages", "messages must not be empty")
        if request.continue_final_message and request.add_generation_prompt:
            raise _error(
                "continue_final_message",
                "continue_final_message and add_generation_prompt cannot both be true",
            )
        if request.logprobs:
            raise _error("logprobs", "logprobs are not available from the worker")
        if request.top_logprobs not in (None, 0):
            raise _error(
                "top_logprobs", "top_logprobs are not available from the worker"
            )
        if request.tools == []:
            raise _error("tools", "tools must not be an empty array")
        if request.tool_choice not in (None, "none") and not request.tools:
            raise _error("tool_choice", "tool_choice requires tools")
        if request.tools:
            tool_names = [tool.function.name for tool in request.tools]
            if len(set(tool_names)) != len(tool_names):
                raise _error("tools", "tool function names must be unique")
            if isinstance(request.tool_choice, NamedToolChoice):
                selected_name = request.tool_choice.function.name
                if selected_name not in tool_names:
                    raise _error(
                        "tool_choice",
                        f"tool_choice function {selected_name!r} is not present in tools",
                    )
    elif isinstance(request, CompletionRequest) and request.logprobs is not None:
        raise _error("logprobs", "logprobs are not available from the worker")


def _truncate_prompt(
    token_ids: list[int], request: GreedyGenerationRequest
) -> list[int]:
    limit = request.truncate_prompt_tokens
    if limit is None or limit == -1 or len(token_ids) <= limit:
        return token_ids
    if request.truncation_side == "right":
        return token_ids[:limit]
    return token_ids[-limit:]


def _effective_tool_choice(
    request: ChatCompletionRequest,
) -> Union[Literal["none", "auto", "required"], NamedToolChoice]:
    if request.tool_choice is not None:
        return request.tool_choice
    return "auto" if request.tools else "none"


def _tool_parsing_enabled(request: ChatCompletionRequest) -> bool:
    return bool(request.tools and _effective_tool_choice(request) != "none")


def _serialized_tool_choice(
    request: ChatCompletionRequest,
) -> Union[str, Dict[str, Any]]:
    tool_choice = _effective_tool_choice(request)
    if isinstance(tool_choice, NamedToolChoice):
        return tool_choice.model_dump()
    return tool_choice


def _tool_template_kwargs(request: ChatCompletionRequest) -> Dict[str, Any]:
    if not _tool_parsing_enabled(request):
        return {}
    return {
        "tools": [tool.model_dump(exclude_none=True) for tool in request.tools or []],
        "tool_choice": _serialized_tool_choice(request),
        "parallel_tool_calls": request.parallel_tool_calls is not False,
    }


def _tool_call_parse_options(request: ChatCompletionRequest) -> Dict[str, Any]:
    tool_choice = _effective_tool_choice(request)
    required_name = (
        tool_choice.function.name
        if isinstance(tool_choice, NamedToolChoice)
        else None
    )
    return {
        "allowed_names": {tool.function.name for tool in request.tools or []},
        "required_name": required_name,
        "require_call": tool_choice == "required" or required_name is not None,
        "allow_parallel": request.parallel_tool_calls is not False,
    }


def _tool_parser_for(request: ChatCompletionRequest, spec: ModelSpec) -> Optional[str]:
    """Resolve and validate the model-specific parser before generation starts."""
    if not _tool_parsing_enabled(request):
        return None
    try:
        parser_name = resolve_tool_parser(
            spec.name,
            getattr(spec, "model_path", ""),
            getattr(spec, "tool_parser", None),
        )
    except ValueError as exc:
        raise _error("tools", str(exc)) from exc
    if parser_name is None:
        raise _error(
            "tools",
            "The selected model has no known tool-call parser. "
            "Set 'tool_parser' in its model configuration to one of the "
            "supported vLLM parsers.",
        )
    return parser_name


def _chat_prompt_token_ids(request: ChatCompletionRequest, tokenizer) -> List[int]:
    messages = normalize_tool_call_messages(
        [m.model_dump(exclude_none=True) for m in request.messages]
    )
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": request.add_generation_prompt,
    }
    normalized_model_name = request.model.lower().replace("-", "").replace("_", "")
    if "qwen3" in normalized_model_name:
        # Qwen3 enables its reasoning block by default. Render the non-thinking
        # assistant prefix so every Aegaeon API caller gets direct answers.
        template_kwargs["enable_thinking"] = False
    if request.continue_final_message:
        template_kwargs["continue_final_message"] = True
    template_kwargs.update(_tool_template_kwargs(request))

    try:
        rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
        token_ids = tokenizer.encode(
            rendered, add_special_tokens=request.add_special_tokens
        )
    except Exception as exc:
        raise _error(
            "messages",
            "The selected model cannot render these chat messages with its "
            f"chat template: {exc}",
        )
    return _truncate_prompt(list(token_ids), request)


async def _chat_engine_input(
    ctx: ServerContext,
    request: ChatCompletionRequest,
    spec,
    max_tokens: int,
) -> tuple[list[int], dict[str, Any] | None]:
    """Use vLLM's renderer/processor when OpenAI content parts are present."""
    if all(not isinstance(message.content, list) for message in request.messages):
        tokenizer = _tokenizer_for(spec)
        return _chat_prompt_token_ids(request, tokenizer), None

    from vllm.renderers import ChatParams, TokenizeParams, renderer_from_config

    from aegaeon.config import (
        ParallelConfig,
        get_model_config,
        get_vllm_config,
    )

    parallel_config = ParallelConfig(tensor_parallel_size=ctx.tensor_parallel_size)
    model_config = get_model_config(spec)
    vllm_config = get_vllm_config(model_config, parallel_config)
    renderer = renderer_from_config(vllm_config)
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": request.add_generation_prompt,
    }
    if request.continue_final_message:
        template_kwargs["continue_final_message"] = True
    template_kwargs.update(_tool_template_kwargs(request))
    normalized_model_name = request.model.lower().replace("-", "").replace("_", "")
    if "qwen3" in normalized_model_name:
        template_kwargs["enable_thinking"] = False

    messages = normalize_tool_call_messages(
        [message.model_dump(exclude_none=True) for message in request.messages]
    )
    try:
        _, engine_inputs = await renderer.render_chat_async(
            [messages],
            ChatParams(
                chat_template_kwargs=template_kwargs,
                mm_processor_kwargs=request.mm_processor_kwargs,
            ),
            TokenizeParams(
                max_total_tokens=model_config.max_model_len,
                max_output_tokens=max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                truncation_side=request.truncation_side,
                add_special_tokens=request.add_special_tokens,
            ),
        )
    except Exception as exc:
        raise _error("messages", f"Multimodal input processing failed: {exc}")

    engine_input = engine_inputs[0]
    prompt_ids = list(engine_input["prompt_token_ids"])
    multimodal_input = (
        dict(engine_input) if engine_input["type"] == "multimodal" else None
    )
    return prompt_ids, multimodal_input


def _completion_prompt_token_ids(request: CompletionRequest, tokenizer) -> list[int]:
    if isinstance(request.prompt, str):
        if not request.prompt:
            raise _error("prompt", "prompt must not be empty")
        token_ids = tokenizer.encode(
            request.prompt, add_special_tokens=request.add_special_tokens
        )
    else:
        if not request.prompt:
            raise _error("prompt", "prompt token list must not be empty")
        if any(token_id < 0 for token_id in request.prompt):
            raise _error("prompt", "prompt token ids must be non-negative")
        token_ids = request.prompt
    return _truncate_prompt(list(token_ids), request)


def _stop_strings(request: GreedyGenerationRequest) -> list[str]:
    if request.stop is None:
        return []
    values = [request.stop] if isinstance(request.stop, str) else request.stop
    if any(value == "" for value in values):
        raise _error("stop", "stop strings must not be empty")
    return values


def _decode_token_ids(tokenizer, token_ids: list[int], request) -> str:
    """Decode without cleanup that could rewrite text already sent to a client."""
    # vLLM's model-specific tool parsers inspect structural tokens (for
    # example, ``<tool_call>`` or GPT-OSS channel markers).  Hugging Face
    # tokenizers may classify those markers as special tokens, so dropping
    # them before parsing would turn a valid call into ordinary assistant
    # text.  Preserve the markup only for tool-enabled chat requests; normal
    # chat/completions keep their caller-selected decoding behavior.
    preserve_tool_markup = (
        isinstance(request, ChatCompletionRequest)
        and _tool_parsing_enabled(request)
    )
    kwargs = {
        "skip_special_tokens": (
            False if preserve_tool_markup else request.skip_special_tokens
        ),
        "spaces_between_special_tokens": (
            False if preserve_tool_markup else request.spaces_between_special_tokens
        ),
        "clean_up_tokenization_spaces": False,
    }
    try:
        return tokenizer.decode(token_ids, **kwargs)
    except TypeError:
        # Some custom tokenizers do not expose the Hugging Face cleanup option.
        kwargs.pop("clean_up_tokenization_spaces")
        return tokenizer.decode(token_ids, **kwargs)


def _decode_result(
    outputs,
    tokenizer,
    request: GreedyGenerationRequest,
    spec,
    prompt_token_ids: list[int],
    internal_id: int,
) -> GenerationResult:
    raw_ids = [int(output.new_token_id) for output in outputs]
    stop_ids = _stop_token_ids(request, tokenizer, spec)

    finish_reason = "length"
    kept_ids: list[int] = []
    for token_id in raw_ids:
        if len(kept_ids) >= request.min_tokens and token_id in stop_ids:
            finish_reason = "stop"
            break
        kept_ids.append(token_id)

    text = _decode_token_ids(tokenizer, kept_ids, request)

    stop_search_start = 0
    if request.min_tokens:
        stop_search_start = len(
            _decode_token_ids(tokenizer, kept_ids[: request.min_tokens], request)
        )

    earliest = None
    matched_stop = None
    for stop_string in _stop_strings(request):
        position = text.find(stop_string, stop_search_start)
        if position >= 0 and (earliest is None or position < earliest):
            earliest = position
            matched_stop = stop_string
    if earliest is not None and matched_stop is not None:
        end = earliest + (
            len(matched_stop) if request.include_stop_str_in_output else 0
        )
        text = text[:end]
        kept_ids = list(tokenizer.encode(text, add_special_tokens=False))
        finish_reason = "stop"

    return GenerationResult(
        text, kept_ids, prompt_token_ids, finish_reason, internal_id
    )


def _prepare_generation(
    ctx: ServerContext,
    spec,
    prompt_token_ids: list[int],
    max_tokens: int,
    ignore_eos: bool = False,
    stop_token_ids: list[int] | None = None,
    min_tokens: int = 0,
    multimodal_input: dict[str, Any] | None = None,
) -> GenerationHandle:
    llm_service = ctx.llm_service
    model_config = get_model_config(spec)
    prompt_tokens = len(prompt_token_ids)
    if prompt_tokens >= model_config.max_model_len:
        raise _error(
            "prompt",
            f"Prompt contains {prompt_tokens} tokens, but model '{spec.name}' "
            f"has a maximum context length of {model_config.max_model_len}. "
            "The prompt must leave room for at least one generated token.",
        )
    internal_id = next(llm_service.counter)
    if multimodal_input is None:
        if model_config.uses_mrope:
            # A decode engine that did not run prefill still needs enough
            # information to reconstruct the text-only M-RoPE delta.
            multimodal_input = {
                "type": "mrope",
                "prompt_token_ids": prompt_token_ids,
                "mm_kwargs": {},
                "mm_placeholders": {},
                "mm_hashes": {},
            }
    aegaeon_req = AegaeonRequest(
        model=spec,
        arrival_time=time.time(),
        request_id=internal_id,
        prompt_token_ids=prompt_token_ids,
        decode_tokens=max_tokens,
        ignore_eos=ignore_eos,
        eos_token_ids=model_config.eos_token_ids,
        stop_token_ids=stop_token_ids,
        min_tokens=min_tokens,
        max_model_len=model_config.max_model_len,
        multimodal_input=multimodal_input,
    )
    try:
        node = llm_service.reserve_node(spec, internal_id)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return GenerationHandle(
        internal_id,
        node,
        aegaeon_req,
        routing_reserved=True,
        routing_service=llm_service,
    )


async def _generate(
    ctx: ServerContext,
    spec,
    prompt_token_ids: list[int],
    request: GreedyGenerationRequest,
    max_tokens: int,
    tokenizer,
    multimodal_input: dict[str, Any] | None = None,
) -> GenerationResult:
    handle = _prepare_generation(
        ctx,
        spec,
        prompt_token_ids,
        max_tokens,
        ignore_eos=request.ignore_eos,
        stop_token_ids=request.stop_token_ids,
        min_tokens=request.min_tokens,
        multimodal_input=multimodal_input,
    )
    try:
        outputs = await handle.node.serve.remote(handle.request)
        return _decode_result(
            outputs,
            tokenizer,
            request,
            spec,
            prompt_token_ids,
            handle.internal_id,
        )
    except RequestCapacityError as exc:
        raise HTTPException(
            status_code=400, detail=_request_capacity_error_payload(exc)
        ) from exc
    finally:
        handle.release_routing()


def _stop_token_ids(request: GreedyGenerationRequest, tokenizer, spec) -> set[int]:
    stop_ids = set(request.stop_token_ids or [])
    if request.ignore_eos:
        return stop_ids

    stop_ids.update(get_model_config(spec).eos_token_ids)
    eos_token_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_token_ids is not None:
        if isinstance(eos_token_ids, int):
            stop_ids.add(eos_token_ids)
        else:
            stop_ids.update(int(value) for value in eos_token_ids)
    return stop_ids


def _visible_stream_text(
    text: str,
    stop_strings: list[str],
    stop_search_start: int,
    include_stop: bool,
    final: bool = False,
) -> tuple[str, bool]:
    """Return text safe to emit and whether a complete stop string was seen.

    A suffix that is only a prefix of a stop string is retained until another
    token proves whether it is ordinary output or the beginning of the stop
    sequence.  This prevents a multi-token stop string from leaking to clients.
    """
    earliest: int | None = None
    matched_stop: str | None = None
    for stop_string in stop_strings:
        position = text.find(stop_string, stop_search_start)
        if position >= 0 and (earliest is None or position < earliest):
            earliest = position
            matched_stop = stop_string

    if earliest is not None and matched_stop is not None:
        end = earliest + (len(matched_stop) if include_stop else 0)
        return text[:end], True

    if final:
        return text, False

    holdback = 0
    for stop_string in stop_strings:
        max_prefix = min(len(stop_string) - 1, len(text))
        for prefix_len in range(1, max_prefix + 1):
            start = len(text) - prefix_len
            if start >= stop_search_start and text.endswith(stop_string[:prefix_len]):
                holdback = max(holdback, prefix_len)

    visible = text[:-holdback] if holdback else text
    # Byte-fallback tokenizers can temporarily decode an incomplete character
    # as U+FFFD.  Do not send a character that the next token would rewrite.
    visible = visible.rstrip("\ufffd")
    return visible, False


def _text_delta(previous: str, current: str) -> str:
    """Return the append-only portion of cumulative decoded text."""
    if current.startswith(previous):
        return current[len(previous) :]
    # Cleanup is disabled above, so this is only a defensive fallback for a
    # custom tokenizer whose cumulative decoding is not prefix-stable.
    return current[len(previous) :] if len(current) > len(previous) else ""


async def _release_stream_when_finished(handle: GenerationHandle, remote_cleanup) -> None:
    """Keep routing load reserved until disconnected remote work is done."""
    try:
        await remote_cleanup
    finally:
        handle.release_routing()


def _schedule_stream_cleanup(handle: GenerationHandle) -> None:
    """Schedule Controller cleanup without blocking a disconnected client."""
    try:
        remote_cleanup = handle.node.release_request_when_finished.remote(
            handle.internal_id
        )
    except BaseException:
        handle.release_routing()
        raise

    task = asyncio.create_task(_release_stream_when_finished(handle, remote_cleanup))
    _STREAM_CLEANUP_TASKS.add(task)
    task.add_done_callback(_STREAM_CLEANUP_TASKS.discard)


async def _generate_stream(
    handle: GenerationHandle,
    tokenizer,
    request: GreedyGenerationRequest,
    prompt_token_ids: list[int],
) -> AsyncGenerator[tuple[str | None, float | None, GenerationResult | None], None]:
    """Yield decoded text deltas as soon as the controller produces tokens."""
    cursor = 0
    kept_ids: list[int] = []
    emitted_text = ""
    final_text = ""
    finish_reason = "length"
    logically_stopped = False
    submitted = False
    released = False
    stop_ids = _stop_token_ids(request, tokenizer, handle.request.model)
    stop_strings = _stop_strings(request)
    decode_rate = DecodeRateEMA()

    try:
        await handle.node.submit.remote(handle.request)
        submitted = True

        backend_finished = False
        while not backend_finished:
            outputs, backend_finished = await handle.node.wait_for_step_outputs.remote(
                handle.internal_id, cursor
            )
            cursor += len(outputs)
            current_decode_rate = decode_rate.observe(len(outputs))

            for output in outputs:
                if logically_stopped:
                    continue

                token_id = int(output.new_token_id)
                if len(kept_ids) >= request.min_tokens and token_id in stop_ids:
                    finish_reason = "stop"
                    logically_stopped = True
                    final_text = _decode_token_ids(tokenizer, kept_ids, request)
                    visible = final_text
                else:
                    kept_ids.append(token_id)
                    decoded = _decode_token_ids(tokenizer, kept_ids, request)
                    stop_search_start = 0
                    if request.min_tokens:
                        stop_search_start = len(
                            _decode_token_ids(
                                tokenizer, kept_ids[: request.min_tokens], request
                            )
                        )
                    visible, matched_stop = _visible_stream_text(
                        decoded,
                        stop_strings,
                        stop_search_start,
                        request.include_stop_str_in_output,
                    )
                    if matched_stop:
                        finish_reason = "stop"
                        logically_stopped = True
                        final_text = visible
                        kept_ids = list(
                            tokenizer.encode(final_text, add_special_tokens=False)
                        )
                    else:
                        final_text = decoded
                delta = _text_delta(emitted_text, visible)
                if delta:
                    emitted_text = visible
                    yield delta, current_decode_rate, None

        if not logically_stopped:
            final_text = _decode_token_ids(tokenizer, kept_ids, request)
        delta = _text_delta(emitted_text, final_text)
        if delta:
            emitted_text = final_text
            yield delta, decode_rate.value, None

        await handle.node.release_request.remote(handle.internal_id)
        released = True
        handle.release_routing()
        yield (
            None,
            decode_rate.value,
            GenerationResult(
                final_text,
                kept_ids,
                prompt_token_ids,
                finish_reason,
                handle.internal_id,
            ),
        )
    finally:
        if submitted and not released:
            # Do not leave the controller's per-request output list behind if
            # the HTTP client disconnects midway through generation.
            _schedule_stream_cleanup(handle)
        else:
            handle.release_routing()


def _response_id(prefix: str, request_id: str | None, fallback: int) -> str:
    return f"{prefix}-{request_id if request_id else fallback}"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _request_capacity_error_payload(exc: RequestCapacityError) -> dict[str, Any]:
    # Ray wraps remote causes in a dynamic subclass; show the capacity reason
    # without the Ray worker traceback in the client-visible message.
    cause = getattr(exc, "cause", None)
    error = cause if isinstance(cause, RequestCapacityError) else exc
    return {
        "error": {
            "message": str(error),
            "type": "request_capacity_error",
            "code": None,
        }
    }


async def _capacity_checked_stream(
    iterator: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Finish a rejected SSE request with an explicit error and terminator."""
    try:
        async for chunk in iterator:
            yield chunk
    except RequestCapacityError as exc:
        yield _sse(_request_capacity_error_payload(exc))
        yield "data: [DONE]\n\n"


def _parsed_chat_output(
    text: str,
    request: ChatCompletionRequest,
    tokenizer,
    parser_name: Optional[str],
) -> ParsedToolCalls:
    if parser_name is not None:
        return parse_tool_calls_with_vllm(
            text,
            parser_name=parser_name,
            tokenizer=tokenizer,
            tools=[
                tool.model_dump(exclude_none=True) for tool in request.tools or []
            ],
            tool_choice=_serialized_tool_choice(request),
            model_name=request.model,
            **_tool_call_parse_options(request),
        )
    return ParsedToolCalls(text, [])


def _chat_stream(
    response_id: str,
    created: int,
    model: str,
    handle: GenerationHandle,
    tokenizer,
    prompt_token_ids: list[int],
    request: ChatCompletionRequest,
    parser_name: Optional[str],
):
    async def iterator():
        first_chunk: Dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        if request.return_token_ids:
            first_chunk["prompt_token_ids"] = prompt_token_ids
        yield _sse(first_chunk)

        result: GenerationResult | None = None
        async for delta, decode_tokens_per_second, completed in _generate_stream(
            handle, tokenizer, request, prompt_token_ids
        ):
            if completed is not None:
                result = completed
                break
            if not delta:
                continue
            if parser_name is not None:
                # Tool parsers need the complete model output. Keep tool-enabled
                # SSE deterministic and emit one parsed delta after generation.
                continue
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None,
                    }
                ],
            }
            if decode_tokens_per_second is not None:
                chunk["decode_tokens_per_second"] = decode_tokens_per_second
            yield _sse(chunk)
        if result is None:
            return
        await _ACCESS_RATE.record_tokens(model, result.completion_tokens)

        finish_reason = result.finish_reason
        if parser_name is not None:
            parsed = _parsed_chat_output(
                result.text, request, tokenizer, parser_name
            )
            parsed_delta: Dict[str, Any] = {}
            if parsed.content is not None:
                parsed_delta["content"] = parsed.content
            if parsed.tool_calls:
                parsed_delta["tool_calls"] = [
                    {"index": index, **tool_call}
                    for index, tool_call in enumerate(parsed.tool_calls)
                ]
                finish_reason = "tool_calls"
            if parsed_delta:
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": parsed_delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                )

        final_choice: Dict[str, Any] = {
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }
        if request.return_token_ids:
            final_choice["token_ids"] = result.token_ids
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [final_choice],
        }
        if decode_tokens_per_second is not None:
            final_chunk["decode_tokens_per_second"] = decode_tokens_per_second
        yield _sse(final_chunk)
        if request.stream_options and request.stream_options.include_usage:
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": result.usage(),
                }
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _capacity_checked_stream(iterator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _completion_stream(
    response_id: str,
    created: int,
    model: str,
    echo_text: str,
    handle: GenerationHandle,
    tokenizer,
    prompt_token_ids: list[int],
    request: CompletionRequest,
):
    async def iterator():
        if echo_text:
            echo_choice: dict[str, Any] = {
                "index": 0,
                "text": echo_text,
                "logprobs": None,
                "finish_reason": None,
            }
            if request.return_token_ids:
                echo_choice["prompt_token_ids"] = prompt_token_ids
            yield _sse(
                {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [echo_choice],
                }
            )

        result: GenerationResult | None = None
        async for delta, decode_tokens_per_second, completed in _generate_stream(
            handle, tokenizer, request, prompt_token_ids
        ):
            if completed is not None:
                result = completed
                break
            if not delta:
                continue
            chunk = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": delta,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            if decode_tokens_per_second is not None:
                chunk["decode_tokens_per_second"] = decode_tokens_per_second
            yield _sse(chunk)
        if result is None:
            return
        await _ACCESS_RATE.record_tokens(model, result.completion_tokens)

        final_choice: dict[str, Any] = {
            "index": 0,
            "text": "",
            "logprobs": None,
            "finish_reason": result.finish_reason,
        }
        if request.return_token_ids:
            final_choice["token_ids"] = result.token_ids
            final_choice["prompt_token_ids"] = result.prompt_token_ids
        final_chunk = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [final_choice],
        }
        if decode_tokens_per_second is not None:
            final_chunk["decode_tokens_per_second"] = decode_tokens_per_second
        yield _sse(final_chunk)
        if request.stream_options and request.stream_options.include_usage:
            yield _sse(
                {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": result.usage(),
                }
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _capacity_checked_stream(iterator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(ctx: ServerContext = Depends(get_context)):
    created = int(time.time())
    deployed = ctx.deployed_models
    return {
        "object": "list",
        "data": [
            {
                "id": spec.name,
                "object": "model",
                "created": created,
                "owned_by": "aegaeon",
            }
            for spec in sorted(deployed.values(), key=lambda s: s.name)
        ],
    }


@app.get("/v1/models/{model_name:path}")
async def retrieve_model(model_name: str, ctx: ServerContext = Depends(get_context)):
    spec = _resolve_model(ctx, model_name)
    return {
        "id": spec.name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "aegaeon",
    }


@app.post("/v1/models/deploy")
async def deploy_model(
    request: DeployRequest, ctx: ServerContext = Depends(get_context)
):
    """Deploy a model to one or more node-local caches."""
    from aegaeon.model_downloader import download_model
    from aegaeon.model_registry import ModelSpec, get_registry

    model_name = request.model.strip()
    if not model_name:
        raise _error("model", "'model' is required.")

    deployed_models: dict[str, ModelSpec] = ctx.deployed_models
    existing_spec = deployed_models.get(model_name)
    registry = get_registry()
    registered_spec = registry.get_by_name(model_name)
    was_registered = registered_spec is not None

    if existing_spec is not None:
        if request.model_path:
            requested_path = os.path.abspath(request.model_path)
            if requested_path != existing_spec.model_path:
                raise _error(
                    "model_path",
                    "Cannot change model_path while replicas are deployed.",
                )
        model_path = existing_spec.model_path
        max_model_len = (
            request.max_model_len
            if request.max_model_len is not None
            else existing_spec.max_model_len
        )
    elif request.model_path:
        model_path = os.path.abspath(request.model_path)
        if not os.path.isdir(model_path):
            raise _error("model", f"model_path '{model_path}' does not exist")
        max_model_len = (
            request.max_model_len
            if request.max_model_len is not None
            else getattr(registered_spec, "max_model_len", None)
        )
    elif registered_spec is not None and registered_spec.model_path:
        model_path = os.path.abspath(registered_spec.model_path)
        if not os.path.isdir(model_path):
            raise _error(
                "model",
                f"configured model_path '{model_path}' does not exist",
            )
        max_model_len = (
            request.max_model_len
            if request.max_model_len is not None
            else registered_spec.max_model_len
        )
    else:
        model_path = await asyncio.to_thread(download_model, model_name)
        max_model_len = request.max_model_len

    config_changed = existing_spec is not None and (
        existing_spec.max_model_len != max_model_len
    )

    spec = ModelSpec(
        name=model_name,
        model_path=model_path,
        max_model_len=max_model_len,
        tool_parser=request.tool_parser,
    )
    spec = registry.register_dynamic(spec)  # update id

    try:
        result = await ctx.llm_service.deploy_model_replicas(
            spec,
            model_path,
            max_model_len,
            node_ids=request.node_ids,
            replica_count=request.replica_count,
            force=config_changed,
        )
    except ValueError as exc:
        raise _error("node_ids", str(exc)) from exc

    for evicted_models in result["evicted"].values():
        for name in evicted_models:
            if ctx.llm_service.model_placements.ready_nodes(name):
                continue
            deployed_models.pop(name, None)
            registry.unregister_dynamic(name)
            unregister_model_config(name)

    ready_nodes = ctx.llm_service.model_placements.ready_nodes(model_name)
    if not ready_nodes:
        if not was_registered:
            registry.unregister_dynamic(model_name)
        raise HTTPException(
            status_code=507,
            detail={
                "message": f"Failed to deploy '{model_name}' on all target nodes",
                "type": "insufficient_cache",
                "param": "model_cache_size_gb",
                "failures": result["failed"],
            },
        )

    register_model_config(spec)

    deployed_models[model_name] = spec
    return {
        "status": "partial" if result["failed"] else "ok",
        "model": model_name,
        "ready_nodes": list(ready_nodes),
        **result,
    }


@app.post("/v1/models/undeploy")
async def undeploy_model(
    request: DeployRequest, ctx: ServerContext = Depends(get_context)
):
    from aegaeon.model_registry import get_registry

    model_name = request.model.strip()
    if not model_name:
        raise _error("model", "'model' is required.")

    deployed_models: dict[str, ModelSpec] = ctx.deployed_models
    if model_name not in deployed_models:
        raise _error("model", f"Model '{model_name}' is not deployed.")

    try:
        result = await ctx.llm_service.undeploy_model_replicas(
            model_name,
            node_ids=request.node_ids,
        )
    except ValueError as exc:
        raise _error("node_ids", str(exc)) from exc

    remaining_nodes = ctx.llm_service.model_placements.nodes(model_name)
    if not remaining_nodes:
        from aegaeon.config import unregister_model_config

        del deployed_models[model_name]
        unregister_model_config(model_name)
        get_registry().unregister_dynamic(model_name)

    return {
        "status": "partial" if result["failed"] else "ok",
        "model": model_name,
        "remaining_nodes": list(remaining_nodes),
        **result,
    }


@app.get("/v1/aegaeon/runtime")
async def runtime_snapshot(ctx: ServerContext = Depends(get_context)):
    """Return current engine placement and recent scheduler switch events."""
    snapshots = await asyncio.gather(
        *(node.runtime_snapshot.remote(0) for node in ctx.llm_service.nodes)
    )
    events = sorted(
        (event for snapshot in snapshots for event in snapshot["events"]),
        key=lambda event: event["timestamp"],
    )
    return {
        "timestamp": time.time(),
        "model_placement_policy": type(
            ctx.llm_service.model_placement_policy
        ).__name__,
        "request_routing_policy": type(
            ctx.llm_service.request_routing_policy
        ).__name__,
        "last_model_placement_decision": getattr(
            ctx.llm_service, "last_model_placement_decision", None
        ),
        "model_placement_stats": ctx.llm_service.model_placement_stats_snapshot(),
        "model_placements": ctx.llm_service.model_placements.snapshot(),
        "request_routing": ctx.llm_service.request_routing_snapshot(),
        "engines": [engine for snapshot in snapshots for engine in snapshot["engines"]],
        "events": events,
        "work_stealing": [
            {
                "node_id": snapshot["node_id"],
                **snapshot["work_stealing"],
            }
            for snapshot in snapshots
            if snapshot["work_stealing"] is not None
        ],
        "decode_load": [
            {
                "node_id": snapshot["node_id"],
                **snapshot["decode_load"],
            }
            for snapshot in snapshots
            if snapshot["decode_load"] is not None
        ],
        "nodes": [
            {
                "node_id": snapshot["node_id"],
                "model_cache": snapshot["model_cache"],
                "request_load": snapshot["request_load"],
            }
            for snapshot in snapshots
        ],
    }


@app.get("/v1/aegaeon/metrics")
async def access_rate_metrics():
    """Return per-model request rate and token throughput (sliding window)."""
    return {
        "timestamp": time.time(),
        "window_seconds": _ACCESS_RATE.window,
        "limits": {
            "max_requests_per_second": _ACCESS_RATE.max_requests_per_second,
            "max_tokens_per_second": _ACCESS_RATE.max_tokens_per_second,
        },
        "metrics": await _ACCESS_RATE.snapshot(),
    }


@app.get("/v1/aegaeon/events")
async def runtime_events(ctx: ServerContext = Depends(get_context)):
    """Stream engine placement and model-switch events as SSE."""

    async def iterator():
        cursors: dict[int, int] = {}
        previous_state = ""
        last_heartbeat = time.monotonic()

        while True:
            snapshots = await asyncio.gather(
                *(
                    node.runtime_snapshot.remote(cursors.get(index, 0))
                    for index, node in enumerate(ctx.llm_service.nodes)
                )
            )
            events = []
            engines = []
            for index, snapshot in enumerate(snapshots):
                cursors[index] = snapshot["sequence"]
                engines.extend(snapshot["engines"])
                for event in snapshot["events"]:
                    event = dict(event)
                    event["id"] = f"{snapshot['node_id']}:{event['sequence']}"
                    events.append(event)

            events.sort(key=lambda event: event["timestamp"])
            state_key = json.dumps(engines, sort_keys=True, separators=(",", ":"))
            if events or state_key != previous_state:
                previous_state = state_key
                last_heartbeat = time.monotonic()
                yield (
                    "event: runtime\n"
                    f"data: {json.dumps({'engines': engines, 'events': events}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
            elif time.monotonic() - last_heartbeat >= 15:
                last_heartbeat = time.monotonic()
                yield ": keep-alive\n\n"

            await asyncio.sleep(0.25)

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tokenize")
async def tokenize(request: TokenizeRequest, ctx: ServerContext = Depends(get_context)):
    spec = _resolve_model(ctx, request.model)
    tokenizer = _tokenizer_for(spec)
    tokens = list(
        tokenizer.encode(request.prompt, add_special_tokens=request.add_special_tokens)
    )
    return {"count": len(tokens), "max_model_len": None, "tokens": tokens}


@app.post("/detokenize")
async def detokenize(
    request: DetokenizeRequest, ctx: ServerContext = Depends(get_context)
):
    spec = _resolve_model(ctx, request.model)
    tokenizer = _tokenizer_for(spec)
    prompt = tokenizer.decode(
        request.tokens, skip_special_tokens=request.skip_special_tokens
    )
    return {"prompt": prompt}


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest, ctx: ServerContext = Depends(get_context)
):
    spec = _resolve_model(ctx, request.model)
    violation = await _ACCESS_RATE.check(request.model)
    if violation is not None:
        raise HTTPException(status_code=429, detail=violation)
    await _ACCESS_RATE.record_request(request.model)
    parser_name = _tool_parser_for(request, spec)
    tokenizer = _tokenizer_for(spec)
    max_tokens = _effective_max_tokens(request, default=128)
    _validate_greedy_parameters(request, max_tokens)
    prompt_ids, multimodal_input = await _chat_engine_input(
        ctx, request, spec, max_tokens
    )
    if request.stream:
        handle = _prepare_generation(
            ctx,
            spec,
            prompt_ids,
            max_tokens,
            ignore_eos=request.ignore_eos,
            stop_token_ids=request.stop_token_ids,
            min_tokens=request.min_tokens,
            multimodal_input=multimodal_input,
        )
        response_id = _response_id("chatcmpl", request.request_id, handle.internal_id)
        return _chat_stream(
            response_id,
            int(time.time()),
            request.model,
            handle,
            tokenizer,
            prompt_ids,
            request,
            parser_name,
        )

    result = await _generate(
        ctx,
        spec,
        prompt_ids,
        request,
        max_tokens,
        tokenizer,
        multimodal_input=multimodal_input,
    )
    await _ACCESS_RATE.record_tokens(request.model, result.completion_tokens)
    response_id = _response_id("chatcmpl", request.request_id, result.internal_id)
    created = int(time.time())
    parsed = _parsed_chat_output(result.text, request, tokenizer, parser_name)
    finish_reason = "tool_calls" if parsed.tool_calls else result.finish_reason
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": parsed.content,
    }
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    choice: Dict[str, Any] = {
        "index": 0,
        "message": message,
        "logprobs": None,
        "finish_reason": finish_reason,
    }
    if request.return_token_ids:
        choice["token_ids"] = result.token_ids
    response: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [choice],
        "usage": result.usage(),
    }
    if request.return_token_ids:
        response["prompt_token_ids"] = result.prompt_token_ids
    return response


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest, ctx: ServerContext = Depends(get_context)
):
    spec = _resolve_model(ctx, request.model)
    violation = await _ACCESS_RATE.check(request.model)
    if violation is not None:
        raise HTTPException(status_code=429, detail=violation)
    await _ACCESS_RATE.record_request(request.model)
    tokenizer = _tokenizer_for(spec)
    max_tokens = _effective_max_tokens(request, default=16)
    _validate_greedy_parameters(request, max_tokens)
    prompt_ids = _completion_prompt_token_ids(request, tokenizer)
    prompt_text = ""
    if request.echo:
        prompt_text = (
            request.prompt
            if isinstance(request.prompt, str)
            else tokenizer.decode(
                prompt_ids, skip_special_tokens=request.skip_special_tokens
            )
        )
    if request.stream:
        handle = _prepare_generation(
            ctx,
            spec,
            prompt_ids,
            max_tokens,
            ignore_eos=request.ignore_eos,
            stop_token_ids=request.stop_token_ids,
            min_tokens=request.min_tokens,
        )
        response_id = _response_id("cmpl", request.request_id, handle.internal_id)
        return _completion_stream(
            response_id,
            int(time.time()),
            request.model,
            prompt_text,
            handle,
            tokenizer,
            prompt_ids,
            request,
        )

    result = await _generate(ctx, spec, prompt_ids, request, max_tokens, tokenizer)
    await _ACCESS_RATE.record_tokens(request.model, result.completion_tokens)
    output_text = prompt_text + result.text
    response_id = _response_id("cmpl", request.request_id, result.internal_id)
    created = int(time.time())
    choice: dict[str, Any] = {
        "index": 0,
        "text": output_text,
        "logprobs": None,
        "finish_reason": result.finish_reason,
    }
    if request.return_token_ids:
        choice["token_ids"] = result.token_ids
        choice["prompt_token_ids"] = result.prompt_token_ids
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [choice],
        "usage": result.usage(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Aegaeon API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray GCS address for Aegaeon; overrides RAY_ADDRESS",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Path to YAML config (models + server settings)",
    )
    parser.add_argument(
        "--models", type=str, required=True, help="Comma-separated model names to load"
    )
    parser.add_argument("--num-prefill-engines", type=int, default=1)
    parser.add_argument("--num-decode-engines", type=int, default=1)
    parser.add_argument("--num-engines", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model-cache-size-gb", type=int, default=20)
    parser.add_argument("--cpu-num-slabs", type=int, default=15)
    parser.add_argument("--cpu-slab-size-gb", type=int, default=1)
    parser.add_argument("--node-id", type=str, default="node_0")
    return parser.parse_args()


@app.get("/", include_in_schema=False)
async def demo_redirect():
    target = "/aegaeon_docs.html#live-demo" if os.path.isfile(_DOCS_FILE) else "/docs"
    return RedirectResponse(url=target)


_WEB_DIR = str(web_directory())

_DOCS_FILE = os.path.join(_WEB_DIR, "aegaeon_docs.html")

# Static assets referenced by aegaeon_docs.html after the single-file split.


def _docs_asset(asset_name: str, media_type: str) -> Response:
    asset_path = os.path.join(_WEB_DIR, asset_name)
    if not os.path.isfile(asset_path):
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(asset_path, media_type=media_type)


@app.get("/aegaeon_docs.css", include_in_schema=False)
async def docs_css():
    return _docs_asset("aegaeon_docs.css", "text/css")


@app.get("/aegaeon_docs.js", include_in_schema=False)
async def docs_js():
    return _docs_asset("aegaeon_docs.js", "application/javascript")


@app.get("/aegaeon_arena.js", include_in_schema=False)
async def arena_js():
    return _docs_asset("aegaeon_arena.js", "application/javascript")


@app.get("/aegaeon_console.html", include_in_schema=False)
async def integrated_console():
    console_file = os.path.join(_WEB_DIR, "aegaeon_console.html")
    if not os.path.isfile(console_file):
        raise HTTPException(status_code=404, detail="aegaeon_console.html not found")
    return FileResponse(console_file, media_type="text/html")


@app.get("/aegaeon_console.css", include_in_schema=False)
async def console_css():
    return _docs_asset("aegaeon_console.css", "text/css")


@app.get("/aegaeon_console.js", include_in_schema=False)
async def console_js():
    return _docs_asset("aegaeon_console.js", "application/javascript")


@app.get("/aegaeon_chat.html", include_in_schema=False)
async def integrated_chat():
    chat_file = os.path.join(_WEB_DIR, "aegaeon_chat.html")
    if not os.path.isfile(chat_file):
        raise HTTPException(status_code=404, detail="aegaeon_chat.html not found")
    return FileResponse(chat_file, media_type="text/html")


@app.get("/aegaeon_chat.css", include_in_schema=False)
async def chat_css():
    return _docs_asset("aegaeon_chat.css", "text/css")


@app.get("/aegaeon_chat.js", include_in_schema=False)
async def chat_js():
    return _docs_asset("aegaeon_chat.js", "application/javascript")


@app.get("/aegaeon_docs.html", include_in_schema=False)
async def integrated_docs():
    if not os.path.isfile(_DOCS_FILE):
        raise HTTPException(status_code=404, detail="aegaeon_docs.html not found")
    return FileResponse(_DOCS_FILE, media_type="text/html")


@app.get("/demo", include_in_schema=False)
@app.get("/demo/", include_in_schema=False)
async def integrated_demo_redirect():
    return await demo_redirect()


if __name__ == "__main__":
    args = parse_args()
    app.state.engine_args = args
    uvicorn.run(app, host=args.host, port=args.port)
