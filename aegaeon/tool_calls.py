"""Normalize model-specific tool-call output to the OpenAI chat shape."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Collection, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_XML_FUNCTION_PATTERN = re.compile(
    r"^\s*<function=([^>\r\n]+)>\s*(.*?)\s*</function>\s*$", re.DOTALL
)
_XML_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\r\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Parsers shipped by the pinned vLLM release that Aegaeon selects
# automatically. A configured parser is deliberately restricted to this set:
# accepting a typo and silently treating generated markup as assistant text is
# much harder to diagnose than rejecting the request.
SUPPORTED_TOOL_PARSERS = frozenset(
    {
        "deepseek_v3",
        "deepseek_v31",
        "deepseek_v32",
        "deepseek_v4",
        "functiongemma",
        "gemma4",
        "glm45",
        "glm47",
        "hermes",
        "hunyuan_a13b",
        "hy_v3",
        "kimi_k2",
        "mimo",
        "minimax_m2",
        "minimax_m3",
        "openai",
        "qwen3_coder",
        "qwen3_xml",
    }
)


class ToolParserError(RuntimeError):
    """Raised when a configured vLLM parser cannot process model output."""


@dataclass(frozen=True)
class ParsedToolCalls:
    content: Optional[str]
    tool_calls: list[dict]
    error: Optional[str] = None


@dataclass(frozen=True)
class ToolCallStreamUpdate:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


def _default_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def resolve_tool_parser(
    model_name: str,
    model_path: str = "",
    configured: Optional[str] = None,
) -> Optional[str]:
    """Resolve the vLLM parser for a representative tool-capable model.

    Public model names are often aliases, so both the served name and local
    checkpoint path participate in inference. For an alias that carries no
    vendor/model information, set ``tool_parser`` explicitly in model config.
    """
    if configured is not None:
        if not isinstance(configured, str):
            raise ValueError("tool_parser must be a string")
        parser_name = configured.strip()
        if parser_name not in SUPPORTED_TOOL_PARSERS:
            supported = ", ".join(sorted(SUPPORTED_TOOL_PARSERS))
            raise ValueError(
                f"Unsupported tool_parser {parser_name!r}; supported values: {supported}"
            )
        return parser_name

    key = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{model_name} {model_path}".lower(),
    ).strip("-")
    compact = key.replace("-", "")

    # Order matters: select version-specific parsers before broader families.
    if "gpt-oss" in key or "gptoss" in compact:
        return "openai"
    if "deepseek-v4" in key or "deepseekv4" in compact:
        return "deepseek_v4"
    if (
        "deepseek-v3-2" in key
        or "deepseek-v32" in key
        or "deepseekv32" in compact
    ):
        return "deepseek_v32"
    if (
        "deepseek-v3-1" in key
        or "deepseek-v31" in key
        or "deepseekv31" in compact
    ):
        return "deepseek_v31"
    if "deepseek-r1" in key or "deepseekr1" in compact:
        return "deepseek_v3"
    if "deepseek-v3" in key or "deepseekv3" in compact:
        return "deepseek_v3"
    if "qwen3-5" in key or "qwen35" in compact:
        return "qwen3_xml"
    if "qwen3-coder" in key or "qwen3coder" in compact:
        return "qwen3_coder"
    if "qwen3" in key or "qwen3" in compact:
        return "qwen3_xml"
    if "qwen" in key:
        return "hermes"
    if (
        "kimi-k2" in key
        or "kimi-k2-5" in key
        or "kimik2" in compact
    ):
        return "kimi_k2"
    if "functiongemma" in key:
        return "functiongemma"
    if "gemma-4" in key or "gemma4" in compact:
        return "gemma4"
    if "glm-4-7" in key or "glm47" in key or "glm47" in compact:
        return "glm47"
    if any(
        version in key
        for version in ("glm-4-5", "glm45", "glm-4-6", "glm46")
    ):
        return "glm45"
    if "glm45" in compact or "glm46" in compact:
        return "glm45"
    if "hunyuan-a13b" in key or "hunyuana13b" in compact:
        return "hunyuan_a13b"
    if (
        "hunyuan-v3" in key
        or "hy-v3" in key
        or "hunyuanv3" in compact
        or "hyv3" in compact
    ):
        return "hy_v3"
    if "mimo" in key:
        return "mimo"
    if "minimax-m3" in key or "minimaxm3" in compact:
        return "minimax_m3"
    if "minimax-m2" in key or "minimaxm2" in compact:
        return "minimax_m2"
    return None


def _is_inside_thinking(text: str, position: int) -> bool:
    """Return whether ``position`` is inside a complete ``<think>`` block."""
    for match in _THINK_PATTERN.finditer(text):
        if match.start() <= position < match.end():
            return True
    return False


def _parse_function(payload: object) -> Optional[tuple[str, str, Optional[str]]]:
    if not isinstance(payload, dict):
        return None

    call_id = payload.get("id")
    function = payload.get("function", payload)
    if not isinstance(function, dict):
        return None

    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        return None

    if isinstance(arguments, str):
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            return None
        encoded_arguments = arguments
    elif isinstance(arguments, (dict, list)):
        encoded_arguments = json.dumps(
            arguments, ensure_ascii=False, separators=(",", ":")
        )
    else:
        return None

    return name, encoded_arguments, call_id if isinstance(call_id, str) else None


def _parse_xml_function(body: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse the Qwen3-Coder/Qwen3.5 XML function-call body."""
    function_match = _XML_FUNCTION_PATTERN.match(body)
    if function_match is None:
        return None

    name = function_match.group(1).strip()
    parameter_text = function_match.group(2)
    if not name:
        return None

    arguments: dict[str, Any] = {}
    cursor = 0
    for match in _XML_PARAMETER_PATTERN.finditer(parameter_text):
        if parameter_text[cursor : match.start()].strip():
            return None
        parameter_name = match.group(1).strip()
        if not parameter_name or parameter_name in arguments:
            return None
        raw_value = match.group(2).strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        arguments[parameter_name] = value
        cursor = match.end()
    if parameter_text[cursor:].strip():
        return None

    encoded_arguments = json.dumps(
        arguments, ensure_ascii=False, separators=(",", ":")
    )
    return name, encoded_arguments, None


def _parse_tool_call_body(body: str) -> Optional[tuple[str, str, Optional[str]]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return _parse_xml_function(body)
    return _parse_function(payload)


def normalize_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make OpenAI assistant tool calls consumable by model chat templates.

    OpenAI transports ``function.arguments`` as a JSON string, while
    Transformers tool-use templates consume the decoded JSON value when
    rendering prior assistant calls.
    """
    normalized = deepcopy(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = {}
                continue
            if not isinstance(arguments, str):
                continue
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                pass
    return normalized


def _load_vllm_tool_parser_components():
    """Import the serving protocol lazily so pure parser tests need no vLLM."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager

    return ChatCompletionRequest, ToolParserManager


def _as_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return deepcopy(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else None
    # vLLM 0.26 uses Pydantic v2, but keeping the v1 spelling here makes the
    # adapter usable with custom parser plugins built against older Pydantic.
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict(exclude_none=True)
        return dumped if isinstance(dumped, dict) else None
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return deepcopy(attributes)
    return None


def _normalize_extracted_calls(
    values: Sequence[Any],
    call_id_factory: Callable[[], str],
) -> Optional[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for value in values:
        call = _as_mapping(value)
        if call is None:
            return None
        function = _as_mapping(call.get("function"))
        if function is None:
            return None
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            return None
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                return None
            encoded_arguments = arguments
        elif isinstance(arguments, (dict, list)):
            encoded_arguments = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
        else:
            return None

        call_id = call.get("id")
        calls.append(
            {
                "id": (
                    call_id
                    if isinstance(call_id, str) and call_id
                    else call_id_factory()
                ),
                "type": "function",
                "function": {"name": name, "arguments": encoded_arguments},
            }
        )
    return calls


def parse_tool_calls_with_vllm(
    text: str,
    *,
    parser_name: str,
    tokenizer: Any,
    tools: Sequence[dict[str, Any]],
    tool_choice: Any = "auto",
    model_name: str = "",
    call_id_factory: Callable[[], str] = _default_call_id,
    allowed_names: Optional[Collection[str]] = None,
    required_name: Optional[str] = None,
    require_call: bool = False,
    allow_parallel: bool = True,
) -> ParsedToolCalls:
    """Parse one complete generation with vLLM's model-specific parser.

    Aegaeon owns only the OpenAI-compatible transport. The model syntax itself
    remains delegated to the parser implementations shipped with the pinned
    vLLM version, avoiding a second set of subtly divergent format parsers.
    """
    if parser_name not in SUPPORTED_TOOL_PARSERS:
        raise ToolParserError(f"Unsupported tool parser: {parser_name}")

    try:
        request_class, parser_manager = _load_vllm_tool_parser_components()
        parser_request = request_class(
            messages=[{"role": "user", "content": ""}],
            # ChatCompletionRequest.model is required by vLLM. The API always
            # supplies the served model name; the fallback keeps this helper
            # valid for direct callers that only want to parse a string.
            model=model_name or "aegaeon",
            tools=list(tools),
            tool_choice=tool_choice,
            parallel_tool_calls=allow_parallel,
        )
        parser_class = parser_manager.get_tool_parser(parser_name)
        parser = parser_class(tokenizer, parser_request.tools)
        extracted = parser.extract_tool_calls(text, parser_request)
    except (ImportError, ModuleNotFoundError):
        # Keep the common Qwen/Hermes markup usable in lightweight tooling and
        # documentation environments where the heavy vLLM package is absent.
        # Production Aegaeon installations use the selected vLLM parser above.
        return parse_tool_calls(
            text,
            call_id_factory,
            allowed_names=allowed_names,
            required_name=required_name,
            require_call=require_call,
            allow_parallel=allow_parallel,
        )
    except Exception as exc:
        raise ToolParserError(
            f"vLLM parser {parser_name!r} failed: {exc}"
        ) from exc

    if not getattr(extracted, "tools_called", False):
        content = getattr(extracted, "content", None)
        return ParsedToolCalls(
            text if content is None else content,
            [],
            "required_tool_call_missing" if require_call else None,
        )

    raw_calls = getattr(extracted, "tool_calls", None)
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return ParsedToolCalls(text, [], "malformed_tool_call")
    calls = _normalize_extracted_calls(raw_calls, call_id_factory)
    if not calls:
        return ParsedToolCalls(text, [], "malformed_tool_call")
    if not allow_parallel and len(calls) > 1:
        return ParsedToolCalls(text, [], "parallel_tool_calls_disabled")

    for call in calls:
        name = call["function"]["name"]
        if allowed_names is not None and name not in allowed_names:
            return ParsedToolCalls(text, [], "unknown_tool")
        if required_name is not None and name != required_name:
            return ParsedToolCalls(text, [], "unexpected_tool")

    call_ids = [call["id"] for call in calls]
    if len(set(call_ids)) != len(call_ids):
        return ParsedToolCalls(text, [], "duplicate_tool_call_id")
    return ParsedToolCalls(getattr(extracted, "content", None), calls)


def parse_tool_calls(
    text: str,
    call_id_factory: Callable[[], str] = _default_call_id,
    *,
    allowed_names: Optional[Collection[str]] = None,
    required_name: Optional[str] = None,
    require_call: bool = False,
    allow_parallel: bool = True,
) -> ParsedToolCalls:
    """Parse complete Qwen/Hermes calls into OpenAI function-call objects.

    Tool-like text inside a Qwen reasoning block is deliberately ignored. If
    any tool-call block outside reasoning is malformed or unclosed, the entire
    generation remains ordinary content instead of returning a partial call.
    Optional request constraints keep unknown, forced-name, and disallowed
    parallel calls out of the structured response while exposing a stable
    error code for observability.
    """
    matches = [
        match
        for match in _TOOL_CALL_PATTERN.finditer(text)
        if not _is_inside_thinking(text, match.start())
    ]
    if not matches:
        error = "required_tool_call_missing" if require_call else None
        return ParsedToolCalls(text, [], error)

    outside_thinking = _THINK_PATTERN.sub("", text)
    if (
        outside_thinking.count("<tool_call>") != len(matches)
        or outside_thinking.count("</tool_call>") != len(matches)
    ):
        return ParsedToolCalls(text, [], "malformed_tool_call")

    if not allow_parallel and len(matches) > 1:
        return ParsedToolCalls(text, [], "parallel_tool_calls_disabled")

    parsed_calls = []
    content_parts = []
    cursor = 0
    for match in matches:
        parsed = _parse_tool_call_body(match.group(1))
        if parsed is None:
            return ParsedToolCalls(text, [], "malformed_tool_call")

        name, arguments, provided_id = parsed
        if allowed_names is not None and name not in allowed_names:
            return ParsedToolCalls(text, [], "unknown_tool")
        if required_name is not None and name != required_name:
            return ParsedToolCalls(text, [], "unexpected_tool")
        parsed_calls.append(
            {
                "id": provided_id or call_id_factory(),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        content_parts.append(text[cursor : match.start()])
        cursor = match.end()

    content_parts.append(text[cursor:])
    call_ids = [call["id"] for call in parsed_calls]
    if len(set(call_ids)) != len(call_ids):
        return ParsedToolCalls(text, [], "duplicate_tool_call_id")
    content = "".join(content_parts).strip() or None
    return ParsedToolCalls(content, parsed_calls)


def _partial_marker_suffix_length(text: str, markers: tuple[str, ...]) -> int:
    """Return the longest suffix that may be the start of a marker."""
    maximum = 0
    for marker in markers:
        for length in range(1, min(len(text), len(marker) - 1) + 1):
            if text.endswith(marker[:length]):
                maximum = max(maximum, length)
    return maximum


class ToolCallStreamParser:
    """Incrementally separate content from complete tool-call blocks.

    Function names and arguments are emitted as soon as one complete block can
    be validated. A short suffix is retained while it could still become an
    opening tag, so markup fragments never leak into an earlier content delta.
    """

    def __init__(
        self,
        call_id_factory: Callable[[], str] = _default_call_id,
        *,
        allowed_names: Optional[Collection[str]] = None,
        required_name: Optional[str] = None,
        require_call: bool = False,
        allow_parallel: bool = True,
    ) -> None:
        self._call_id_factory = call_id_factory
        self._allowed_names = allowed_names
        self._required_name = required_name
        self._require_call = require_call
        self._allow_parallel = allow_parallel
        self._buffer = ""
        self._mode = "text"
        self._tool_calls: list[dict] = []
        self._seen_call_ids: set[str] = set()
        self._error: Optional[str] = None
        self._finished = False

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._tool_calls)

    @property
    def error(self) -> Optional[str]:
        if self._error is not None:
            return self._error
        if self._finished and self._require_call and not self._tool_calls:
            return "required_tool_call_missing"
        return None

    def _set_error(self, error: Optional[str]) -> None:
        if self._error is None and error is not None:
            self._error = error

    def feed(self, delta: str) -> ToolCallStreamUpdate:
        if self._finished:
            raise RuntimeError("cannot feed a finished tool-call stream")
        self._buffer += delta
        content_parts: list[str] = []
        new_calls: list[dict] = []

        while self._buffer:
            if self._mode == "text":
                marker_positions = [
                    (self._buffer.find(marker), marker, mode)
                    for marker, mode in (
                        (_TOOL_CALL_OPEN, "tool"),
                        (_THINK_OPEN, "think"),
                    )
                ]
                marker_positions = [
                    item for item in marker_positions if item[0] >= 0
                ]
                if marker_positions:
                    position, _marker, mode = min(
                        marker_positions, key=lambda item: item[0]
                    )
                    if position:
                        content_parts.append(self._buffer[:position])
                        self._buffer = self._buffer[position:]
                    self._mode = mode
                    continue

                suffix_length = _partial_marker_suffix_length(
                    self._buffer, (_TOOL_CALL_OPEN, _THINK_OPEN)
                )
                if suffix_length:
                    content_parts.append(self._buffer[:-suffix_length])
                    self._buffer = self._buffer[-suffix_length:]
                else:
                    content_parts.append(self._buffer)
                    self._buffer = ""
                break

            closing_marker = (
                _TOOL_CALL_CLOSE if self._mode == "tool" else _THINK_CLOSE
            )
            closing_position = self._buffer.find(closing_marker)
            if closing_position < 0:
                break
            end = closing_position + len(closing_marker)
            block = self._buffer[:end]
            self._buffer = self._buffer[end:]

            if self._mode == "think":
                content_parts.append(block)
            elif not self._allow_parallel and self._tool_calls:
                content_parts.append(block)
                self._set_error("parallel_tool_calls_disabled")
            else:
                parsed = parse_tool_calls(
                    block,
                    self._call_id_factory,
                    allowed_names=self._allowed_names,
                    required_name=self._required_name,
                )
                if not parsed.tool_calls:
                    content_parts.append(block)
                    self._set_error(parsed.error or "malformed_tool_call")
                else:
                    tool_call = parsed.tool_calls[0]
                    if tool_call["id"] in self._seen_call_ids:
                        content_parts.append(block)
                        self._set_error("duplicate_tool_call_id")
                    else:
                        self._seen_call_ids.add(tool_call["id"])
                        self._tool_calls.append(tool_call)
                        new_calls.append(tool_call)
            self._mode = "text"

        return ToolCallStreamUpdate("".join(content_parts), new_calls)

    def finish(self) -> ToolCallStreamUpdate:
        if self._finished:
            return ToolCallStreamUpdate()
        self._finished = True
        content = self._buffer
        if self._mode == "tool":
            self._set_error("malformed_tool_call")
        self._buffer = ""
        self._mode = "text"
        return ToolCallStreamUpdate(content, [])
