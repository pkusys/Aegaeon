#!/usr/bin/env python3
"""Aegaeon parameter conformance & matrix tester (local / SSH remote).

Two execution modes:

1. Local mode (default) - test a running Aegaeon server over HTTP:
     python3 tools/param_test.py --host 127.0.0.1 --port 8000 --models Qwen/Qwen3-4B

2. Remote mode - orchestrate everything over SSH (e.g. `ssh gpu-server`):
     python3 tools/param_test.py --remote gpu-server --models Qwen/Qwen3-4B

   Remote mode: uploads this script to the server, optionally starts
   Ray + the Aegaeon API server (unless --no-start), runs the matrix
   tests on the server, fetches the JSON report back and prints it.

Server-side subcommands (invoked by remote mode internally):
     python3 tools/param_test.py --server-side status|start|stop|test ...

Only the Python standard library is used so it runs on any python3.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Default model path map for the generated server test config (remote mode).
# --------------------------------------------------------------------------
DEFAULT_MODEL_PATHS = {
    "Qwen/Qwen3-4B": "/srv/models/Qwen/Qwen3-4B",
    "Qwen/Qwen3-0.6B": "/srv/models/Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-7B-Instruct": "/srv/models/Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "/srv/models/Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3.5-9B": "/srv/models/Qwen/Qwen3.5-9B",
}
MODEL_PARAMS = {  # approximate parameter counts (B), used for cache sizing
    "Qwen/Qwen3-4B": 4.1,
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen2.5-7B-Instruct": 7.1,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.67,
    "Qwen/Qwen3.5-9B": 9.0,
}

CHAT = {
    "model": "MODEL",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "max_tokens": 32,
}


def _case(cid, group, endpoint, payload, expect, method="POST", skip=False):
    return {
        "id": cid,
        "group": group,
        "endpoint": endpoint,
        "method": method,
        "payload": payload,
        "expect": expect,
        "skip": skip,
    }


def build_matrix(models):
    """Build the parameter matrix. ``models`` is a list of model names."""
    m = models[0] if models else "MODEL"

    def with_model(payload):
        out = dict(payload)
        if "model" in out and out["model"] == "MODEL":
            out["model"] = m
        return out

    cases = [
        # ---- health / model listing -------------------------------------
        _case("health.ok", "health", "/health", None, {"status": 200}, method="GET"),
        _case("models.list", "models", "/v1/models", None,
              {"status": 200, "checks": ["model_in_list"]}, method="GET"),
        _case("models.get", "models", "/v1/models/{model}", None,
              {"status": 200}, method="GET"),
        _case("models.get_missing", "models", "/v1/models/definitely-not-a-model", None,
              {"status": 400, "checks": ["error_param:model"]}, method="GET"),

        # ---- chat: valid parameter combinations -------------------------
        _case("chat.basic", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 32},
              {"status": 200, "checks": ["has_text", "usage", "finish_reason"]}),
        _case("chat.max_tokens_1", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 1},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.max_completion_tokens", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_completion_tokens": 16},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.both_max_match", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"],
               "max_tokens": 16, "max_completion_tokens": 16},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.stop_str", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 64, "stop": " France"},
              {"status": 200, "checks": ["has_text", "finish_reason"]}),
        _case("chat.stop_list", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 64,
               "stop": ["\n", "France"]},
              {"status": 200, "checks": ["has_text", "finish_reason"]}),
        _case("chat.stop_token_ids", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 32,
               "stop_token_ids": [13]},
              {"status": 200, "checks": ["has_text", "finish_reason"]}),
        _case("chat.include_stop_str", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 32,
               "stop": "Paris", "include_stop_str_in_output": True},
              {"status": 200, "checks": ["has_text", "finish_reason"]}),
        _case("chat.seed", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16, "seed": 42},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.greedy_explicit", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "n": 1, "temperature": 0, "top_p": 1, "top_k": 1, "min_p": 0,
               "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1,
               "length_penalty": 1.0, "priority": 0},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.logprobs_off", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "logprobs": False, "top_logprobs": 0},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.stream", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16, "stream": True},
              {"status": 200, "checks": ["sse"]}),
        _case("chat.stream_usage", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "stream": True, "stream_options": {"include_usage": True}},
              {"status": 200, "checks": ["sse", "sse_usage"]}),
        _case("chat.return_token_ids", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "return_token_ids": True},
              {"status": 200, "checks": ["has_text", "usage", "token_ids"]}),
        _case("chat.user", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "user": "alice"},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.request_id", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "request_id": "paramtest-abc123"},
              {"status": 200, "checks": ["has_text", "request_id:chatcmpl-paramtest-abc123"]}),
        _case("chat.multi_turn", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [
                  {"role": "system", "content": "You are terse."},
                  {"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Hello"},
                  {"role": "user", "content": "What is 2+2?"},
               ], "max_tokens": 16},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.tool_result_followup", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [
                  {"role": "user", "content": "What is the weather in Beijing?"},
                  {"role": "assistant", "content": None, "tool_calls": [{
                      "id": "call-paramtest-weather", "type": "function",
                      "function": {"name": "get_weather",
                                   "arguments": "{\"city\":\"北京\"}"},
                  }]},
                  {"role": "tool", "tool_call_id": "call-paramtest-weather",
                   "name": "get_weather",
                   "content": "{\"ok\":true,\"result\":{\"weather\":\"晴\"}}"},
               ], "max_tokens": 32,
               "tools": [{"type": "function", "function": {
                   "name": "get_weather", "parameters": {"type": "object"},
               }}], "tool_choice": "auto", "parallel_tool_calls": False},
              {"status": 200, "checks": ["usage", "finish_reason"]}),
        _case("chat.ignore_eos", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 16,
               "ignore_eos": True},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.min_tokens", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 32,
               "min_tokens": 4},
              {"status": 200, "checks": ["has_text", "usage", "min_tokens:4"]}),
        _case("chat.truncate_prompt", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL",
               "messages": [{"role": "user", "content": "word " * 200}],
               "max_tokens": 8, "truncate_prompt_tokens": 32},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.truncate_prompt_neg1", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "truncate_prompt_tokens": -1},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.truncate_right", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL",
               "messages": [{"role": "user", "content": "word " * 200}],
               "max_tokens": 8, "truncate_prompt_tokens": 32, "truncation_side": "right"},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.decode_flags", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "skip_special_tokens": True, "spaces_between_special_tokens": True},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.continue_final", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [
                  {"role": "user", "content": "List three fruits:"},
                  {"role": "assistant", "content": "1. Apple, 2."},
               ], "max_tokens": 8, "continue_final_message": True,
               "add_generation_prompt": False},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("chat.tools_auto", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 32,
               "tools": [{"type": "function", "function": {
                   "name": "get_weather",
                   "description": "Get the weather for a city.",
                   "parameters": {"type": "object", "properties": {
                       "city": {"type": "string"}}, "required": ["city"]},
               }}], "tool_choice": "auto"},
              {"status": 200, "checks": ["usage", "finish_reason"]}),
        _case("chat.tools_required", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [{"role": "user", "content":
                   "Use get_weather to check the weather in Beijing."}], "max_tokens": 64,
               "tools": [{"type": "function", "function": {
                   "name": "get_weather",
                   "parameters": {"type": "object", "properties": {
                       "city": {"type": "string"}}, "required": ["city"]},
               }}], "tool_choice": "required"},
              {"status": 200, "checks": ["usage", "tool_calls"]}),
        _case("chat.tools_named", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [{"role": "user", "content":
                   "Use get_weather to check the weather in Beijing."}], "max_tokens": 64,
               "tools": [{"type": "function", "function": {
                   "name": "get_weather", "parameters": {"type": "object"},
               }}], "tool_choice": {"type": "function", "function": {
                   "name": "get_weather"}}, "parallel_tool_calls": False},
              {"status": 200, "checks": ["usage", "tool_calls"]}),
        _case("chat.tools_named_stream", "chat-valid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [{"role": "user", "content":
                   "Use get_weather to check the weather in Beijing."}], "max_tokens": 64,
               "stream": True, "tools": [{"type": "function", "function": {
                   "name": "get_weather", "parameters": {"type": "object"},
               }}], "tool_choice": {"type": "function", "function": {
                   "name": "get_weather"}}, "parallel_tool_calls": False},
              {"status": 200, "checks": ["sse_tool_calls"]}),

        # ---- chat: explicit greedy-equivalent rejection (400) ------------
        _case("chat.reject.temperature", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "temperature": 0.7},
              {"status": 400, "checks": ["error_param:temperature"]}),
        _case("chat.reject.top_p", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "top_p": 0.9},
              {"status": 400, "checks": ["error_param:top_p"]}),
        _case("chat.reject.top_k", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "top_k": 3},
              {"status": 400, "checks": ["error_param:top_k"]}),
        _case("chat.reject.min_p", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "min_p": 0.5},
              {"status": 400, "checks": ["error_param:min_p"]}),
        _case("chat.reject.frequency_penalty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "frequency_penalty": 1.0},
              {"status": 400, "checks": ["error_param:frequency_penalty"]}),
        _case("chat.reject.presence_penalty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "presence_penalty": 1.0},
              {"status": 400, "checks": ["error_param:presence_penalty"]}),
        _case("chat.reject.repetition_penalty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "repetition_penalty": 1.2},
              {"status": 400, "checks": ["error_param:repetition_penalty"]}),
        _case("chat.reject.logit_bias", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "logit_bias": {"1234": -100}},
              {"status": 400, "checks": ["error_param:logit_bias"]}),
        _case("chat.reject.beam_search", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "use_beam_search": True},
              {"status": 400, "checks": ["error_param:use_beam_search"]}),
        _case("chat.reject.length_penalty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "length_penalty": 1.5},
              {"status": 400, "checks": ["error_param:length_penalty"]}),
        _case("chat.reject.n2", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "n": 2},
              {"status": 400, "checks": ["error_param:n"]}),
        _case("chat.reject.priority", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "priority": 1},
              {"status": 400, "checks": ["error_param:priority"]}),
        _case("chat.reject.logprobs", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "logprobs": True},
              {"status": 400, "checks": ["error_param:logprobs"]}),
        _case("chat.reject.top_logprobs", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "top_logprobs": 3},
              {"status": 400, "checks": ["error_param:top_logprobs"]}),
        _case("chat.reject.min_tokens_over", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "min_tokens": 9},
              {"status": 400, "checks": ["error_param:min_tokens"]}),
        _case("chat.reject.truncate_zero", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "truncate_prompt_tokens": 0},
              {"status": 400, "checks": ["error_param:truncate_prompt_tokens"]}),
        _case("chat.reject.stop_empty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "stop": ""},
              {"status": 400, "checks": ["error_param:stop"]}),
        _case("chat.reject.stop_token_neg", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "stop_token_ids": [-1]},
              {"status": 400, "checks": ["error_param:stop_token_ids"]}),
        _case("chat.reject.stream_options_no_stream", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "stream": False, "stream_options": {"include_usage": True}},
              {"status": 400, "checks": ["error_param:stream_options"]}),
        _case("chat.reject.max_tokens_zero", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 0},
              {"status": 400, "checks": ["error_param:max_tokens"]}),
        _case("chat.reject.max_tokens_neg", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": -5},
              {"status": 400, "checks": ["error_param:max_tokens"]}),
        _case("chat.reject.max_conflict", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"],
               "max_tokens": 16, "max_completion_tokens": 32},
              {"status": 400, "checks": ["error_param:max_completion_tokens"]}),
        _case("chat.reject.no_messages", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": [], "max_tokens": 8},
              {"status": 400, "checks": ["error_param:messages"]}),
        _case("chat.reject.continue_and_prompt", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "continue_final_message": True, "add_generation_prompt": True},
              {"status": 400, "checks": ["error_param:continue_final_message"]}),
        _case("chat.reject.bad_model", "chat-invalid", "/v1/chat/completions",
              {"model": "definitely-not-a-model", "messages": CHAT["messages"], "max_tokens": 8},
              {"status": 400, "checks": ["error_param:model"]}),

        # ---- chat: invalid tool settings ----------------------------------
        _case("chat.reject.tools_empty", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8, "tools": []},
              {"status": 400, "checks": ["error_param:tools"]}),
        _case("chat.reject.tool_choice_without_tools", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "tool_choice": "auto"},
              {"status": 400, "checks": ["error_param:tool_choice"]}),
        _case("chat.reject.required_without_tools", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "tool_choice": "required"},
              {"status": 400, "checks": ["error_param:tool_choice"]}),
        _case("chat.reject.named_tool_missing", "chat-invalid", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "tools": [{"type": "function", "function": {
                   "name": "first", "parameters": {"type": "object"},
               }}], "tool_choice": {"type": "function", "function": {
                   "name": "second"}}},
              {"status": 400, "checks": ["error_param:tool_choice"]}),

        # ---- chat: unknown fields rejected by schema (422) ---------------
        _case("chat.unknown.response_format", "chat-unknown", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "response_format": {"type": "json_object"}},
              {"status": 422}),
        _case("chat.unknown.chat_template_kwargs", "chat-unknown", "/v1/chat/completions",
              {"model": "MODEL", "messages": CHAT["messages"], "max_tokens": 8,
               "chat_template_kwargs": {"enable_thinking": False}},
              {"status": 422}),

        # ---- legacy completions ------------------------------------------
        _case("comp.basic", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "The capital of France is", "max_tokens": 8},
              {"status": 200, "checks": ["has_text", "usage", "finish_reason"]}),
        _case("comp.prompt_ids", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": [1, 2, 3, 4], "max_tokens": 8},
              {"status": 200, "checks": ["has_text", "usage"]}),
        _case("comp.echo", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "The capital of France is", "max_tokens": 8,
               "echo": True},
              {"status": 200, "checks": ["has_text", "usage", "echo"]}),
        _case("comp.stream", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "The capital of France is", "max_tokens": 8,
               "stream": True},
              {"status": 200, "checks": ["sse"]}),
        _case("comp.reject.empty_prompt", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "", "max_tokens": 8},
              {"status": 400, "checks": ["error_param:prompt"]}),
        _case("comp.reject.neg_token_id", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": [-1], "max_tokens": 8},
              {"status": 400, "checks": ["error_param:prompt"]}),
        _case("comp.reject.temperature", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "hello", "max_tokens": 8, "temperature": 0.5},
              {"status": 400, "checks": ["error_param:temperature"]}),
        _case("comp.reject.logprobs", "completion", "/v1/completions",
              {"model": "MODEL", "prompt": "hello", "max_tokens": 8, "logprobs": 1},
              {"status": 400, "checks": ["error_param:logprobs"]}),

        # Read-only checks for the distributed runtime contract.
        _case("runtime.snapshot", "runtime", "/v1/aegaeon/runtime", None,
              {"status": 200, "checks": ["runtime"]}, method="GET"),

        # ---- tokenizer helpers -------------------------------------------
        _case("tokenize.ok", "tokenize", "/tokenize",
              {"model": "MODEL", "prompt": "hello world"},
              {"status": 200, "checks": ["tokenize"]}),
        _case("tokenize.bad_model", "tokenize", "/tokenize",
              {"model": "definitely-not-a-model", "prompt": "hi"},
              {"status": 400}),
        _case("detokenize.ok", "detokenize", "/detokenize",
              {"model": "MODEL", "tokens": [1, 2, 3]},
              {"status": 200, "checks": ["detokenize"]}),
    ]

    for case in cases:
        if case["payload"]:
            case["payload"] = with_model(case["payload"])
        if case["endpoint"].startswith("/v1/models/{model}"):
            case["endpoint"] = "/v1/models/" + m
    return cases


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def http_request(method, url, payload=None, timeout=120):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, raw
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        raise ConnectionError(f"{method} {url}: {exc}")


def _detail_param(raw):
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    detail = data.get("detail")
    if isinstance(detail, dict):
        return detail.get("param")
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            return first.get("loc", ["?", "?"])[-1] if first.get("loc") else None
    return None


def _json_body(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _operation_succeeded(status, raw):
    data = _json_body(raw)
    return (status == 200 and isinstance(data, dict)
            and data.get("status") == "ok" and not data.get("failed"))


# --------------------------------------------------------------------------
# Check functions. Each returns (ok, message).
# --------------------------------------------------------------------------

def check_has_text(checks, data, raw, echo=False):
    if not data:
        return False, "response is not JSON"
    choices = data.get("choices") or []
    if not choices:
        return False, "no choices"
    choice = choices[0]
    text = None
    if isinstance(choice.get("message"), dict):
        text = choice["message"].get("content")
    elif "text" in choice:
        text = choice["text"]
    if not text:
        return False, "empty content/text"
    return True, ""


def check_usage(checks, data, raw):
    usage = (data or {}).get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(key), int):
            return False, f"usage.{key} missing"
    return True, ""


def check_finish_reason(checks, data, raw):
    choices = (data or {}).get("choices") or []
    if not choices:
        return False, "no choices"
    reason = choices[0].get("finish_reason")
    if reason not in ("stop", "length", "tool_calls"):
        return False, f"finish_reason={reason!r}"
    return True, ""


def check_token_ids(checks, data, raw):
    choices = (data or {}).get("choices") or []
    if not choices:
        return False, "no choices"
    choice = choices[0]
    if "token_ids" not in choice:
        return False, "choices[0].token_ids missing"
    return True, ""


def check_tool_calls(checks, data, raw):
    choices = (data or {}).get("choices") or []
    if not choices:
        return False, "no choices"
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        return False, f"finish_reason={choice.get('finish_reason')!r}"
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []
    if not tool_calls:
        return False, "message.tool_calls is empty"
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        if not tool_call.get("id") or tool_call.get("type") != "function":
            return False, f"tool_calls[{index}] has invalid id/type"
        if not function.get("name"):
            return False, f"tool_calls[{index}].function.name is empty"
        try:
            json.loads(function.get("arguments", ""))
        except (TypeError, ValueError):
            return False, f"tool_calls[{index}].function.arguments is not JSON"
    return True, f"{len(tool_calls)} call(s)"


def check_sse(checks, data, raw):
    if not raw or "data:" not in raw:
        return False, "not an SSE response"
    if "[DONE]" not in raw:
        return False, "missing [DONE]"
    chunks = [ln for ln in raw.splitlines() if ln.startswith("data:") and ln != "data: [DONE]"]
    content_chunks = 0
    for ln in chunks:
        event = _json_body(ln[len("data:"):].strip())
        if not event:
            continue
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta_content = (choice.get("delta") or {}).get("content")
        # chat streams use delta.content; legacy completions use choice.text
        if delta_content or choice.get("text"):
            content_chunks += 1
    if content_chunks == 0:
        return False, "no content delta chunks"
    return True, f"{len(chunks)} chunks"


def check_sse_usage(checks, data, raw):
    for ln in raw.splitlines():
        if ln.startswith("data:"):
            event = _json_body(ln[len("data:"):].strip())
            if event and event.get("usage"):
                usage = event["usage"]
                if all(isinstance(usage.get(k), int) for k in
                       ("prompt_tokens", "completion_tokens", "total_tokens")):
                    return True, ""
    return False, "no usage chunk in stream"


def check_sse_tool_calls(checks, data, raw):
    calls = {}
    finish_reason = None
    for line in raw.splitlines():
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        event = _json_body(line[len("data:"):].strip())
        choices = (event or {}).get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        for delta in (choice.get("delta") or {}).get("tool_calls") or []:
            index = delta.get("index")
            if not isinstance(index, int):
                return False, "tool-call delta is missing an integer index"
            call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            call["id"] += delta.get("id") or ""
            function = delta.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
    if finish_reason != "tool_calls" or not calls:
        return False, f"finish_reason={finish_reason!r}, calls={len(calls)}"
    for index, call in calls.items():
        if not call["id"] or not call["name"]:
            return False, f"tool call {index} has empty id/name"
        try:
            json.loads(call["arguments"])
        except ValueError:
            return False, f"tool call {index} arguments are not JSON"
    return True, f"{len(calls)} streamed call(s)"


def check_echo(checks, data, raw):
    if not data:
        return False, "response is not JSON"
    choices = data.get("choices") or []
    if not choices:
        return False, "no choices"
    text = choices[0].get("text") or ""
    if not text.strip():
        return False, "empty text"
    return True, ""


def check_request_id(checks, data, raw, expected):
    rid = (data or {}).get("id") or ""
    if rid != expected:
        return False, f"id={rid!r} != {expected!r}"
    return True, ""


def check_error_param(checks, data, raw, param):
    got = _detail_param(raw)
    if got != param:
        return False, f"detail.param={got!r} != {param!r}"
    return True, ""


def check_model_in_list(checks, data, raw, models):
    if not data:
        return False, "response is not JSON"
    listed = [item.get("id") for item in data.get("data") or []]
    missing = [m for m in models if m not in listed]
    if missing:
        return False, f"models not listed: {missing}"
    return True, ""


def check_tokenize(checks, data, raw):
    if not data:
        return False, "response is not JSON"
    if "count" not in data or "tokens" not in data:
        return False, "missing count/tokens"
    return True, ""


def check_detokenize(checks, data, raw):
    if not data:
        return False, "response is not JSON"
    if "prompt" not in data:
        return False, "missing prompt"
    return True, ""


def check_runtime(checks, data, raw):
    if not isinstance(data, dict):
        return False, "response is not a JSON object"
    for key, expected in (("engines", list), ("nodes", list),
                          ("model_placements", dict), ("request_routing", dict),
                          ("decode_load", list), ("work_stealing", list)):
        if not isinstance(data.get(key), expected):
            return False, f"runtime.{key} missing or invalid"
    return True, ""


CHECK_DISPATCH = {
    "runtime": check_runtime,
    "has_text": lambda c, d, r: check_has_text(c, d, r),
    "usage": lambda c, d, r: check_usage(c, d, r),
    "finish_reason": lambda c, d, r: check_finish_reason(c, d, r),
    "token_ids": lambda c, d, r: check_token_ids(c, d, r),
    "tool_calls": lambda c, d, r: check_tool_calls(c, d, r),
    "sse": lambda c, d, r: check_sse(c, d, r),
    "sse_usage": lambda c, d, r: check_sse_usage(c, d, r),
    "sse_tool_calls": lambda c, d, r: check_sse_tool_calls(c, d, r),
    "echo": lambda c, d, r: check_echo(c, d, r),
    "tokenize": lambda c, d, r: check_tokenize(c, d, r),
    "detokenize": lambda c, d, r: check_detokenize(c, d, r),
}


def run_case(case, base_url, models, timeout):
    endpoint = case["endpoint"]
    if case["payload"] is None and case["method"] == "GET":
        url = base_url + endpoint
        try:
            status, raw = http_request("GET", url, timeout=timeout)
        except ConnectionError as exc:
            return "FAIL", None, f"connection: {exc}"
    else:
        url = base_url + endpoint
        try:
            status, raw = http_request(case["method"], url, case["payload"], timeout=timeout)
        except ConnectionError as exc:
            return "FAIL", None, f"connection: {exc}"

    data = _json_body(raw)
    expect = case["expect"]
    problems = []
    if status != expect["status"]:
        problems.append(f"status={status} != {expect['status']}")

    for check in expect.get("checks", []):
        if check.startswith("min_tokens:"):
            minimum = int(check.split(":", 1)[1])
            count = ((data or {}).get("usage") or {}).get("completion_tokens")
            ok = isinstance(count, int) and count >= minimum
            msg = f"completion_tokens={count!r}, expected at least {minimum}"
        elif check.startswith("error_param:"):
            param = check.split(":", 1)[1]
            ok, msg = check_error_param(None, data, raw, param)
        elif check.startswith("request_id:"):
            expected = check.split(":", 1)[1]
            ok, msg = check_request_id(None, data, raw, expected)
        elif check == "model_in_list":
            ok, msg = check_model_in_list(None, data, raw, models)
        elif check in CHECK_DISPATCH:
            ok, msg = CHECK_DISPATCH[check](None, data, raw)
        else:
            ok, msg = False, f"unknown check {check!r}"
        if not ok:
            problems.append(f"{check}: {msg}")

    if problems:
        snippet = raw[:400] if raw else ""
        return "FAIL", None, "; ".join(problems) + f" | body: {snippet}"
    return "PASS", data, ""


def run_matrix(args):
    models = [m for m in args.models.split(",") if m.strip()] if args.models else []
    if not models:
        print("warning: --models empty; model-dependent cases will use 'MODEL' and fail")
    matrix = build_matrix(models)
    if args.filter:
        matrix = [c for c in matrix if c["group"] == args.filter or c["id"] == args.filter]
    if args.list_cases:
        for c in matrix:
            print(f"{c['id']:42s} {c['group']:14s} {c['method']:4s} {c['endpoint']} "
                  f"-> {c['expect']['status']}")
        return 0

    base_url = f"http://{args.host}:{args.port}"
    results = []
    summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    for case in matrix:
        if case["skip"]:
            summary["skipped"] += 1
            results.append({**case, "result": "SKIP", "detail": ""})
            continue
        print(f"[..] {case['id']} ...", end="", flush=True)
        start = time.time()
        outcome, data, detail = run_case(case, base_url, models, args.timeout)
        elapsed = time.time() - start
        summary["total"] += 1
        if outcome == "PASS":
            summary["passed"] += 1
        else:
            summary["failed"] += 1
        results.append({**case, "result": outcome, "detail": detail,
                        "elapsed": round(elapsed, 2)})
        if outcome == "PASS":
            print(f" PASS ({elapsed:.1f}s)")
        else:
            print(f" {outcome} ({elapsed:.1f}s): {detail}")

    report = {"summary": summary, "models": models, "results": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {args.output}")
    print(f"\nsummary: {summary}")
    return 0 if summary["failed"] == 0 else 1


# --------------------------------------------------------------------------
# Remote (SSH) orchestration
# --------------------------------------------------------------------------

def ssh(host, command, timeout=600, check=True):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"ssh {host} failed (rc={proc.returncode}):\n"
            f"cmd: {command}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    return proc.stdout


def scp_put(host, src, dst, timeout=120):
    subprocess.run(["scp", "-o", "BatchMode=yes", src, f"{host}:{dst}"],
                   capture_output=True, text=True, timeout=timeout, check=True)


# --- server-side subcommands (run on the remote host) ----------------------

def _env_py(args):
    return args.py or sys.executable


def _ray_bin(args):
    """The ``ray`` CLI executable next to the remote python interpreter."""
    py = _env_py(args)
    return os.path.join(os.path.dirname(py), "ray")


def server_side_check_config(args):
    """Validate an Aegaeon server YAML without starting any GPU service.

    Replicates the deployment-parameter rules from ``api.create_service``:
    nnodes >= 1, tensor_parallel_size >= 1, engine-mode mutual exclusion,
    and the requirement that at least one engine is configured.
    """
    if args.config:
        cfg_path = args.config
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        cfg_path = _render_server_config(args.remote_dir, models, getattr(args, "engine_mode", "simple"))
    py = _env_py(args)
    code = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from aegaeon.model_registry import ModelSpecRegistry
from aegaeon.config import ServerConfig

cfg = sys.argv[2]
problems = []
registry = ModelSpecRegistry.from_yaml(cfg)
server = ServerConfig.from_yaml(cfg)
if server.nnodes < 1:
    problems.append("server.nnodes must be at least 1")
if server.tensor_parallel_size < 1:
    problems.append("server.tensor_parallel_size must be at least 1")
if server.num_engines > 0:
    if server.num_prefill_engines != 0 or server.num_decode_engines != 0:
        problems.append("server.num_engines (simple mode) is mutually exclusive with "
                        "server.num_prefill_engines/server.num_decode_engines")
elif server.num_prefill_engines + server.num_decode_engines < 1:
    problems.append("Configure at least one simple engine, or at least one "
                    "prefill/decode engine")
for spec in registry:
    if not spec.model_path and spec.path is None:
        problems.append(f"model '{spec.name}' has no path and no params")
print(json.dumps({
    "nnodes": server.nnodes,
    "num_engines": server.num_engines,
    "num_prefill_engines": server.num_prefill_engines,
    "num_decode_engines": server.num_decode_engines,
    "tensor_parallel_size": server.tensor_parallel_size,
    "memory_utilization": server.memory_utilization,
    "cpu_num_slabs": server.cpu_num_slabs,
    "model_cache_size_gb": server.model_cache_size_gb,
    "dummy_weights": server.dummy_weights,
    "models": [s.name for s in registry],
    "problems": problems,
}, indent=2))
sys.exit(1 if problems else 0)
"""
    proc = subprocess.run([py, "-c", code, args.repo, cfg_path],
                          capture_output=True, text=True, timeout=60)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def _deployment_config_cases():
    """Deployment-parameter matrix (config-level, no GPU service needed).

    Each case is a full ``models``/``server`` dict plus the expected outcome.
    """
    model = {
        "name": "Qwen/Qwen3-4B",
        "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
        "params": MODEL_PARAMS["Qwen/Qwen3-4B"],
    }
    base_server = {
        "cpu_num_slabs": 1,
        "cpu_slab_size_bytes": 1073741824,
        "memory_utilization": 0.75,
        "worker_num_cpus": 4,
        "nnodes": 1,
        "num_engines": 2,
        "num_prefill_engines": 0,
        "num_decode_engines": 0,
        "tensor_parallel_size": 1,
        "model_cache_size_gb": 1,
        "dummy_weights": True,
    }

    def case(cid, server_overrides=None, models=None, expect_ok=True):
        server = dict(base_server)
        if server_overrides:
            server.update(server_overrides)
        return {
            "id": cid,
            "yaml": {"server": server, "models": [model] if models is None else models},
            "expect_ok": expect_ok,
        }

    return [
        case("deploycfg.valid_simple"),
        case("deploycfg.valid_pd", {"num_engines": 0, "num_prefill_engines": 1,
                                    "num_decode_engines": 1}),
        case("deploycfg.valid_pd_prefill_only", {"num_engines": 0,
                                                 "num_prefill_engines": 1,
                                                 "num_decode_engines": 0}),
        case("deploycfg.valid_tp2", {"tensor_parallel_size": 2, "num_engines": 1}),
        case("deploycfg.valid_memory_util", {"memory_utilization": 0.9}),
        case("deploycfg.valid_cpu_cache", {"cpu_num_slabs": 4}),
        case("deploycfg.invalid_nnodes_zero", {"nnodes": 0}, expect_ok=False),
        case("deploycfg.invalid_nnodes_neg", {"nnodes": -2}, expect_ok=False),
        case("deploycfg.invalid_tp_zero", {"tensor_parallel_size": 0}, expect_ok=False),
        case("deploycfg.invalid_tp_neg", {"tensor_parallel_size": -1}, expect_ok=False),
        case("deploycfg.invalid_engine_neg", {"num_engines": -1,
                                              "num_prefill_engines": 1},
             expect_ok=False),
        case("deploycfg.invalid_mutex_simple_pd",
             {"num_engines": 1, "num_prefill_engines": 1, "num_decode_engines": 0},
             expect_ok=False),
        case("deploycfg.invalid_no_engine",
             {"num_engines": 0, "num_prefill_engines": 0, "num_decode_engines": 0},
             expect_ok=False),
        case("deploycfg.invalid_memory_util_zero", {"memory_utilization": 0},
             expect_ok=False),
        case("deploycfg.invalid_memory_util_over", {"memory_utilization": 1.5},
             expect_ok=False),
        case("deploycfg.valid_max_model_len", models=[{
            "name": "Qwen/Qwen3-4B", "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
            "params": 4.1, "max_model_len": 4096,
        }]),
        case("deploycfg.invalid_max_model_len_str", models=[{
            "name": "Qwen/Qwen3-4B", "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
            "max_model_len": "abc",
        }], expect_ok=False),
        case("deploycfg.invalid_params_str", models=[{
            "name": "Qwen/Qwen3-4B", "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
            "params": "many",
        }], expect_ok=False),
        case("deploycfg.invalid_missing_name", models=[{
            "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"], "params": 4.1,
        }], expect_ok=False),
        case("deploycfg.invalid_mixed_ids", models=[
            {"name": "a", "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
             "params": 1.0, "id": 0},
            {"name": "b", "path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"],
             "params": 1.0},
        ], expect_ok=False),
    ]


def server_side_config_matrix(args):
    """Matrix-test deployment parameters at the config level.

    Each case is parsed with the real ``ModelSpecRegistry``/``ServerConfig`` and
    validated against the ``api.create_service`` rules. No GPU service needed.
    """
    py = _env_py(args)
    cases_path = os.path.join(args.remote_dir, "deploy-cases.json")
    out_path = os.path.join(args.remote_dir, "deploy-config-results.json")
    with open(cases_path, "w") as f:
        json.dump(_deployment_config_cases(), f)

    code = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from aegaeon.model_registry import ModelSpecRegistry
from aegaeon.config import ServerConfig

cases = json.load(open(sys.argv[2]))
results = []
for case in cases:
    problems = []
    try:
        registry = ModelSpecRegistry.from_dict(case["yaml"])
        server = ServerConfig.from_dict(case["yaml"])
        if server.nnodes < 1:
            problems.append("nnodes")
        if server.tensor_parallel_size < 1:
            problems.append("tensor_parallel_size")
        if server.num_engines < 0 or server.num_prefill_engines < 0 or \
                server.num_decode_engines < 0:
            problems.append("negative-engine")
        if server.memory_utilization <= 0 or server.memory_utilization > 1:
            problems.append("memory_utilization")
        if server.num_engines > 0:
            if server.num_prefill_engines != 0 or server.num_decode_engines != 0:
                problems.append("mutually-exclusive")
        elif server.num_prefill_engines + server.num_decode_engines < 1:
            problems.append("no-engine")
    except Exception as exc:
        problems.append(type(exc).__name__ + ": " + str(exc)[:100])
    ok = not problems
    results.append({
        "id": case["id"],
        "expect_ok": case["expect_ok"],
        "actual_ok": ok,
        "problems": problems,
        "pass": ok == case["expect_ok"],
    })
with open(sys.argv[3], "w") as f:
    json.dump(results, f)
"""
    proc = subprocess.run([py, "-c", code, args.repo, cases_path, out_path],
                          capture_output=True, text=True, timeout=120)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if not os.path.isfile(out_path):
        print("config-matrix: no result file produced")
        return 1
    with open(out_path) as f:
        results = json.load(f)
    passed = sum(1 for r in results if r["pass"])
    print(f"\n== deployment config matrix: {passed}/{len(results)} ==")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        detail = "" if r["pass"] else (
            f"expect_ok={r['expect_ok']} problems={r['problems']}"
        )
        print(f"{status:4s} {r['id']:42s} {detail}")
    return 0 if passed == len(results) else 1


def server_side_deploy_matrix(args):
    """Matrix-test the /v1/models/deploy & /v1/models/undeploy API behavior."""
    port = args.port
    model = "Qwen/Qwen3-4B"
    model_path = DEFAULT_MODEL_PATHS[model]
    # Use a DIFFERENT model path for the temp model: undeploy uncaches by
    # model_path, so reusing the primary model's path would evict its cached
    # weights/tokenizer and break subsequent requests.
    tmp_model = "paramtest-mml-override"
    tmp_path = DEFAULT_MODEL_PATHS["Qwen/Qwen3-0.6B"]

    def post(payload):
        return http_request("POST", f"http://127.0.0.1:{port}/v1/models/deploy",
                            payload, timeout=args.timeout)

    def get(path):
        return http_request("GET", f"http://127.0.0.1:{port}{path}", timeout=30)

    def undeploy(payload):
        return http_request("POST", f"http://127.0.0.1:{port}/v1/models/undeploy",
                            payload, timeout=args.timeout)

    def run(cid, fn, expect, check_param=None):
        try:
            status, raw = fn()
        except ConnectionError as exc:
            print(f"FAIL {cid}: connection {exc}")
            return False
        ok = status == expect
        detail = f"status={status} != {expect}"
        if ok and expect == 200 and cid.startswith(("deploy.", "undeploy.")):
            ok = _operation_succeeded(status, raw)
            if not ok:
                detail = "deployment operation did not fully succeed"
        if ok and check_param:
            param = _detail_param(raw)
            if param != check_param:
                ok = False
                detail = f"detail.param={param!r} != {check_param!r}"
        if ok:
            print(f"PASS {cid}")
            return True
        print(f"FAIL {cid}: {detail} | {raw[:200]}")
        return False

    ok = []
    ok.append(run("deploy.invalid_replica_count",
                  lambda: post({"model": model, "replica_count": 0}),
                  422, check_param="replica_count"))
    ok.append(run("deploy.empty_nodes",
                  lambda: post({"model": model, "model_path": model_path, "node_ids": []}),
                  400, check_param="node_ids"))
    ok.append(run("deploy.invalid_path",
                  lambda: post({"model": "paramtest-no", "model_path": "/nonexistent/x"}),
                  400, check_param="model"))
    ok.append(run("deploy.duplicate",
                  lambda: post({"model": model, "model_path": model_path}),
                  200))
    ok.append(run("deploy.max_model_len_override",
                  lambda: post({"model": tmp_model, "model_path": tmp_path,
                                "max_model_len": 4096}),
                  200))
    ok.append(run("models.get_after_deploy",
                  lambda: get(f"/v1/models/{tmp_model}"),
                  200))
    ok.append(run("undeploy.not_deployed",
                  lambda: undeploy({"model": "paramtest-nope"}),
                  400))
    ok.append(run("undeploy.roundtrip",
                  lambda: undeploy({"model": tmp_model}),
                  200))
    ok.append(run("undeploy.gone",
                  lambda: get(f"/v1/models/{tmp_model}"),
                  400))

    passed = sum(1 for item in ok if item)
    print(f"\n== deploy/undeploy matrix: {passed}/{len(ok)} ==")
    return 0 if passed == len(ok) else 1


def _render_rate_config(remote_dir):
    """A 1-engine server config for the rate-limit test service."""
    cfg_path = os.path.join(remote_dir, "rate-config.yaml")
    cfg = {
        "server": {
            "cpu_num_slabs": 1,
            "cpu_slab_size_bytes": 1073741824,
            "memory_utilization": 0.75,
            "worker_num_cpus": 4,
            "nnodes": 1,
            "num_engines": 1,
            "num_prefill_engines": 0,
            "num_decode_engines": 0,
            "tensor_parallel_size": 1,
            "model_cache_size_gb": 1,
            "dummy_weights": True,
        },
        "models": [],
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


def server_side_rate_matrix(args):
    """End-to-end test of the model access-rate tracking & rate limiting.

    Spins up a dedicated rate-limited service (port 8001, 1 engine, dummy
    weights) on the shared ray, runs the assertions, then tears it down.

    Limits used: AEGAEON_MAX_REQUESTS_PER_SECOND=0.05 (3 req / 60s window),
    AEGAEON_MAX_TOKENS_PER_SECOND=0.05 (3 tok / 60s window).
    """
    port = 8001
    ray_port = args.ray_port
    py = _env_py(args)
    shm_dir = f"/dev/shm/aegaeon_param_test_{os.environ.get('USER', 'user')}"
    os.makedirs(shm_dir, exist_ok=True)
    os.chmod(shm_dir, 0o700)
    ray_temp_dir = os.path.join(args.remote_dir, "ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)

    # 0. ensure ray is up
    if not _tcp_up("127.0.0.1", ray_port):
        subprocess.run([_ray_bin(args), "start", "--head", "--port", str(ray_port),
                        "--num-cpus", str(os.cpu_count() or 8),
                        "--resources", '{"node_0": 4}', "--temp-dir", ray_temp_dir],
                       check=True, timeout=180)
    ray_addr = f"127.0.0.1:{ray_port}"

    # 1. launch the rate-limited service
    cfg_path = _render_rate_config(args.remote_dir)
    log_path = os.path.join(args.remote_dir, "rate-server.log")
    log_file = open(log_path, "w")
    env = dict(os.environ)
    env.update({
        "AEGAEON_RAY_ADDRESS": ray_addr,
        "AEGAEON_RAY_TEMP_DIR": ray_temp_dir,
        "AEGAEON_SHM_DIR": shm_dir,
        "AEGAEON_MAX_REQUESTS_PER_SECOND": "0.05",
        "AEGAEON_MAX_TOKENS_PER_SECOND": "0.05",
    })
    subprocess.Popen([py, "-m", "aegaeon", "start", "--config", cfg_path,
                      "--host", "0.0.0.0", "--port", str(port)],
                     cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                     stdout=log_file, stderr=subprocess.STDOUT,
                     start_new_session=True)
    deadline = time.time() + 300
    healthy = False
    while time.time() < deadline:
        try:
            status, _ = http_request("GET", f"http://127.0.0.1:{port}/health", timeout=3)
            if status == 200:
                healthy = True
                break
        except ConnectionError:
            pass
        time.sleep(2)
    if not healthy:
        print("rate service failed to start")
        try:
            print("".join(open(log_path).readlines()[-20:]))
        except OSError:
            pass
        return 1

    base = f"http://127.0.0.1:{port}"
    results = []

    def run(cid, fn, expect, check=None):
        try:
            status, raw = fn()
        except ConnectionError as exc:
            results.append((cid, "FAIL", f"connection {exc}"))
            return
        ok = status == expect
        detail = f"status={status} != {expect}"
        if ok and check is not None:
            detail = ""
            for key, expected in check.items():
                data = _json_body(raw)
                got = data
                for part in key.split("."):
                    got = (got or {}).get(part) if isinstance(got, dict) else None
                if got != expected:
                    ok = False
                    detail = f"{key}={got!r} != {expected!r}"
        results.append((cid, "PASS" if ok else "FAIL", "" if ok else detail))

    try:
        # deploy the model on the rate service
        status, raw = http_request(
            "POST", f"{base}/v1/models/deploy",
            {"model": "Qwen/Qwen3-4B", "model_path": DEFAULT_MODEL_PATHS["Qwen/Qwen3-4B"]},
            timeout=args.timeout)
        if status != 200:
            print("rate service deploy failed:", raw[:200])
            return 1

        def chat(max_tokens=2, delay=0):
            if delay:
                time.sleep(delay)
            return http_request("POST", f"{base}/v1/chat/completions",
                                {"model": "Qwen/Qwen3-4B",
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": max_tokens},
                                timeout=args.timeout)

        # metrics endpoint shape
        run("rate.metrics_shape", lambda: http_request("GET", f"{base}/v1/aegaeon/metrics", timeout=30), 200,
            check={"window_seconds": 60,
                   "limits.max_requests_per_second": 0.05,
                   "limits.max_tokens_per_second": 0.05})
        # request-rate limiting: 3 allowed, 4th rejected.
        # max_tokens=1 keeps cumulative generated tokens at 3/60s = 0.05 so the
        # token limit does not trip before the request limit does.
        run("rate.req_ok_1", lambda: chat(max_tokens=1), 200)
        run("rate.req_ok_2", lambda: chat(max_tokens=1), 200)
        run("rate.req_ok_3", lambda: chat(max_tokens=1), 200)
        run("rate.req_burst_429", lambda: chat(max_tokens=1), 429)

        # window resets after 60s
        run("rate.req_recover", lambda: chat(delay=61), 200)

        # token-rate limiting: 2 tokens allowed, cumulative 4 rejected
        run("rate.token_ok", lambda: chat(max_tokens=2), 200)
        run("rate.token_429", lambda: chat(max_tokens=2), 429)
        run("rate.token_recover", lambda: chat(max_tokens=2, delay=61), 200)
    finally:
        # tear down only this service (port 8001); keep ray for the main server
        subprocess.run(["pkill", "-f", f"[a]egaeon start --config .*--port {port}"],
                       capture_output=True)
        time.sleep(2)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    print(f"\n== rate-limit matrix: {passed}/{len(results)} ==")
    for cid, status, detail in results:
        print(f"{status:4s} {cid:22s} {detail}")
    return 0 if passed == len(results) else 1


def server_side_status(args):
    health = "no"
    try:
        status, raw = http_request("GET", f"http://127.0.0.1:{args.port}/health", timeout=3)
        health = f"yes (status={status})"
    except ConnectionError:
        pass
    print(f"server(127.0.0.1:{args.port}): {health}")
    return 0


def _render_server_config(remote_dir, models, engine_mode="simple"):
    """Write and return the path of a test server YAML (always fresh).

    ``models`` stays empty: with dummy_weights=true the static pre-cache path
    would try to allocate the real model size in the (1GB) model cache, which
    fails.  Models are instead added at runtime via ``POST /v1/models/deploy``
    (the controller's dummy branch).

    ``engine_mode``: "simple" (num_engines=2) or "pd" (1 prefill + 1 decode).
    """
    cfg_path = os.path.join(remote_dir, "test-config.yaml")
    if engine_mode == "pd":
        num_engines, num_prefill, num_decode = 0, 1, 1
    else:
        num_engines, num_prefill, num_decode = 2, 0, 0
    # No static model entries: see docstring (dummy deploy happens at runtime).
    entries = []
    cfg = {
        "server": {
            # dummy_weights=true: parameter-conformance tests only exercise the
            # API/control plane, so no real weights are needed.
            # cpu_num_slabs=1 and model_cache_size_gb=1: cudaHostRegister on
            # /dev/shm fails above ~1-2 GB when the shell's RLIMIT_MEMLOCK is
            # small (verified on gpu-server: 1GB OK, 2GB invalid argument).
            # Both the KV cache and the QuickCache model cache are pinned.
            "cpu_num_slabs": 1,
            "cpu_slab_size_bytes": 1073741824,
            "memory_utilization": 0.75,
            "worker_num_cpus": 4,
            "nnodes": 1,
            "num_engines": num_engines,
            "num_prefill_engines": num_prefill,
            "num_decode_engines": num_decode,
            "tensor_parallel_size": 1,
            "model_cache_size_gb": 1,
            "dummy_weights": True,
        },
        "models": entries,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote server config: {cfg_path}")
    return cfg_path


def _tcp_up(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def server_side_start(args):
    py = _env_py(args)
    # 1. already running?
    try:
        status, _ = http_request("GET", f"http://127.0.0.1:{args.port}/health", timeout=3)
        if status == 200:
            print(f"server already running on :{args.port}, skip start")
            return 0
    except ConnectionError:
        pass

    # 2. generate + validate the server config (deployment parameters)
    if args.config:
        cfg_path = args.config
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        cfg_path = _render_server_config(args.remote_dir, models, getattr(args, "engine_mode", "simple"))
    if server_side_check_config(args) != 0:
        print("check-config FAILED; aborting start")
        return 1

    # 3. ray up? (use our own ray-port + temp-dir so we never touch a
    #    colleague's ray or the shared /tmp/ray state)
    ray_addr = f"127.0.0.1:{args.ray_port}"
    ray_temp_dir = os.path.join(args.remote_dir, "ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    ray_bin = _ray_bin(args)
    if _tcp_up("127.0.0.1", args.ray_port):
        print(f"ray already up at {ray_addr}")
    else:
        print(f"starting ray head on {ray_addr} (temp-dir {ray_temp_dir}) ...")
        subprocess.run(
            [ray_bin, "start", "--head", "--port", str(args.ray_port),
             "--num-cpus", str(os.cpu_count() or 8),
             "--resources", '{"node_0": 4}',
             "--temp-dir", ray_temp_dir],
            check=True, timeout=180,
        )
        if not _tcp_up("127.0.0.1", args.ray_port, timeout=10):
            print("ray failed to listen; aborting")
            return 1

    # 4. launch the API server
    #    CPU/model caches live in a private /dev/shm directory (tmpfs is
    #    required by cudaHostRegister; the shared /dev/shm/aegaeon_* files are
    #    owned by whoever created them first).
    shm_dir = f"/dev/shm/aegaeon_param_test_{os.environ.get('USER', 'user')}"
    os.makedirs(shm_dir, exist_ok=True)
    os.chmod(shm_dir, 0o700)
    log_path = os.path.join(args.remote_dir, "server.log")
    log_file = open(log_path, "w")
    env = dict(os.environ)
    env.update({
        "AEGAEON_RAY_ADDRESS": ray_addr,
        "AEGAEON_RAY_TEMP_DIR": ray_temp_dir,
        "AEGAEON_SHM_DIR": shm_dir,
    })
    proc = subprocess.Popen(
        [py, "-m", "aegaeon", "start", "--config", cfg_path,
         "--host", "0.0.0.0", "--port", str(args.port)],
        cwd=args.repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"server launched (pid={proc.pid}), waiting for /health ...")

    deadline = time.time() + args.start_timeout
    while time.time() < deadline:
        try:
            status, _ = http_request("GET", f"http://127.0.0.1:{args.port}/health", timeout=3)
            if status == 200:
                print("server is up")
                return 0
        except ConnectionError:
            pass
        time.sleep(2)
    print("server failed to become healthy. last log lines:")
    try:
        with open(log_path) as f:
            print("".join(f.readlines()[-40:]))
    except OSError:
        pass
    return 1


def server_side_stop(args):
    # ray 2.58's `ray stop` kills every local ray process, which would also
    # terminate a colleague's cluster. Match only our own processes by the
    # private ray temp-dir / shm dir instead. The [x] trick prevents pkill
    # from matching this very command line over ssh.
    subprocess.run(["pkill", "-f", "[a]egaeon start --config"], capture_output=True)
    subprocess.run(["pkill", "-f", os.path.join(args.remote_dir, "[r]ay_tmp")],
                   capture_output=True)
    subprocess.run(["pkill", "-f", "[a]egaeon_param_test"], capture_output=True)
    time.sleep(2)
    shm_dir = f"/dev/shm/aegaeon_param_test_{os.environ.get('USER', 'user')}"
    subprocess.run(["rm", "-rf", shm_dir], capture_output=True)
    print("stopped aegaeon server and ray")
    return 0


def server_side_deploy(args):
    """POST /v1/models/deploy for each --models entry (dummy branch)."""
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_path = DEFAULT_MODEL_PATHS.get(name)
        if not model_path:
            print(f"deploy {name}: unknown model path, skipped")
            return 1
        payload = {"model": name, "model_path": model_path}
        status, raw = http_request(
            "POST", f"http://127.0.0.1:{args.port}/v1/models/deploy",
            payload, timeout=args.timeout,
        )
        print(f"deploy {name}: status={status} {raw[:200]}")
        if not _operation_succeeded(status, raw):
            return 1
    return 0


def server_side_test(args):
    return run_matrix(args)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aegaeon parameter matrix tester (local or via SSH)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models", default="Qwen/Qwen3-4B",
                        help="comma-separated model names (also used by remote config)")
    parser.add_argument("--output", default="")
    parser.add_argument("--filter", default="",
                        help="only run cases with this group or id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)

    parser.add_argument("--remote", default="",
                        help="SSH host alias (e.g. gpu-server) to run everything on")
    parser.add_argument("--remote-dir", default="~/param_test",
                        help="remote scratch dir (script + config + logs)")
    parser.add_argument("--py", default="",
                        help="remote python interpreter (default: aegaeon conda env)")
    parser.add_argument("--repo", default="~/aegaeon",
                        help="remote aegaeon repo (used only to launch the server)")
    parser.add_argument("--ray-port", type=int, default=6381,
                        help="dedicated ray head port on the remote host")
    parser.add_argument("--start-timeout", type=int, default=600,
                        help="seconds to wait for /health after launch")
    parser.add_argument("--engine-mode", choices=["simple", "pd"], default="simple",
                        help="engine layout: simple (num_engines=2) or pd (1 prefill + 1 decode)")
    parser.add_argument("--no-start", action="store_true",
                        help="do not start the server remotely; assume it runs")
    parser.add_argument("--stop", action="store_true",
                        help="stop the remote server + ray after the tests")
    parser.add_argument("--server-side", choices=["status", "start", "stop", "test", "check-config", "deploy", "config-matrix", "deploy-matrix", "rate-matrix"],
                        default="", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="",
                        help="path to an existing Aegaeon server YAML (remote mode)")
    args = parser.parse_args()

    if args.server_side:
        # `~/...` defaults are expanded on the remote host, not locally.
        args.remote_dir = os.path.expanduser(args.remote_dir)
        args.repo = os.path.expanduser(args.repo)
        if args.server_side == "status":
            return server_side_status(args)
        if args.server_side == "check-config":
            return server_side_check_config(args)
        if args.server_side == "config-matrix":
            return server_side_config_matrix(args)
        if args.server_side == "deploy-matrix":
            return server_side_deploy_matrix(args)
        if args.server_side == "rate-matrix":
            return server_side_rate_matrix(args)
        if args.server_side == "start":
            return server_side_start(args)
        if args.server_side == "deploy":
            return server_side_deploy(args)
        if args.server_side == "stop":
            return server_side_stop(args)
        if args.server_side == "test":
            return server_side_test(args)

    if args.remote:
        return remote_main(args)
    return run_matrix(args)


def remote_main(args):
    host = args.remote
    remote_dir = args.remote_dir  # literal "~/..." is expanded by the remote shell
    repo = args.repo
    remote_py = args.py or "python3"

    print(f"[1/5] ensure remote dir {remote_dir}")
    ssh(host, f"mkdir -p {remote_dir}")

    print(f"[2/5] upload param_test.py")
    scp_put(host, os.path.abspath(__file__), f"{remote_dir}/param_test.py")

    print("[3/5] validate server config (deployment parameters)")
    out = ssh(host,
              f"{remote_py} {remote_dir}/param_test.py --server-side check-config "
              f"--py {remote_py} --repo {repo} --remote-dir {remote_dir} "
              f"--models '{args.models}' --config '{args.config}'")
    if out.strip():
        print(out)

    print("[3b/5] deployment config matrix (20 cases, no GPU needed)")
    out = ssh(host,
              f"{remote_py} {remote_dir}/param_test.py --server-side config-matrix "
              f"--py {remote_py} --repo {repo} --remote-dir {remote_dir}",
              check=False)
    if out.strip():
        print(out)

    print("[3c/5] rate-limit matrix (dedicated rate-limited service)")
    out = ssh(host,
              f"{remote_py} {remote_dir}/param_test.py --server-side rate-matrix "
              f"--py {remote_py} --repo {repo} --remote-dir {remote_dir} "
              f"--ray-port {args.ray_port} --timeout {args.timeout}",
              check=False, timeout=900)
    if out.strip():
        print(out)

    if not args.no_start:
        print(f"[4/5] start server (ray + aegaeon, mode={args.engine_mode}) if not running")
        out = ssh(host,
                  f"{remote_py} {remote_dir}/param_test.py --server-side start "
                  f"--py {remote_py} --repo {repo} --remote-dir {remote_dir} "
                  f"--port {args.port} --ray-port {args.ray_port} "
                  f"--models '{args.models}' --config '{args.config}' "
                  f"--engine-mode {args.engine_mode} "
                  f"--start-timeout {args.start_timeout}",
                  timeout=args.start_timeout + 120)
        if out.strip():
            print(out)
        print("[5/6] deploy models (runtime dummy deploy)")
        out = ssh(host,
                  f"{remote_py} {remote_dir}/param_test.py --server-side deploy "
                  f"--host 127.0.0.1 --port {args.port} --models '{args.models}' "
                  f"--timeout {args.timeout}")
        if out.strip():
            print(out)
        print("[5b/6] deploy/undeploy API matrix")
        out = ssh(host,
                  f"{remote_py} {remote_dir}/param_test.py --server-side deploy-matrix "
                  f"--host 127.0.0.1 --port {args.port} --timeout {args.timeout}",
                  check=False)
        if out.strip():
            print(out)
    else:
        print("[4/5] skip server start (--no-start)")
        print("[5/6] skip deploy (--no-start)")

    print("[6/6] run parameter matrix on remote host")
    report_path = f"{remote_dir}/report.json"
    ssh(host,
        f"{remote_py} {remote_dir}/param_test.py --server-side test "
        f"--host 127.0.0.1 --port {args.port} --models '{args.models}' "
        f"--output {report_path} --timeout {args.timeout}",
        check=False)

    print("fetch report")
    local_report = args.output or "param_test_report.json"
    subprocess.run(["scp", "-o", "BatchMode=yes",
                    f"{host}:{report_path}", local_report],
                   capture_output=True, text=True, timeout=120, check=True)
    with open(local_report) as f:
        report = json.load(f)
    print(json.dumps(report["summary"], indent=2))
    for result in report["results"]:
        if result["result"] != "PASS":
            print(f"{result['result']:4s} {result['id']}: {result.get('detail', '')[:300]}")
    print(f"full report: {local_report}")

    if args.stop:
        print("--stop: stopping remote server + ray")
        ssh(host,
            f"{remote_py} {remote_dir}/param_test.py --server-side stop "
            f"--py {remote_py} --ray-port {args.ray_port}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
