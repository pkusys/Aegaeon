# Online Inference

## OpenAI-style client

OpenAI clients connect through the service's `/v1` URL. The `model` value identifies a deployed model, and generation currently uses greedy decoding:

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='local-placeholder')
response = client.chat.completions.create(
    model='Qwen/Qwen3-4B',
    messages=[{'role': 'user', 'content': 'Explain model switching.'}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

The key satisfies the client constructor; configure authentication at your gateway.

## Streaming

With `stream=True`, text arrives through `chunk.choices[0].delta.content`. Setting `stream_options.include_usage=true` adds a final usage chunk with empty `choices`:

```python
stream = client.chat.completions.create(
    model='Qwen/Qwen3-4B', messages=[{'role': 'user', 'content': 'Hello'}],
    temperature=0, stream=True, stream_options={'include_usage': True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or '', end='', flush=True)
```

Raw SSE responses end with `data: [DONE]`. Completion requests use `/v1/completions` with a string or token-ID list as `prompt`.

## Tool calling

Aegaeon supports OpenAI-style `tools` and `tool_choice`. With tools present, `tool_choice` defaults to `auto`; `none` disables parsing. Tool calling requires a supported vLLM parser, inferred from the model name/path or set explicitly through `tool_parser` in the model's YAML entry or deployment request.

Clients execute the returned `tool_calls` and submit results as `tool` messages linked by `tool_call_id`. With a tool parser active, streaming buffers generated output until completion, then emits the parsed content and tool calls together.

See [Generation Parameters](generation.md) for accepted values and [HTTP API](api.md) for endpoints and schemas.
