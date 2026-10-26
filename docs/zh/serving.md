# 在线推理

## OpenAI 风格客户端

OpenAI 客户端通过服务的 `/v1` 地址连接。`model` 指定已部署模型，当前生成采用贪心解码：

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='local-placeholder')
response = client.chat.completions.create(
    model='Qwen/Qwen3-4B',
    messages=[{'role': 'user', 'content': '解释模型切换。'}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

这里的 key 用于满足客户端构造要求；鉴权由部署侧网关配置。

## 流式输出

设置 `stream=True` 后，文本通过 `chunk.choices[0].delta.content` 返回。`stream_options.include_usage=true` 会在末尾添加 usage chunk，其 `choices` 为空：

```python
stream = client.chat.completions.create(
    model='Qwen/Qwen3-4B', messages=[{'role': 'user', 'content': '你好'}],
    temperature=0, stream=True, stream_options={'include_usage': True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or '', end='', flush=True)
```

原始 SSE 响应以 `data: [DONE]` 结束。Completion 请求使用 `/v1/completions`，`prompt` 接受字符串或 token ID 列表。

## 工具调用

Aegaeon 支持 OpenAI 风格的 `tools` 和 `tool_choice`。传入工具时，`tool_choice` 默认为 `auto`；`none` 关闭解析。工具调用需要受支持的 vLLM 解析器，可根据模型名称或路径推断，也可通过 YAML 模型条目或部署请求中的 `tool_parser` 显式指定。

客户端执行返回的 `tool_calls`，并将结果作为通过 `tool_call_id` 关联的 `tool` 消息提交。启用工具解析器时，流式响应会缓冲生成的输出，直到生成完成，再一起返回解析后的文本和工具调用。

可用取值见[生成参数](generation.md)，端点和请求结构见 [HTTP API](api.md)。
