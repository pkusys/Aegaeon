# 生成参数

## 贪心解码

当前 Worker 为每个请求生成一个贪心续写结果。采样控制参数仅接受下表中的取值，其他值返回 HTTP 400。

| 参数 | 可用取值 |
|---|---|
| `n` | `1` |
| `temperature` | `0` 或 `null` |
| `top_p`、`top_k` | `1` 或 `null` |
| `min_p` | `0` 或 `null` |
| `frequency_penalty`、`presence_penalty` | `0` 或 `null` |
| `repetition_penalty` | `1` 或 `null` |
| `logit_bias` | 空字典或 `null` |
| `use_beam_search` | `false` |
| `length_penalty` | `1` |
| `priority` | `0` |
| `seed` | 任意整数或 `null`；不影响贪心解码 |
| Chat `logprobs` / `top_logprobs` | `false` 或 `null` / `0` 或 `null` |
| Completion `logprobs` | `null` 或省略 |

## 长度与停止

Chat 默认生成 128 个 token，Completion 默认为 16 个。Chat 接受 `max_tokens` 或 `max_completion_tokens`；同时提供时，两者必须一致。输出上限至少为 1，prompt 必须为至少一个输出 token 留出空间，生成的总长度受模型 `max_model_len` 限制。

| 参数 | 含义 |
|---|---|
| `stop` | 在可见文本中匹配的一个或多个停止字符串 |
| `stop_token_ids` | 额外的非负停止 token ID |
| `ignore_eos` | 忽略 EOS；显式停止条件和长度上限仍然生效 |
| `min_tokens` | 接受停止条件前的最少输出 token 数，范围为 0 到输出上限 |
| `include_stop_str_in_output` | 保留匹配的停止字符串 |
| `skip_special_tokens` | 解码时省略特殊 token |
| `return_token_ids` | 在响应中返回 token ID |

`finish_reason` 为 `stop`、`length`，成功解析函数调用时为 `tool_calls`。

## Prompt 格式

`truncate_prompt_tokens` 接受正整数输入 token 上限或 `-1`。对于纯文本输入，`-1` 保持输入不变，截断默认保留末尾。`truncation_side='left'` 保留末尾，`'right'` 保留开头。

Chat 默认 `add_special_tokens=false`、`add_generation_prompt=true`。`continue_final_message=true` 表示续写最后一条消息，此时要求 `add_generation_prompt=false`。

完整请求类型和默认值见运行中服务的 `/docs` 和 `/openapi.json`。
