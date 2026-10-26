# Generation Parameters

## Greedy decoding

The current Worker generates one greedy continuation per request. Sampling controls are accepted only at the values below; other values return HTTP 400.

| Parameter | Accepted values |
|---|---|
| `n` | `1` |
| `temperature` | `0` or `null` |
| `top_p`, `top_k` | `1` or `null` |
| `min_p` | `0` or `null` |
| `frequency_penalty`, `presence_penalty` | `0` or `null` |
| `repetition_penalty` | `1` or `null` |
| `logit_bias` | Empty dictionary or `null` |
| `use_beam_search` | `false` |
| `length_penalty` | `1` |
| `priority` | `0` |
| `seed` | Any integer or `null`; has no effect on greedy decoding |
| Chat `logprobs` / `top_logprobs` | `false` or `null` / `0` or `null` |
| Completion `logprobs` | `null` or omitted |

## Length and stopping

Chat defaults to 128 output tokens, and Completion defaults to 16. Chat accepts either `max_tokens` or `max_completion_tokens`; if both are supplied, their values must match. The output limit must be at least 1. The prompt must leave room for at least one output token, and generation is bounded by the model's total `max_model_len`.

| Parameter | Meaning |
|---|---|
| `stop` | Stop string or strings applied to visible text |
| `stop_token_ids` | Additional nonnegative token IDs that end generation |
| `ignore_eos` | Ignore EOS; explicit stop conditions and length limits still apply |
| `min_tokens` | Minimum output before accepting stop conditions; from 0 to the output limit |
| `include_stop_str_in_output` | Retain the matched stop string |
| `skip_special_tokens` | Omit special tokens when decoding |
| `return_token_ids` | Include token IDs in the response |

`finish_reason` is `stop`, `length`, or `tool_calls` for a parsed function call.

## Prompt formatting

`truncate_prompt_tokens` accepts a positive input-token limit or `-1`. For plain-text input, `-1` leaves the input unchanged, and truncation defaults to keeping the end. `truncation_side='left'` keeps the end; `'right'` keeps the beginning.

Chat defaults to `add_special_tokens=false` and `add_generation_prompt=true`. `continue_final_message=true` continues the last message and requires `add_generation_prompt=false`.

Complete request types and defaults are available at the running service's `/docs` and `/openapi.json`.
