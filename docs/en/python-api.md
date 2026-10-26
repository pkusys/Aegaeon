# Python API

`LLMService`, `NodeConfig`, and `Request` provide direct inference from token IDs. The YAML file supplies model and device metadata; `NodeConfig` defines each node's engines, caches, and initially cached models.

The example uses [the Simple-mode configuration](examples/simple.yaml), with paths and profiles adapted to the host. Its relative `config_path` assumes a repository-root working directory; an installed-wheel application can use an absolute path to its own copy. The Ray runtime must already expose the `node_0` resource described in [Deployment](deployment.md).

## Example

```python
import os
import time
import ray
from aegaeon import LLMService, NodeConfig, Request
from aegaeon.config import get_model_config
from aegaeon.models import set_model_registry
from aegaeon.model_registry import get_registry
from aegaeon.utils import get_tokenizer

config_path = os.path.abspath('docs/en/examples/simple.yaml')
os.environ['AEGAEON_CONFIG_PATH'] = config_path
set_model_registry(config_path)
ray.init(address='127.0.0.1:6789')
spec = get_registry().get_by_name('Qwen/Qwen3-4B')
assert spec is not None
model_config = get_model_config(spec)
tokenizer = get_tokenizer(spec.path())

service = LLMService([
    NodeConfig(
        node_id='node_0',
        num_prefill_engines=0,
        num_decode_engines=0,
        num_engines=1,
        tensor_parallel_size=1,
        cpu_num_slabs=8,
        cpu_slab_size_bytes=1024**3,
        model_cache_size=20 * 1024**3,
        cached_models=[spec],
        config_path=config_path,
    )
])
request = Request(
    model=spec,
    arrival_time=time.time(),
    request_id=0,
    prompt_token_ids=tokenizer.encode('KV Cache is'),
    decode_tokens=16,
    eos_token_ids=model_config.eos_token_ids,
    max_model_len=model_config.max_model_len,
)
try:
    outputs = service.serve([request])
    if isinstance(outputs[0], BaseException):
        raise outputs[0]
    token_ids = [step.new_token_id for step in outputs[0]]
    print(tokenizer.decode(token_ids, skip_special_tokens=True))
finally:
    ray.shutdown()
```

## Interfaces

- `NodeConfig` sets each node's engines, caches, and models; memory sizes use bytes.
- `Request` supplies prompt token IDs, output length, EOS, and context limits. Apply a chat template before tokenizing Chat messages.
- `LLMService.serve()` returns one `StepOutput` list or exception per request, in input order. It runs its own event loop and is called from synchronous code.
- `deploy_model_replicas()` and `undeploy_model_replicas()` manage replicas asynchronously.

Use the [HTTP API](api.md) for chat formatting, streaming, and response schemas.
