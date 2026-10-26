from typing import Optional, TYPE_CHECKING

from aegaeon.logger import init_logger

if TYPE_CHECKING:
    from aegaeon.model_registry import ModelSpec

logger = init_logger(__name__)


class _ModelNamespace:
    def __getattr__(self, name: str) -> "ModelSpec":
        from aegaeon.model_registry import get_registry
        spec = get_registry().get_by_name(name)
        if spec is not None:
            return spec
        raise AttributeError(
            f"Model '{name}' not found in registry. "
            f"Ensure --model-config YAML defines it, or the default registry includes it."
        )

    def from_str(self, s: str, default=None) -> Optional["ModelSpec"]:
        from aegaeon.model_registry import get_registry
        return get_registry().from_str(s, default=default)

    def __iter__(self):
        from aegaeon.model_registry import get_registry
        return iter(get_registry())

    def __len__(self):
        from aegaeon.model_registry import get_registry
        return len(get_registry())

    def __contains__(self, item):
        from aegaeon.model_registry import get_registry
        return item in get_registry()


ModelType = _ModelNamespace()


def set_model_registry(path_or_registry):
    from aegaeon.model_registry import (
        ModelSpecRegistry, set_registry as set_model_reg, get_registry,
    )
    from aegaeon.device_registry import DeviceRegistry, set_registry as set_device_reg
    from aegaeon.config import ServerConfig, set_server_config

    # Skip if already initialised (Worker re-loads the same config)
    try:
        get_registry()
        return
    except RuntimeError:
        pass

    if isinstance(path_or_registry, str):
        model_registry = ModelSpecRegistry.from_yaml(path_or_registry)
        device_registry = DeviceRegistry.from_yaml(path_or_registry)
        server_config = ServerConfig.from_yaml(path_or_registry)
        set_model_reg(model_registry)
        set_device_reg(device_registry)
        set_server_config(server_config)
    elif isinstance(path_or_registry, ModelSpecRegistry):
        set_model_reg(path_or_registry)
    else:
        raise TypeError(
            f"Expected str (path to YAML) or ModelSpecRegistry, "
            f"got {type(path_or_registry).__name__}"
        )


def load_config_from_content(content: str):
    import yaml
    from aegaeon.model_registry import (
        ModelSpecRegistry, set_registry as set_model_reg, get_registry,
    )
    from aegaeon.device_registry import DeviceRegistry, set_registry as set_device_reg
    from aegaeon.config import ServerConfig, set_server_config

    # Skip if already initialised (Worker re-loads the same config)
    try:
        get_registry()
        return
    except RuntimeError:
        pass

    raw = yaml.safe_load(content)
    model_registry = ModelSpecRegistry.from_dict(raw)
    device_registry = DeviceRegistry.from_dict(raw)
    server_config = ServerConfig.from_dict(raw)
    set_model_reg(model_registry)
    set_device_reg(device_registry)
    set_server_config(server_config)
