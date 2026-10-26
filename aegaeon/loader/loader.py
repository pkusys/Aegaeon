from typing import Optional
from vllm.config import VllmConfig, LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from .quick_loader import QuickLoader

DEF_LOAD_CONFIG = LoadConfig()

def _normalize_hf_config(vllm_config):
    """Inject attributes old remote-code checkpoints omit but vLLM reads."""
    hf_config = vllm_config.model_config.hf_config
    if str(getattr(hf_config, "model_type", "")) == "falcon" \
            and not hasattr(hf_config, "tie_word_embeddings"):
        hf_config.tie_word_embeddings = None

def get_model(
    *,
    vllm_config: VllmConfig,
    quick_loader: Optional["QuickLoader"] = None,
):
    _normalize_hf_config(vllm_config)
    loader = quick_loader if quick_loader is not None else DefaultModelLoader(DEF_LOAD_CONFIG)
    return loader.load_model(
        vllm_config=vllm_config,
        model_config=vllm_config.model_config,
    )