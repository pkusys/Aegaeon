import os
import torch
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from functools import cached_property
from dataclasses import dataclass, field
import threading
import itertools

import yaml

from aegaeon.utils import VRAM_ALIGN, prod
from aegaeon.logger import init_logger
from vllm.config import ModelConfig as VllmModelConfig
from vllm.config import ParallelConfig as VllmParallelConfig
from vllm.config import DeviceConfig as VllmDeviceConfig
from vllm.config import LoadConfig as VllmLoadConfig
from vllm.config import CompilationConfig as VllmCompilationConfig
from vllm.config import CompilationMode
from vllm.config import CacheConfig as VllmCaccheConfig
from vllm.config import VllmConfig

from aegaeon.graph_config import CudaGraphConfig 

if TYPE_CHECKING:
    from aegaeon.model_registry import ModelSpec
    from aegaeon.device_registry import DeviceSpec
    from aegaeon.loader import ModelHandle

# Fixed block size for Aegaeon
BLOCK_SIZE: int = 16


@dataclass
class NodeConfig:
    """Configuration for the setup of Aegaeon on a physical node."""

    # Node ID
    # This is a logic resource for ray to place a Controller and its Workers
    # all on the same physical node.
    node_id: str

    # Number of prefill engines on the node (PD mode)
    num_prefill_engines: int
    # Number of decode engines on the node (PD mode)
    num_decode_engines: int
    # Number of simple engines on the node (simple mode, mutually exclusive with PD)
    num_engines: int = 0
    # TODO: parallel config for engines goes here..
    tensor_parallel_size: int = 1

    # Policies
    prefill_disp_policy: str = "fcfs-avgload"
    prefill_sched_policy: str = "uni"
    decode_work_stealing: bool = False
    decode_work_stealing_min_batch_gap: int = 2

    # CPU cache related
    # Number of slabs for the CPU cache
    cpu_num_slabs: int = 128
    # Size of a slab for the CPU cache
    cpu_slab_size_bytes: int = 1024**3

    # Model cache related
    model_cache_size: int = 0
    cached_models: List["ModelSpec"] = field(default_factory=list)
    config_path: str = ""

    # Optional per-Controller isolation for single-host logical-node tests.
    # Physical multi-node deployments normally leave these unset because each
    # Ray node already has an independent CUDA namespace and /dev/shm.
    cuda_visible_devices: Optional[str] = None
    shm_dir: Optional[str] = None


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_size: number of pipeline parallel groups.
        pipeline_parallel_rank: rank in the pipeline parallel group.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_rank: int = 0,
    ) -> None:
        assert pipeline_parallel_size == 1, f"pipeline parallelism is not supported yet"
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_list(self) -> List[int]:
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1

    def to_vllm_config(self) -> VllmParallelConfig:
        return VllmParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

# Unique index of config version
_CONFIG_VERSION = itertools.count(1)

# Serving-length fallback for models whose configs advertise no usable sequence
# length (e.g. ALiBi-only checkpoints such as BLOOM). Small and conservative;
# deployments can raise it per model via the registry/server config.
_DEFAULT_MODEL_MAX_LEN = 2048

class ModelConfig:
    """Configuration for the model.

    Args:
        model: Model type.
    """

    def __init__(
        self,
        model: "ModelSpec",
    ):
        from vllm.transformers_utils.config import get_config, get_hf_text_config

        self.model = model
        self.hf_config = get_config(model=model.path(), trust_remote_code=True)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        # MPT exposes d_model/n_layers/n_heads and a dict attn_config; expose the
        # standard aliases on our own copy too so internal sizing code works.
        _normalize_mpt_config(self.hf_config)
        if self.hf_text_config is not self.hf_config:
            _normalize_mpt_config(self.hf_text_config)
        self.max_model_len_override = model.max_model_len
        self.version = next(_CONFIG_VERSION)

    def __repr__(self):
        return f"ModelConfig(model={self.model})"

    @cached_property
    def dtype_size(self) -> int:
        return 2

    @cached_property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16
        # return torch.float16

    @cached_property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @cached_property
    def model_impl(self) -> str:
        return "auto"

    @cached_property
    def uses_mrope(self) -> bool:
        return self.to_vllm_config().uses_mrope

    @cached_property
    def vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    @cached_property
    def hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    @cached_property
    def head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if hasattr(self.hf_text_config, "model_type") and (
            self.hf_text_config.model_type == "zamba2"
        ):
            return self.hf_text_config.attention_head_dim

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    @cached_property
    def has_attention_kv(self) -> bool:
        """Whether the model has attention layers with a conventional KV cache.

        Pure state-space models (Mamba/FalconMamba and other SSM-only
        checkpoints) expose no ``num_attention_heads``/KV-head attributes; their
        cache is recurrent state owned by the composite cache, not an
        attention KV block.
        """
        if self.hf_config.model_type in ("mpt", "dbrx"):
            return True
        for attr in (
            "num_attention_heads",
            "n_head_kv",
            "num_kv_heads",
            "num_key_value_heads",
            "multi_query_group_num",
        ):
            if getattr(self.hf_text_config, attr, None) is not None:
                return True
        return False

    @cached_property
    def total_num_kv_heads(self) -> int:
        if not self.has_attention_kv:
            return 0
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    @cached_property
    def quantization(self):
        return None

    @cached_property
    def revision(self):
        return None

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        if not self.has_attention_kv:
            return 0
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, self.total_num_kv_heads // parallel_config.tensor_parallel_size)

    @cached_property
    def max_model_len(self) -> int:
        if self.max_model_len_override is not None:
            return self.max_model_len_override
        max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # Others
            "max_sequence_length",
            "max_seq_length",
            # TeleChat and other Megatron-style configs
            "seq_length",
            "seq_len",
        ]
        for key in possible_keys:
            max_len_key = getattr(
                self.hf_config, key, getattr(self.hf_text_config, key, None)
            )
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == float("inf"):
            from transformers import AutoTokenizer
            max_model_len = int(
                AutoTokenizer.from_pretrained(
                    self.model.path(), trust_remote_code=True
                ).model_max_length
            )
        # A model that advertises no usable sequence-length configuration (e.g.
        # ALiBi-only checkpoints such as BLOOM) has no length key above, and its
        # tokenizer reports the ~1e30 placeholder. Instead of failing, give it a
        # conservative small serving length; deployments can still override via
        # the registry/server config max_model_len.
        if max_model_len == float("inf") or max_model_len >= 10**30:
            max_model_len = _DEFAULT_MODEL_MAX_LEN
        return max_model_len

    @cached_property
    def eos_token_ids(self) -> frozenset[int]:
        """Return every model-defined EOS id, including generation config ids.

        Some instruction-tuned models have more than one valid end token.  In
        particular, Gemma 3 uses both ``<eos>`` and ``<end_of_turn>``.  The
        tokenizer exposes only its primary scalar EOS id, while vLLM also
        merges the ids from ``generation_config.json``.  Aegaeon's scheduler
        must use the same complete set or it will keep decoding invisible
        special tokens after the answer has ended.
        """
        token_ids: set[int] = set()

        def add(value) -> None:
            if value is None:
                return
            if isinstance(value, int):
                token_ids.add(value)
                return
            token_ids.update(int(token_id) for token_id in value)

        add(getattr(self.hf_config, "eos_token_id", None))
        add(getattr(self.hf_text_config, "eos_token_id", None))
        generation_config = self.to_vllm_config().try_get_generation_config()
        add(generation_config.get("eos_token_id"))
        return frozenset(token_ids)

    def get_num_attention_heads(
        self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        if hasattr(self.hf_config, "num_hidden_layers"):
            total_num_hidden_layers = self.hf_config.num_hidden_layers
        else:
            total_num_hidden_layers = self.hf_text_config.num_hidden_layers
        assert total_num_hidden_layers % parallel_config.pipeline_parallel_size == 0, (
            f"Number of layers ({total_num_hidden_layers}) must be divisible "
            f"by the size of pipeline parallel group "
            f"({parallel_config.pipeline_parallel_size})."
        )
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_model_size_in_bytes(self, parallel_config: ParallelConfig) -> int:
        """The actual VRAM used by a single card is usually a bit more."""
        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            return 0
        nbytes = int(
            self.model.n_params()
            * self.dtype_size
            / parallel_config.tensor_parallel_size
        )
        aligned = int(nbytes) + VRAM_ALIGN - 1
        return aligned - aligned % VRAM_ALIGN

        # total_params = (
        #     self.vocab_size * self.hidden_size  # vocab embed
        #     + self.max_model_len * self.hidden_size  # position embed
        #     + 4
        #     * self.get_num_layers(parallel_config)
        #     * (self.hidden_size ** 2)  # attention
        #     / parallel_config.tensor_parallel_size # attention is divided by tp
        #     + 8
        #     * self.get_num_layers(parallel_config)
        #     * (self.hidden_size ** 2)  # FFN
        #     / parallel_config.tensor_parallel_size # FFN is divided by tp
        #     + 5 * self.get_num_layers(parallel_config) * self.hidden_size  # bias
        # )
        # return total_params * self.dtype_size

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla

    def get_kv_block_shape(
        self,
        parallel_config: ParallelConfig = ParallelConfig(),
    ) -> Optional[Tuple[int, ...]]:
        """
        Get the KV cache block shape for the model.

        This is per-block shape; i.e., it does not include num_blocks as a dimension.
        Returns ``None`` for attention-free (SSM-only) models: their cache is
        recurrent state laid out by the composite cache, not an attention KV
        block, so no memory-derived block estimate can be computed here.
        """
        if not self.has_attention_kv:
            return None
        if self.is_deepseek_mla:
            return (
                self.get_num_layers(parallel_config),
                BLOCK_SIZE,
                self.head_size,
            )
        else:
            # Assume FlashAttention
            return (
                self.get_num_layers(parallel_config),
                BLOCK_SIZE,
                self.get_num_kv_heads(parallel_config),
                2 * self.head_size,
            )

    def get_max_num_blocks(
        self,
        parallel_config: ParallelConfig = ParallelConfig(),
        device_type: Optional["DeviceSpec"] = None,
        memory_utilization: Optional[float] = None,
        prefetch_model_config: Optional["ModelConfig"] = None,
    ) -> Optional[int]:
        """
        Get the maximum number of KV cache blocks that can be allocated on the GPU device.

        Returns ``None`` for attention-free (SSM-only) models whose cache is not
        organised as per-token KV blocks.
        """
        kv_block_shape = self.get_kv_block_shape(parallel_config)
        if kv_block_shape is None:
            return None
        if memory_utilization is None:
            memory_utilization = get_server_config().memory_utilization

        if device_type is None:
            from aegaeon.device_registry import get_device_registry as get_dev_reg

            try:
                import torch

                name = torch.cuda.get_device_name(0)
                spec = get_dev_reg().from_str(name)
                if spec is not None:
                    device_type = spec
            except Exception:
                pass
            if device_type is None:
                device_type = next(iter(get_dev_reg()))
        total_size = self.get_available_cache_memory_bytes(
            parallel_config, device_type, memory_utilization, prefetch_model_config
        )
        block_size = prod(kv_block_shape) * self.dtype_size
        return int(total_size // block_size)

    def get_available_cache_memory_bytes(
        self,
        parallel_config: ParallelConfig,
        device_type: "DeviceSpec",
        memory_utilization: Optional[float] = None,
        prefetch_model_config: Optional["ModelConfig"] = None,
    ) -> int:
        if memory_utilization is None:
            memory_utilization = get_server_config().memory_utilization

        capacity_in_bytes = device_type.mem_capacity_in_bytes()
        if os.getenv("AEGAEON_DUMMY_WEIGHTS") == "1":
            model_size_in_bytes = 0
            pf_model_size_in_bytes = 0
        else:
            model_size_in_bytes = self.get_model_size_in_bytes(
                parallel_config=parallel_config
            )
            pf_model_size_in_bytes = (
                0
                if prefetch_model_config is None
                else prefetch_model_config.get_model_size_in_bytes(
                    parallel_config=parallel_config
                )
            )
        total_size_in_bytes = (
            capacity_in_bytes * memory_utilization
            - model_size_in_bytes
            - pf_model_size_in_bytes
        )
        if total_size_in_bytes <= 0:
            raise MemoryError(
                f"Model {self.model} does not fit on one TP rank: "
                f"capacity={capacity_in_bytes}, model={model_size_in_bytes}, "
                f"prefetch={pf_model_size_in_bytes}, TP={parallel_config.tensor_parallel_size}"
            )
        return int(total_size_in_bytes)

    def to_vllm_config(self) -> VllmModelConfig:
        # MPT uses non-standard config field names and a dict attn_config;
        # TeleChat3 declares its architecture with a lowercase 'c' while vLLM
        # registers the canonical 'TeleChat3ForCausalLM'; normalize the freshly
        # loaded vLLM config via an hf_overrides callable.
        model_type = getattr(self.hf_text_config, "model_type", "")
        if model_type == "mpt":
            hf_overrides = _normalize_mpt_config
        elif model_type == "telechat3":
            hf_overrides = _normalize_telechat3_config
        else:
            hf_overrides = {}
        return VllmModelConfig(
            model=self.model.path(),
            revision=self.revision,
            quantization=self.quantization,
            dtype=self.torch_dtype,
            model_impl=self.model_impl,
            # Align with Aegaeon::ModelConfig implementation
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=self.max_model_len,
            hf_overrides=hf_overrides,
        )


class _MPTAttnConfig:
    """Attribute, ``in`` and ``.get`` view over MPT's dict ``attn_config``."""

    def __init__(self, data: dict):
        self.__dict__["_data"] = dict(data)

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        data = self.__dict__.get("_data")
        if data is None or item not in data:
            raise AttributeError(item)
        return data[item]

    def __contains__(self, item) -> bool:
        return item in self.__dict__.get("_data", {})

    def get(self, item, default=None):
        return self.__dict__.get("_data", {}).get(item, default)


def _normalize_mpt_config(config):
    """Give vLLM's MPT path the standard config fields it reads.

    The MPT remote config uses d_model/n_layers/n_heads and keeps attn_config
    as a plain dict, while vLLM reads standard names and attribute-style
    attn_config; align them before the model/arch config is built.
    """
    attn = getattr(config, "attn_config", None)
    if isinstance(attn, dict):
        config.attn_config = _MPTAttnConfig(attn)
    for vllm_name, mpt_name in (
        ("hidden_size", "d_model"),
        ("num_hidden_layers", "n_layers"),
        ("num_attention_heads", "n_heads"),
    ):
        if not hasattr(config, vllm_name) and hasattr(config, mpt_name):
            setattr(config, vllm_name, getattr(config, mpt_name))
    return config


def _normalize_telechat3_config(config):
    """Give vLLM the canonical architecture name for TeleChat3 checkpoints.

    The TeleAI/TeleChat3-1.8B-0715 config declares
    ``architectures: ["Telechat3ForCausalLM"]`` (lowercase ``c``) while vLLM
    registers ``TeleChat3ForCausalLM``; without this rewrite vLLM rejects the
    model as unsupported.
    """
    if getattr(config, "model_type", "") != "telechat3":
        return config
    architectures = list(getattr(config, "architectures", ()) or ())
    renamed = [
        "TeleChat3ForCausalLM"
        if str(arch) == "Telechat3ForCausalLM"
        else arch
        for arch in architectures
    ]
    if renamed != architectures:
        config.architectures = renamed
    return config


@dataclass(eq=False)
class QuickLoaderConfig:
    """Config for the QuickLoader."""

    # File name for the model cache shared memory.
    model_cache_filename: Optional[str] = None
    # Size of the model cache in bytes.
    model_cache_size: Optional[int] = None
    # Snapshot of cached models.
    model_cache_snapshot: Optional[Dict[str, "ModelHandle"]] = None
    # Size of pinned buffer in the loader in bytes.
    pinned_buffer_size: int = 4 * (1024**3)
    # Fraction of device memory reserved for the QuickLoader arena.
    arena_fraction: float = 0.875

    def get_model_loading_scratch_size(self, model_path: str) -> int:
        """Return QuickLoader's peak temporary GPU allocation for a model."""
        if not self.model_cache_snapshot:
            return 0
        handle = self.model_cache_snapshot.get(model_path)
        if handle is None:
            return 0
        return handle.get_max_loading_scratch_size()


logger = init_logger(__name__)


@dataclass
class ServerConfig:
    cpu_num_slabs: int = 128
    cpu_slab_size_bytes: int = 1024**3
    memory_utilization: float = 62 / 80
    worker_num_cpus: int = 16
    nnodes: int = 1
    num_prefill_engines: int = 1
    num_decode_engines: int = 1
    num_engines: int = 0
    tensor_parallel_size: int = 1
    model_cache_size_gb: int = 0
    dummy_weights: bool = False
    decode_work_stealing: bool = False
    decode_work_stealing_min_batch_gap: int = 2
    model_placement_policy: str = "least-loaded"
    request_routing_policy: str = "least-outstanding"
    # None preserves the historical behavior of starting every configured
    # model. An empty tuple starts an empty service for later explicit deploys.
    startup_models: tuple[str, ...] | None = None
    cuda_graph: CudaGraphConfig = field(default_factory=CudaGraphConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ServerConfig":
        if not os.path.isfile(path):
            return cls()
        with open(path, "r") as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.warning(
                    "Failed to parse YAML '%s' for server config: %s", path, e
                )
                return cls()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw) -> "ServerConfig":
        if raw is None or "server" not in raw:
            return cls()

        entry = raw["server"]
        if not isinstance(entry, dict):
            return cls()

        startup_models_raw = entry.get("startup_models", cls.startup_models)
        if startup_models_raw is None:
            startup_models = None
        else:
            if not isinstance(startup_models_raw, (list, tuple)):
                raise ValueError("server.startup_models must be a list of model names")
            startup_models = tuple(
                model_name.strip()
                for model_name in startup_models_raw
                if isinstance(model_name, str) and model_name.strip()
            )
            if len(startup_models) != len(startup_models_raw):
                raise ValueError(
                    "server.startup_models must contain only non-empty strings"
                )
            if len(set(startup_models)) != len(startup_models):
                raise ValueError("server.startup_models must not contain duplicates")

        return cls(
            cpu_num_slabs=int(entry.get("cpu_num_slabs", cls.cpu_num_slabs)),
            cpu_slab_size_bytes=int(
                entry.get("cpu_slab_size_bytes", cls.cpu_slab_size_bytes)
            ),
            memory_utilization=float(
                entry.get("memory_utilization", cls.memory_utilization)
            ),
            worker_num_cpus=int(entry.get("worker_num_cpus", cls.worker_num_cpus)),
            nnodes=int(entry.get("nnodes", cls.nnodes)),
            num_prefill_engines=int(
                entry.get("num_prefill_engines", cls.num_prefill_engines)
            ),
            num_decode_engines=int(
                entry.get("num_decode_engines", cls.num_decode_engines)
            ),
            num_engines=int(entry.get("num_engines", cls.num_engines)),
            dummy_weights=bool(entry.get("dummy_weights", cls.dummy_weights)),
            tensor_parallel_size=int(
                entry.get("tensor_parallel_size", cls.tensor_parallel_size)
            ),
            model_cache_size_gb=int(
                entry.get("model_cache_size_gb", cls.model_cache_size_gb)
            ),
            decode_work_stealing=bool(
                entry.get("decode_work_stealing", cls.decode_work_stealing)
            ),
            decode_work_stealing_min_batch_gap=int(
                entry.get(
                    "decode_work_stealing_min_batch_gap",
                    cls.decode_work_stealing_min_batch_gap,
                )
            ),
            model_placement_policy=str(
                entry.get(
                    "model_placement_policy",
                    cls.model_placement_policy,
                )
            ),
            request_routing_policy=str(
                entry.get(
                    "request_routing_policy",
                    cls.request_routing_policy,
                )
            ),
            startup_models=startup_models,
            cuda_graph=CudaGraphConfig.from_dict(entry.get("cuda_graph")),
        )


_SERVER_CONFIG: Optional[ServerConfig] = None


def make_vllm_cache_config() -> VllmCaccheConfig:
    return VllmCaccheConfig(
        block_size=BLOCK_SIZE, enable_prefix_caching=False, mamba_cache_mode="none"
    )


def get_server_config() -> ServerConfig:
    global _SERVER_CONFIG
    if _SERVER_CONFIG is None:
        raise RuntimeError(
            "Server config is not set. "
            "Call set_server_config() or ensure AEGAEON_CONFIG_PATH is configured."
        )
    return _SERVER_CONFIG


def set_server_config(config: ServerConfig) -> None:
    global _SERVER_CONFIG
    _SERVER_CONFIG = config


@dataclass
class ModelConfigEntry:
    spec: "ModelSpec"
    model_config: ModelConfig
    vllm_configs: Dict[Tuple[int, int, int, int], VllmConfig] = field(
        default_factory=dict
    )


class ModelConfigRegistry:
    """Registry of mapping from ModelSpec to ModelConfig and VllmConfig."""

    def __init__(self):
        self._entries: List[ModelConfigEntry] = []
        self._lock = threading.Lock()

    def _find_index(self, name: str) -> Optional[int]:
        for index, entry in enumerate(self._entries):
            if entry.spec.name == name:
                return index
        return None

    def active_names(self) -> List[str]:
        """Return the model names currently registered (for debugging/snapshot)."""
        with self._lock:
            return [entry.spec.name for entry in self._entries]

    def register(self, spec: "ModelSpec") -> ModelConfig:
        model_config = ModelConfig(spec)  # model.path() may trigger download
        with self._lock:
            idx = self._find_index(spec.name)
            entry = ModelConfigEntry(spec=spec, model_config=model_config)
            if idx is None:
                self._entries.append(entry)
            else:
                self._entries[idx] = entry
            return entry.model_config

    def unregister(self, name: str) -> Optional[ModelConfigEntry]:
        with self._lock:
            idx = self._find_index(name)
            return None if idx is None else self._entries.pop(idx)

    def get_model_config(self, spec: Optional["ModelSpec"]) -> Optional[ModelConfig]:
        """Get globally cached model config for a model type."""
        if spec is None:
            return None
        with self._lock:
            idx = self._find_index(spec.name)
            if idx is not None:
                return self._entries[idx].model_config
        return self.register(spec)

    def get_vllm_config(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: Optional[VllmDeviceConfig] = None,
        load_config: Optional[VllmLoadConfig] = None,
        compilation_config: Optional[VllmCompilationConfig] = None,
    ) -> Optional[VllmConfig]:
        """Get globally cached vLLM config for a model config."""
        if model_config is None:
            return None
        key = tuple(parallel_config.to_list())
        with self._lock:
            idx = self._find_index(model_config.model.name)
            if idx is not None and self._entries[idx].model_config is model_config:
                cached = self._entries[idx].vllm_configs.get(key)
                if cached is not None:
                    return cached
        if device_config is None:
            device_config = VllmDeviceConfig()
        if load_config is None:
            load_config = VllmLoadConfig()
        if compilation_config is None:
            compilation_config = VllmCompilationConfig(mode=CompilationMode.NONE)
        vllm_config = VllmConfig(
            model_config=model_config.to_vllm_config(),
            parallel_config=parallel_config.to_vllm_config(),
            cache_config=make_vllm_cache_config(),
            device_config=device_config,
            load_config=load_config,
            compilation_config=compilation_config,
        )
        with self._lock:
            idx = self._find_index(model_config.model.name)
            if idx is not None and self._entries[idx].model_config is model_config:
                self._entries[idx].vllm_configs[key] = vllm_config
        return vllm_config


_model_config_registry = ModelConfigRegistry()


def get_model_config(model):
    return _model_config_registry.get_model_config(model)


def get_vllm_config(model_config, parallel_config, **kwargs):
    return _model_config_registry.get_vllm_config(
        model_config, parallel_config, **kwargs
    )


def register_model_config(spec):
    return _model_config_registry.register(spec)


def unregister_model_config(name):
    return _model_config_registry.unregister(name)
