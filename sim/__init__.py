from pathlib import Path
from aenum import Enum, NoAlias, extend_enum
from typing_extensions import Optional

SIMDATA = Path(__file__).parent.absolute().joinpath("simdata")

class DeviceType(Enum):
    a10 = "a10"
    l20 = "l20"
    a100 = "a100"
    a100_40gb = "a100_40gb"
    h800 = "h800"
    h100 = "h100"

    def __str__(self):
        return self.value
    
    def mem_capacity(self):
        """Nominal GPU memory capacity in GiB."""
        match self:
            case DeviceType.a10: return 24
            case DeviceType.l20: return 48
            case DeviceType.a100: return 80
            case DeviceType.a100_40gb: return 40
            case DeviceType.h800: return 80
            case DeviceType.h100: return 80
            case _: raise NotImplementedError(f"mem_capacity not implemented for {self}")

class ModelType(Enum):

    _settings_ = NoAlias

    yi_9b = "yi-9b"
    llama2_13b = "llama2-13b"
    qwen2_5_7b = "qwen2.5-7b"
    qwen2_5_1_5b = "qwen2.5-1.5b"


    @staticmethod
    def extend_alias(canonical: str, alias: str) -> 'ModelType':
        if ModelType.from_str(alias) is None:
            raise ValueError(f'{alias} is not a valid alias for {canonical}')
        extend_enum(ModelType, canonical, alias)
        return ModelType[canonical]
            
    @staticmethod
    def from_str(s: str) -> Optional['ModelType']:
        try:
            return ModelType[s]
        except:
            for e in ModelType:
                if e.value == s:
                    return e
            return None
    
    def as_str(self) -> str:
        return self.value
    
    def mem_size(self) -> float:
        """Approximate BF16 model weight memory in GiB, excluding runtime buffers."""
        match self:
            case ModelType.qwen2_5_7b: return 14.2
            case ModelType.qwen2_5_1_5b: return 2.9
            case ModelType.yi_9b: return 18
            case ModelType.llama2_13b: return 26
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.mem_size()
                raise NotImplementedError(f"mem_size not implemented for {self}")
    
    def kvcache_size(self) -> float:
        """BF16 KV cache per token in GiB, including K and V for all layers."""
        match self:
            case ModelType.qwen2_5_7b: return 4 * 28 * 4 * 128 / (1024**3)
            case ModelType.qwen2_5_1_5b: return 4 * 28 * 2 * 128 / (1024**3)
            case ModelType.yi_9b: return 4 * 48 * 4096 / (1024**3)
            case ModelType.llama2_13b: return 4 * 40 * 5120 / (1024**3)
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.kvcache_size()
                raise NotImplementedError(f"kvcache_size not implemented for {self}")

    def max_seq_len(self) -> int:
        match self:
            case ModelType.qwen2_5_7b: return 8192
            case ModelType.qwen2_5_1_5b: return 8192
            case ModelType.yi_9b: return 4096
            case ModelType.llama2_13b: return 4096
            case _: 
                if (alias := ModelType.from_str(self.value)) is not None:
                    return alias.max_seq_len()
                raise NotImplementedError(f"max_seq_len not implemented for {self}")
    
    def max_num_tokens(self, device: DeviceType = DeviceType.a10) -> int:
        KVCACHE_PERCENTAGE = 0.8
        return int((device.mem_capacity() - self.mem_size()) * KVCACHE_PERCENTAGE / self.kvcache_size())
