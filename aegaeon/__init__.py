"""
Aegaeon: An LLM serving system that mixes requests.

The implementation depends on vLLM's adaptation of various models 
and the various kernels (e.g., PagedAttn, FlashAttn). The remaining
infrastructure borrows from DistServe. We update its block manager,
KV cache management, and scheduler implementation.
"""

__all__ = [
    "LLMService",
    "NodeConfig",
    "Request",
]

def __getattr__(name):
    if name == "LLMService":
        from aegaeon.llm import LLMService
        return LLMService
    if name == "NodeConfig":
        from aegaeon.config import NodeConfig
        return NodeConfig
    if name == "Request":
        from aegaeon.request import Request
        return Request
    raise AttributeError(f"module 'aegaeon' has no attribute {name!r}")
