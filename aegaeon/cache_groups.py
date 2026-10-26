"""Compatibility exports for Aegaeon's canonical cache-group contracts.

The live scheduler, block-table, and worker contracts are defined together in
``aegaeon.block_manager``. Keeping a second implementation here previously
allowed its constructor and null-block semantics to drift from the code used
by the service. Importing from this module now returns those canonical types.
"""

from aegaeon.block_manager import (
    LOGICAL_NULL_BLOCK_ID,
    VLLM_NULL_BLOCK_ID,
    BlockTable,
    CacheGroupConfig,
    CacheGroupRuntimeConfig,
)

__all__ = [
    "LOGICAL_NULL_BLOCK_ID",
    "VLLM_NULL_BLOCK_ID",
    "BlockTable",
    "CacheGroupConfig",
    "CacheGroupRuntimeConfig",
]
