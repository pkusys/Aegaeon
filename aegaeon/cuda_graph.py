from dataclasses import dataclass

from aegaeon.graph_config import CudaGraphConfig


@dataclass(frozen=True)
class GraphKey:
    batch_size: int
    seq_len_bucket: int


def select_decode_graph_key(
    config: CudaGraphConfig,
    batch_size: int,
    max_seq_len: int,
) -> GraphKey | None:
    """Select the smallest configured decode graph that fits the batch."""
    if config.mode == "off" or batch_size not in config.batch_sizes:
        return None

    for seq_len_bucket in config.seq_len_buckets:
        if max_seq_len <= seq_len_bucket:
            return GraphKey(
                batch_size=batch_size,
                seq_len_bucket=seq_len_bucket,
            )

    return None
