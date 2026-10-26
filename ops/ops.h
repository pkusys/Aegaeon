#include <torch/extension.h>
#include <vector>

void swap(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,
	torch::Tensor& kv_cache,
	torch::Tensor& kv_swap
);