#include <torch/all.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/StorageUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>

#include <vector>
#include <cstdio>
#include <cstdlib>

/*
The Aegaeon swapping kernel.

The source_block_ids and target_block_ids are the block ids of the blocks to be swapped.
source_block_ids[0] will be copied to target_block_ids[0] and so on
`is_swap_in` defines whether the swap is a swap-in or swap-out (swap-in means
to swap from CPU to GPU, swap-out means to swap from GPU to CPU)

Here we do not pass a cudaStream to the function. Instead we use the current
stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
responsibility to set the current stream before calling this function.
*/
void swap(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,
	torch::Tensor& kv_cache,
	torch::Tensor& kv_swap
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
	size_t block_size_in_bytes = kv_cache.numel() * 2 /* float16 */ / kv_cache.size(0);
	int num_blocks_to_swap = source_block_ids.size();
	for (int i = 0; i < num_blocks_to_swap; i++) {
		int64_t source_block_id = source_block_ids[i];
		int64_t target_block_id = target_block_ids[i];

		if (is_swap_in) {
			// Copy from CPU to GPU
			cudaMemcpyAsync(
				((char*)kv_cache.data_ptr()) + target_block_id * block_size_in_bytes,
				((char*)kv_swap.data_ptr()) + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyHostToDevice,
				stream
			);
		} else {
			// Copy from GPU to CPU
			cudaMemcpyAsync(
				((char*)kv_swap.data_ptr()) + target_block_id * block_size_in_bytes,
				((char*)kv_cache.data_ptr()) + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyDeviceToHost,
				stream
			);
		}
	}
}