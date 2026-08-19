#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash 2 grouped dynamic causal convolution across one speculation block.
 *
 * Each drafter sublayer is wrapped by one of these. The coefficient applied to a tap is the
 * sum of a per-channel base kernel and a per-group dynamic delta produced by projecting the
 * sublayer input, so a single projection serves both the input side and the output side.
 *
 *   out[h, c] = sum over taps t of (base[h, t] + delta[t, h / group_size, c]) * in[h, c - t]
 *
 * restricted to t <= c mod block_size. Columns are the flattened [block, batch] token axis, so
 * that restriction is what stops a block from convolving with the tail of the block before it;
 * tap 0 is always applied and the base kernel is identity there in a released checkpoint.
 *
 * input and output are distinct contiguous BF16 [H, C] tensors. base_kernel is a contiguous
 * BF16 [H, T] view of one side of the stored kernel. coefficients is a contiguous BF16
 * [T * (H / group_size), C] tensor whose rows are ordered tap-major then group.
 *
 * The registered domain is T=2, block_size=1..16, group_size dividing H, and C a positive
 * multiple of block_size. The Op owns no workspace or persistent state.
 */
void dflash_conv(const Tensor& input, const Tensor& coefficients, const Tensor& base_kernel,
                 std::int32_t block_size, std::int32_t group_size, Tensor& output,
                 cudaStream_t stream);

} // namespace ninfer::ops
