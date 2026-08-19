#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// One CUDA block per token column. Threads stride the hidden axis; every tap reads the same
// column-local position, so the block-boundary predicate is computed once per block.
template <int Taps>
__global__ void dflash_conv_kernel(const __nv_bfloat16* __restrict__ input,
                                   const __nv_bfloat16* __restrict__ coefficients,
                                   const __nv_bfloat16* __restrict__ base_kernel,
                                   __nv_bfloat16* __restrict__ output, std::int32_t hidden,
                                   std::int32_t groups, std::int32_t group_size,
                                   std::int32_t block_size) {
    const int column   = static_cast<int>(blockIdx.x);
    const int position = column % block_size;
    const __nv_bfloat16* coefficient_column = coefficients + static_cast<std::int64_t>(column) *
                                                                 static_cast<std::int64_t>(Taps) *
                                                                 groups;

    for (int h = static_cast<int>(threadIdx.x); h < hidden;
         h += static_cast<int>(blockDim.x)) {
        const int group = h / group_size;
        float accumulator = 0.0F;
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            if (tap > position) { break; }
            const float base    = __bfloat162float(base_kernel[static_cast<std::int64_t>(tap) * hidden + h]);
            const float dynamic = __bfloat162float(coefficient_column[tap * groups + group]);
            const float value   = __bfloat162float(
                input[static_cast<std::int64_t>(column - tap) * hidden + h]);
            accumulator += (base + dynamic) * value;
        }
        output[static_cast<std::int64_t>(column) * hidden + h] = __float2bfloat16(accumulator);
    }
}

} // namespace ninfer::ops
