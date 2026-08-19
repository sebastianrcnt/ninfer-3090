#include "ops/launcher/dflash_conv.h"

#include "ops/kernel/dflash_conv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {

void dflash_conv_launch(const Tensor& input, const Tensor& coefficients, const Tensor& base_kernel,
                        std::int32_t block_size, std::int32_t group_size, Tensor& output,
                        cudaStream_t stream) {
    const std::int32_t hidden  = input.ne[0];
    const std::int32_t columns = input.ne[1];
    const std::int32_t taps    = base_kernel.ne[1];
    const std::int32_t groups  = hidden / group_size;

    const auto* in   = static_cast<const __nv_bfloat16*>(input.data);
    const auto* coef = static_cast<const __nv_bfloat16*>(coefficients.data);
    const auto* base = static_cast<const __nv_bfloat16*>(base_kernel.data);
    auto* out        = static_cast<__nv_bfloat16*>(output.data);

    const int threads = hidden >= 512 ? 512 : 128;
    if (taps != 2) { throw std::invalid_argument("dflash_conv: only two taps are registered"); }
    dflash_conv_kernel<2><<<columns, threads, 0, stream>>>(in, coef, base, out, hidden, groups,
                                                           group_size, block_size);
}

} // namespace ninfer::ops::detail
