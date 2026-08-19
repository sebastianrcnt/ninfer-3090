#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash_conv_launch(const Tensor& input, const Tensor& coefficients, const Tensor& base_kernel,
                        std::int32_t block_size, std::int32_t group_size, Tensor& output,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
