#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash_selector_launch(const Tensor& logits, const Tensor& projected, const Tensor& anchors,
                            const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                            std::int32_t top_k, std::int32_t positions, std::int32_t batch,
                            Tensor& path, Tensor& candidates, Tensor& scores, cudaStream_t stream);

} // namespace ninfer::ops::detail
