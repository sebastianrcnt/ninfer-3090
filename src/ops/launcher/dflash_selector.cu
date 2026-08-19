#include "ops/launcher/dflash_selector.h"

#include "ops/kernel/dflash_selector.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void dflash_selector_launch(const Tensor& logits, const Tensor& projected, const Tensor& anchors,
                            const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                            std::int32_t top_k, std::int32_t positions, std::int32_t batch,
                            Tensor& path, Tensor& candidates, Tensor& scores,
                            cudaStream_t stream) {
    const std::int32_t vocabulary = logits.ne[0];
    const std::int32_t rank       = projected.ne[0];
    const std::int32_t columns    = positions * batch;

    const auto* logit_data       = static_cast<const __nv_bfloat16*>(logits.data);
    const auto* projected_data   = static_cast<const __nv_bfloat16*>(projected.data);
    const auto* predecessor_data = static_cast<const __nv_bfloat16*>(predecessor_codebook.data);
    const auto* successor_data   = static_cast<const __nv_bfloat16*>(successor_codebook.data);
    const auto* anchor_data      = static_cast<const std::int32_t*>(anchors.data);
    auto* candidate_data         = static_cast<std::int32_t*>(candidates.data);
    auto* path_data              = static_cast<std::int32_t*>(path.data);
    auto* score_data             = static_cast<float*>(scores.data);

    // The unary term is the selected logit; it is produced with the candidates and consumed by
    // the walk, so it lives in the score buffer until the walk overwrites it.
    dflash_selector_topk_kernel<<<columns, kDFlashSelectorThreads,
                                  static_cast<std::size_t>(top_k) * sizeof(int), stream>>>(
        logit_data, candidate_data, score_data, vocabulary, top_k);

    dflash_selector_walk_kernel<<<batch, kDFlashSelectorThreads,
                                  static_cast<std::size_t>(rank) * sizeof(float), stream>>>(
        projected_data, anchor_data, predecessor_data, successor_data, candidate_data, score_data,
        path_data, score_data, rank, positions, top_k);
}

} // namespace ninfer::ops::detail
