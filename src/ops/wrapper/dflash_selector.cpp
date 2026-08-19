#include "ninfer/ops/dflash_selector.h"

#include "ops/launcher/dflash_selector.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* detail) {
    if (!condition) { throw std::invalid_argument(std::string("dflash_selector: ") + detail); }
}

} // namespace

void dflash_selector(const Tensor& logits, const Tensor& projected, const Tensor& anchors,
                     const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                     std::int32_t top_k, std::int32_t positions, Tensor& path, Tensor& candidates,
                     Tensor& scores, cudaStream_t stream) {
    require(top_k == 16, "only sixteen candidates per position are registered");
    require(positions >= 1 && positions <= 15, "positions must be 1..15");

    const std::int32_t vocabulary = logits.ne[0];
    const std::int32_t columns    = logits.ne[1];
    const std::int32_t rank       = projected.ne[0];
    require(rank == 256, "only selector rank 256 is registered");
    require(columns >= positions && columns % positions == 0,
            "columns must be a positive multiple of the position count");
    const std::int32_t batch = columns / positions;
    require(batch >= 1 && batch <= 8, "batch must be 1..8");

    require(logits.dtype == DType::BF16 && logits.is_contiguous() && logits.data != nullptr,
            "logits must be a contiguous BF16 matrix");
    require(projected.dtype == DType::BF16 && projected.ne[1] == columns &&
                projected.is_contiguous() && projected.data != nullptr,
            "projected must be a contiguous BF16 [R, P * B] matrix");
    require(anchors.dtype == DType::I32 && anchors.ne[0] == batch && anchors.is_contiguous() &&
                anchors.data != nullptr,
            "anchors must be a contiguous I32 [B] vector");
    for (const Tensor* codebook : {&predecessor_codebook, &successor_codebook}) {
        require(codebook->dtype == DType::BF16 && codebook->ne[0] == rank &&
                    codebook->ne[1] == vocabulary && codebook->is_contiguous() &&
                    codebook->data != nullptr,
                "each codebook must be a contiguous BF16 [R, V] matrix");
    }
    require(path.dtype == DType::I32 && path.ne[0] == positions && path.ne[1] == batch &&
                path.is_contiguous() && path.data != nullptr,
            "path must be a contiguous I32 [P, B] matrix");
    require(candidates.dtype == DType::I32 && candidates.ne[0] == top_k &&
                candidates.ne[1] == positions && candidates.ne[2] == batch &&
                candidates.is_contiguous() && candidates.data != nullptr,
            "candidates must be a contiguous I32 [K, P, B] tensor");
    require(scores.dtype == DType::FP32 && scores.ne[0] == top_k && scores.ne[1] == positions &&
                scores.ne[2] == batch && scores.is_contiguous() && scores.data != nullptr,
            "scores must be a contiguous F32 [K, P, B] tensor");

    detail::dflash_selector_launch(logits, projected, anchors, predecessor_codebook,
                                   successor_codebook, top_k, positions, batch, path, candidates,
                                   scores, stream);
}

} // namespace ninfer::ops
