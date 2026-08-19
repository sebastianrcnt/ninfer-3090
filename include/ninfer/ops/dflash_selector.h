#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash 2 candidate path selection across one speculation block.
 *
 * The drafter emits an independent distribution at every block position. Taking each position's
 * argmax would ignore that consecutive tokens must agree, so the selector keeps the top K
 * candidates per position and walks one path through them, scoring each candidate against the
 * token already chosen at the preceding position:
 *
 *   score[k] = unary[k] + sum over r of predecessor[previous, r] * projected[r, position] *
 *                                       successor[candidate_k, r]
 *
 * The walk starts from the block's anchor token and is sequential in position: the winner at one
 * position is the predecessor at the next.
 *
 * logits is BF16 [V, P * B]. projected is BF16 [R, P * B], the drafter hidden state already
 * multiplied by the selector's projection. anchors is I32 [B]. Both codebooks are BF16 [R, V],
 * so one token's row is contiguous. path is I32 [P, B]; candidates is I32 [K, P, B]; scores is
 * F32 [K, P, B]. Candidate order within a position is unspecified.
 *
 * The registered domain is K=16, R=256, P=1..15, and B=1..8. The Op owns no persistent state.
 */
void dflash_selector(const Tensor& logits, const Tensor& projected, const Tensor& anchors,
                     const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                     std::int32_t top_k, std::int32_t positions, Tensor& path, Tensor& candidates,
                     Tensor& scores, cudaStream_t stream);

} // namespace ninfer::ops
