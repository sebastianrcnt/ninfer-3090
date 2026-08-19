#pragma once

#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kDFlashSelectorThreads = 256;

// One block per (position, batch) column. K rounds of a block-wide maximum, each excluding the
// indices already taken, which keeps the selection exact without a merge network.
__global__ void dflash_selector_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                            std::int32_t* __restrict__ candidates,
                                            float* __restrict__ unary, std::int32_t vocabulary,
                                            std::int32_t top_k) {
    __shared__ float shared_value[kDFlashSelectorThreads];
    __shared__ int shared_index[kDFlashSelectorThreads];
    extern __shared__ int taken[];

    const int column = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row =
        logits + static_cast<std::int64_t>(column) * static_cast<std::int64_t>(vocabulary);

    for (int slot = 0; slot < top_k; ++slot) {
        float best_value = -FLT_MAX;
        int best_index   = -1;
        for (int v = thread; v < vocabulary; v += kDFlashSelectorThreads) {
            bool already = false;
            for (int t = 0; t < slot; ++t) {
                if (taken[t] == v) { already = true; break; }
            }
            if (already) { continue; }
            const float value = __bfloat162float(row[v]);
            if (value > best_value || (value == best_value && v < best_index)) {
                best_value = value;
                best_index = v;
            }
        }
        shared_value[thread] = best_value;
        shared_index[thread] = best_index;
        __syncthreads();
        for (int stride = kDFlashSelectorThreads / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                const float other = shared_value[thread + stride];
                const int other_index = shared_index[thread + stride];
                if (other > shared_value[thread] ||
                    (other == shared_value[thread] && other_index < shared_index[thread])) {
                    shared_value[thread] = other;
                    shared_index[thread] = other_index;
                }
            }
            __syncthreads();
        }
        if (thread == 0) {
            taken[slot] = shared_index[0];
            candidates[static_cast<std::int64_t>(column) * top_k + slot] = shared_index[0];
            unary[static_cast<std::int64_t>(column) * top_k + slot]     = shared_value[0];
        }
        __syncthreads();
    }
}

// One block per batch row. Positions are walked in order because each winner is the next
// position's predecessor.
__global__ void dflash_selector_walk_kernel(const __nv_bfloat16* __restrict__ projected,
                                            const std::int32_t* __restrict__ anchors,
                                            const __nv_bfloat16* __restrict__ predecessor,
                                            const __nv_bfloat16* __restrict__ successor,
                                            const std::int32_t* __restrict__ candidates,
                                            const float* __restrict__ unary,
                                            std::int32_t* __restrict__ path,
                                            float* __restrict__ scores, std::int32_t rank,
                                            std::int32_t positions, std::int32_t top_k) {
    extern __shared__ float query[];              // rank entries
    __shared__ float reduce[kDFlashSelectorThreads];
    __shared__ float candidate_score[32];
    __shared__ int previous_token;

    const int batch  = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (thread == 0) { previous_token = anchors[batch]; }
    __syncthreads();

    for (int position = 0; position < positions; ++position) {
        const std::int64_t column =
            static_cast<std::int64_t>(batch) * positions + position;
        const __nv_bfloat16* predecessor_row =
            predecessor + static_cast<std::int64_t>(previous_token) * rank;
        const __nv_bfloat16* projected_column = projected + column * rank;
        for (int r = thread; r < rank; r += kDFlashSelectorThreads) {
            query[r] = __bfloat162float(predecessor_row[r]) *
                       __bfloat162float(projected_column[r]);
        }
        __syncthreads();

        for (int k = 0; k < top_k; ++k) {
            const int token = candidates[column * top_k + k];
            const __nv_bfloat16* successor_row =
                successor + static_cast<std::int64_t>(token) * rank;
            float partial = 0.0F;
            for (int r = thread; r < rank; r += kDFlashSelectorThreads) {
                partial += query[r] * __bfloat162float(successor_row[r]);
            }
            reduce[thread] = partial;
            __syncthreads();
            for (int stride = kDFlashSelectorThreads / 2; stride > 0; stride >>= 1) {
                if (thread < stride) { reduce[thread] += reduce[thread + stride]; }
                __syncthreads();
            }
            if (thread == 0) {
                candidate_score[k] = unary[column * top_k + k] + reduce[0];
                scores[column * top_k + k] = candidate_score[k];
            }
            __syncthreads();
        }

        if (thread == 0) {
            int best = 0;
            for (int k = 1; k < top_k; ++k) {
                if (candidate_score[k] > candidate_score[best]) { best = k; }
            }
            previous_token = candidates[column * top_k + best];
            path[static_cast<std::int64_t>(batch) * positions + position] = previous_token;
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops
