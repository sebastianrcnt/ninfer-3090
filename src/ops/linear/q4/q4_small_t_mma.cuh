#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4SmallTMmaStoreEpilogue {};

struct Q4SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 32;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }

    __device__ __forceinline__ int output_row_base(int output_row0, int tile) const {
        return output_row0 + tile * 16;
    }
};

template <int OutputRows, int InputRows>
struct Q4SmallTGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

template <int InputRows>
using Q4DraftHeadGeometry = Q4SmallTGeometry<131072, InputRows>;

struct Q4DraftSmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerMmaTile    = 16;
    // Two row tiles per CTA. The activation staging cost is per CTA and independent of the
    // row count, so widening the CTA halves the activation traffic through the L1/TEX pipe
    // -- which the profiler shows saturated at ~89% while DRAM sits at 38%.
    static constexpr int kRowTilesPerCta    = 2;
    static constexpr int kRowsPerCta        = kRowsPerMmaTile * kRowTilesPerCta;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q4_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q4SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q4_small_t_bf16_pair(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    Q4SmallTBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <class Schedule, class Geometry, int TileCols, int ActiveCols,
          class Epilogue = Q4SmallTMmaStoreEpilogue,
          class RowPolicy = Q4SmallTMmaIdentityRows>
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) __global__
    void q4_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                               RowPolicy row_policy = {}) {
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    constexpr int kMt           = Schedule::kRowTilesPerCta;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 2 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert((kWarps % 8) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK / 2];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kMt * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + q4_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16>(
                dst,
                &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK + k8 * 8]);
        }
    };

    const auto stage_weight = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(row0, row);
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Schedule::kCodeCache>(
                    &code_shared[row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
        }
        constexpr int kScaleChunks = kWarps / 8;
        for (int item = tid; item < kRowsPerCta * kScaleChunks; item += kWarps * 32) {
            const int row         = item / kScaleChunks;
            const int scale_chunk = item - row * kScaleChunks;
            const int weight_row  = row_policy.weight_row(row0, row);
            cp_async<16>(
                &scale_shared[row][scale_chunk * 8],
                scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                          group_k0 / 64 + scale_chunk * 8) *
                             2);
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kMt][kNt][4] = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0      = group_index * kGroupK;
        float group_acc[kMt][kNt][4] = {};

        // The four k-steps of a group are walked in halves so that each row's codes arrive as
        // one 16-byte shared load instead of eight single-byte ones. L1/TEX transaction count,
        // not DRAM bytes, is what saturates on this kernel.
        const int code_base = warp_koff / 2;
        const int shift     = 8 * lid;
#pragma unroll
        for (int khalf = 0; khalf < 2; ++khalf) {
            // The activation fragment depends only on (ks, nt), so it is loaded once and
            // reused by every row tile: one ldmatrix now feeds kMt mma instructions.
            unsigned bf[2][kNt][2];
#pragma unroll
            for (int ks_in = 0; ks_in < 2; ++ks_in) {
                const int ks = khalf * 2 + ks_in;
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    const int br = nt * 8 + b_rin;
                    ldmatrix_x2(bf[ks_in][nt][0], bf[ks_in][nt][1],
                                smem_addr(&x_shared[k_split]
                                                   [br * kTileK +
                                                    q4_small_t_swizzle_64(br, ks * 16 + b_koff)]));
                }
            }
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt) {
                const int top   = mt * Schedule::kRowsPerMmaTile + gid;
                const uint4 top_codes =
                    *reinterpret_cast<const uint4*>(&code_shared[top][code_base + khalf * 16]);
                const uint4 bot_codes =
                    *reinterpret_cast<const uint4*>(&code_shared[top + 8][code_base + khalf * 16]);
                const unsigned top_words[4] = {top_codes.x, top_codes.y, top_codes.z, top_codes.w};
                const unsigned bot_words[4] = {bot_codes.x, bot_codes.y, bot_codes.z, bot_codes.w};
#pragma unroll
                for (int ks_in = 0; ks_in < 2; ++ks_in) {
                    const unsigned af0 = q4_small_t_bf16_pair(
                        static_cast<std::uint8_t>(top_words[ks_in * 2] >> shift));
                    const unsigned af1 = q4_small_t_bf16_pair(
                        static_cast<std::uint8_t>(bot_words[ks_in * 2] >> shift));
                    const unsigned af2 = q4_small_t_bf16_pair(
                        static_cast<std::uint8_t>(top_words[ks_in * 2 + 1] >> shift));
                    const unsigned af3 = q4_small_t_bf16_pair(
                        static_cast<std::uint8_t>(bot_words[ks_in * 2 + 1] >> shift));
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt) {
                        mma_bf16(group_acc[mt][nt][0], group_acc[mt][nt][1], group_acc[mt][nt][2],
                                 group_acc[mt][nt][3], af0, af1, af2, af3, bf[ks_in][nt][0],
                                 bf[ks_in][nt][1]);
                    }
                }
            }
        }

#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            const int top = mt * Schedule::kRowsPerMmaTile + gid;
            const float top_scale =
                __half2float(__ushort_as_half(scale_shared[top][k_split]));
            const float bot_scale =
                __half2float(__ushort_as_half(scale_shared[top + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[mt][nt][0] = fmaf(group_acc[mt][nt][0], top_scale, acc[mt][nt][0]);
                acc[mt][nt][1] = fmaf(group_acc[mt][nt][1], top_scale, acc[mt][nt][1]);
                acc[mt][nt][2] = fmaf(group_acc[mt][nt][2], bot_scale, acc[mt][nt][2]);
                acc[mt][nt][3] = fmaf(group_acc[mt][nt][3], bot_scale, acc[mt][nt][3]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                store_vec(partial + (((k_split * kMt + mt) * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2],
                                      acc[mt][nt][3]));
            }
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float4 partner = load_vec<float4>(
                    partial + ((((k_split + 1) * kMt + mt) * kNt + nt) * 32 + lane) * 4);
                acc[mt][nt][0] += partner.x;
                acc[mt][nt][1] += partner.y;
                acc[mt][nt][2] += partner.z;
                acc[mt][nt][3] += partner.w;
                if (k_split != 0) {
                    store_vec(partial + (((k_split * kMt + mt) * kNt + nt) * 32 + lane) * 4,
                              make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2],
                                          acc[mt][nt][3]));
                }
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            const int row_base = row_policy.output_row_base(row0, mt);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum = make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2],
                                         acc[mt][nt][3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value = load_vec<float4>(
                        partial + (((split * kMt + mt) * kNt + nt) * 32 + lane) * 4);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col0 = nt * 8 + 2 * lid;
                if constexpr (std::is_same_v<Epilogue, Q4SmallTMmaStoreEpilogue>) {
                    if (col0 < ActiveCols) {
                        out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row_base +
                            gid] = __float2bfloat16_rn(sum.x);
                        out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row_base +
                            gid + 8] = __float2bfloat16_rn(sum.z);
                    }
                    if (col0 + 1 < ActiveCols) {
                        out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows +
                            row_base + gid] = __float2bfloat16_rn(sum.y);
                        out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows +
                            row_base + gid + 8] = __float2bfloat16_rn(sum.w);
                    }
                } else {
                    epilogue.template store<ActiveCols>(row_base + gid, col0, sum);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
