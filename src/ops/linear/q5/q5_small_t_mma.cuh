#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q5SmallTMmaStoreEpilogue {};

template <int OutputRows, int InputRows>
struct Q5SmallTGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q5SmallTMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerMmaTile    = 16;
    static constexpr int kRowTilesPerCta    = 4;
    static constexpr int kRowsPerCta        = kRowsPerMmaTile * kRowTilesPerCta;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q5_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q5SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q5_small_t_bf16_pair(std::uint8_t packed,
                                                          std::uint8_t high, int shift) {
    const int q0 =
        ((static_cast<int>(packed & 0x0fu) | (((high >> shift) & 1) << 4)) ^ 0x10) - 0x10;
    const int q1 =
        ((static_cast<int>(packed >> 4) | (((high >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10;
    Q5SmallTBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

// Exact 2..8-column Q5G64 x BF16 kernel for decode/speculative verification. Eight warps split K
// rather than output rows. Each CTA consequently reads every weight code once, reuses the staged
// activation fragment across four m16 row tiles, and reduces eight Tensor Core partials only after
// all K groups have been accumulated.
template <class Geometry, int ActiveCols, class Epilogue = Q5SmallTMmaStoreEpilogue>
__launch_bounds__(256, 2) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, Epilogue epilogue = {}) {
    using Schedule              = Q5SmallTMmaSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kHighRowBytes = kHidden / 8;
    constexpr int kMt           = Schedule::kRowTilesPerCta;
    static_assert(ActiveCols >= 2 && ActiveCols <= 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert((Geometry::kOutputRows % kRowsPerCta) == 0);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK / 2];
            std::uint8_t high[kRowsPerCta][kGroupK / 8];
            __nv_bfloat16 activations[kWarps][8 * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;
        float partial[kWarps * kMt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& high_shared  = shared.staging.high;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + q5_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16>(
                dst,
                &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK + k8 * 8]);
        }
    };

    const auto stage_weight = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row0 + row;
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Cache::cg>(
                    &code_shared[row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
            for (int chunk = lane; chunk < kGroupK / 128; chunk += 32) {
                cp_async<16, Cache::cg>(
                    &high_shared[row][chunk * 16],
                    high + static_cast<std::int64_t>(weight_row) * kHighRowBytes + group_k0 / 8 +
                        chunk * 16);
            }
        }
        constexpr int kScaleChunks = kWarps / 8;
        for (int item = tid; item < kRowsPerCta * kScaleChunks; item += kWarps * 32) {
            const int row         = item / kScaleChunks;
            const int scale_chunk = item - row * kScaleChunks;
            const int weight_row  = row0 + row;
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
    float acc[kMt][4]   = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        float group_acc[kMt][4] = {};
        const int code_base      = warp_koff / 2;
        const int high_base      = warp_koff / 8;
        const int high_shift     = 2 * lid;

#pragma unroll
        for (int khalf = 0; khalf < 2; ++khalf) {
            unsigned bf[2][2];
#pragma unroll
            for (int ks_in = 0; ks_in < 2; ++ks_in) {
                const int ks = khalf * 2 + ks_in;
                ldmatrix_x2(bf[ks_in][0], bf[ks_in][1],
                            smem_addr(&x_shared[k_split]
                                               [b_rin * kTileK +
                                                q5_small_t_swizzle_64(b_rin, ks * 16 + b_koff)]));
            }
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt) {
                const int top = mt * Schedule::kRowsPerMmaTile + gid;
                const uint4 top_codes =
                    *reinterpret_cast<const uint4*>(&code_shared[top][code_base + khalf * 16]);
                const uint4 bot_codes =
                    *reinterpret_cast<const uint4*>(&code_shared[top + 8][code_base + khalf * 16]);
                const std::uint32_t top_high = *reinterpret_cast<const std::uint32_t*>(
                    &high_shared[top][high_base + khalf * 4]);
                const std::uint32_t bot_high = *reinterpret_cast<const std::uint32_t*>(
                    &high_shared[top + 8][high_base + khalf * 4]);
                const unsigned top_words[4] = {top_codes.x, top_codes.y, top_codes.z, top_codes.w};
                const unsigned bot_words[4] = {bot_codes.x, bot_codes.y, bot_codes.z, bot_codes.w};
#pragma unroll
                for (int ks_in = 0; ks_in < 2; ++ks_in) {
                    const int word0 = ks_in * 2;
                    const std::uint8_t th0 = static_cast<std::uint8_t>(top_high >> (8 * word0));
                    const std::uint8_t th1 = static_cast<std::uint8_t>(top_high >> (8 * (word0 + 1)));
                    const std::uint8_t bh0 = static_cast<std::uint8_t>(bot_high >> (8 * word0));
                    const std::uint8_t bh1 = static_cast<std::uint8_t>(bot_high >> (8 * (word0 + 1)));
                    const unsigned af0 = q5_small_t_bf16_pair(
                        static_cast<std::uint8_t>(top_words[word0] >> (8 * lid)), th0, high_shift);
                    const unsigned af1 = q5_small_t_bf16_pair(
                        static_cast<std::uint8_t>(bot_words[word0] >> (8 * lid)), bh0, high_shift);
                    const unsigned af2 = q5_small_t_bf16_pair(
                        static_cast<std::uint8_t>(top_words[word0 + 1] >> (8 * lid)), th1,
                        high_shift);
                    const unsigned af3 = q5_small_t_bf16_pair(
                        static_cast<std::uint8_t>(bot_words[word0 + 1] >> (8 * lid)), bh1,
                        high_shift);
                    mma_bf16(group_acc[mt][0], group_acc[mt][1], group_acc[mt][2],
                             group_acc[mt][3], af0, af1, af2, af3, bf[ks_in][0], bf[ks_in][1]);
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
            acc[mt][0] = fmaf(group_acc[mt][0], top_scale, acc[mt][0]);
            acc[mt][1] = fmaf(group_acc[mt][1], top_scale, acc[mt][1]);
            acc[mt][2] = fmaf(group_acc[mt][2], bot_scale, acc[mt][2]);
            acc[mt][3] = fmaf(group_acc[mt][3], bot_scale, acc[mt][3]);
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
            store_vec(partial + ((k_split * kMt + mt) * 32 + lane) * 4,
                      make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            const float4 partner = load_vec<float4>(
                partial + (((k_split + 1) * kMt + mt) * 32 + lane) * 4);
            acc[mt][0] += partner.x;
            acc[mt][1] += partner.y;
            acc[mt][2] += partner.z;
            acc[mt][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * kMt + mt) * 32 + lane) * 4,
                          make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int mt = 0; mt < kMt; ++mt) {
            float4 sum = make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kMt + mt) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int row_base = row0 + mt * Schedule::kRowsPerMmaTile;
            const int col0     = 2 * lid;
            if constexpr (std::is_same_v<Epilogue, Q5SmallTMmaStoreEpilogue>) {
                if (col0 < ActiveCols) {
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row_base + gid] =
                        __float2bfloat16_rn(sum.x);
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row_base + gid +
                        8] = __float2bfloat16_rn(sum.z);
                }
                if (col0 + 1 < ActiveCols) {
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row_base +
                        gid] = __float2bfloat16_rn(sum.y);
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row_base +
                        gid + 8] = __float2bfloat16_rn(sum.w);
                }
            } else {
                epilogue.template store<ActiveCols>(row_base + gid, col0, sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
