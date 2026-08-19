#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / kGroupK;
};

struct Q4SwiGluSmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 2;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerMmaTile    = 16;
    static constexpr int kRowTilesPerCta    = 4;
    static constexpr int kRowsPerCta        = kRowsPerMmaTile * kRowTilesPerCta;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 16;

    // Local rows form two 16-row mma tiles. Within a tile the top half carries gate rows and
    // the bottom half the matching up rows, so a single mma pair yields both operands of the
    // SwiGLU product for one output row.
    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + ((local_row >> 4) * 8) + (local_row & 7) +
               (((local_row >> 3) & 1) != 0 ? kIntermediate : 0);
    }

    __device__ __forceinline__ int output_row_base(int output_row0, int tile) const {
        return output_row0 + tile * 8;
    }
};

struct Q4SwiGluWideSmallTRows {
    static constexpr int kOutputRowsPerCta = 32;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + ((local_row >> 4) * 8) + (local_row & 7) +
               (((local_row >> 3) & 1) != 0 ? kIntermediate : 0);
    }

    __device__ __forceinline__ int output_row_base(int output_row0, int tile) const {
        return output_row0 + tile * 8;
    }
};

struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_small_t_mma_active(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Schedule =
        std::conditional_t<(ActiveCols <= 8), Q4SwiGluSmallTSchedule, Q4DraftSmallTSchedule>;
    using Rows =
        std::conditional_t<(ActiveCols <= 8), Q4SwiGluWideSmallTRows, Q4SwiGluSmallTRows>;
    constexpr int kBlocks = kIntermediate / Rows::kOutputRowsPerCta;
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data)};
    q4_small_t_mma_kernel<Schedule, Q4SwiGluSmallTGeometry, TileCols, ActiveCols,
                          Q4SwiGluSmallTEpilogue, Rows>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Rows{});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_mma_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(std::make_index_sequence<31>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

// Small-T SwiGLU with the exact arithmetic contract of the T=1 GEMV.  A warp owns one
// gate/up row pair and keeps one accumulator pair per token while streaming each weight tile
// only once.  Interleaving tokens leaves both the per-token FMA sequence and warp reduction
// order unchanged, so Verify is bitwise identical to repeated ordinary decode without
// re-reading the 52.4 MiB gate/up matrix for every speculative position.
template <int ActiveCols>
__global__ void q4_linear_swiglu_small_t_exact_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    static_assert(ActiveCols >= 2 && ActiveCols <= 8);

    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row =
        scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;

    float gate_acc[ActiveCols] = {};
    float up_acc[ActiveCols]   = {};
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales =
            reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales =
            reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale = __half2float(
                __ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale = __half2float(
                __ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));
            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed =
                static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0 = sign_extend<4>(up_packed & 0x0f);
            const int up_q1 = sign_extend<4>(up_packed >> 4);
            const int k0 = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
#pragma unroll
            for (int token = 0; token < ActiveCols; ++token) {
                const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(
                    x + static_cast<std::int64_t>(token) * kK);
                const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
                gate_acc[token] =
                    fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc[token]);
                gate_acc[token] =
                    fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc[token]);
                up_acc[token] =
                    fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc[token]);
                up_acc[token] =
                    fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc[token]);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int token = 0; token < ActiveCols; ++token) {
        const float gate_sum = warp_reduce_sum(gate_acc[token]);
        const float up_sum   = warp_reduce_sum(up_acc[token]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * kIntermediate + out_row] =
                __float2bfloat16(silu(gate_sum) * up_sum);
        }
    }
}

template <int ActiveCols>
void launch_small_t_exact_active(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    constexpr int kBlocks = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_small_t_exact_kernel<ActiveCols><<<kBlocks, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_exact_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_exact_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallTExactLaunchers = make_small_t_exact_launchers(std::make_index_sequence<7>{});

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    if (x.ne[1] <= 8) {
        kSmallTExactLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, w, out, stream);
    } else {
        kSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, w, out, stream);
    }
}

} // namespace ninfer::ops::detail
