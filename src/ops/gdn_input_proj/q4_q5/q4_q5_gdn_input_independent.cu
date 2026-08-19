#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

struct Q4GdnIndependentExactEpilogue {
    std::int32_t out_ld;

    template <bool, int, int Tokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[Tokens]) const {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            out[static_cast<std::int64_t>(token) * out_ld + row] =
                __float2bfloat16_rn(values[token]);
        }
    }
};

struct Q5GdnIndependentExactEpilogue {
    std::int32_t value_ld;

    template <bool, int, int Tokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16* value, __nv_bfloat16* z,
                                               std::int32_t, std::int32_t, std::int32_t row,
                                               const float (&values)[Tokens]) const {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (row < kValueRows) {
                value[static_cast<std::int64_t>(token) * value_ld + row] =
                    __float2bfloat16_rn(values[token]);
            } else {
                z[static_cast<std::int64_t>(token) * kZRows + row - kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, out, stream);
    }
}

template <int Tokens>
void launch_q4_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    constexpr int kBlocks = kQkRows / Schedule::kRowsPerCta;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    q4_rowsplit_gemv_small_t_exact_kernel<Schedule, Tokens, Q4GdnIndependentExactEpilogue>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), nullptr, kQkRows, kHidden,
            Q4GdnIndependentExactEpilogue{out_ld});
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv(x, weight, out, stream);
        return;
    }
    switch (x.ne[1]) {
    case 2:
        launch_q4_exact<2>(x, weight, out, stream);
        return;
    case 3:
        launch_q4_exact<3>(x, weight, out, stream);
        return;
    case 4:
        launch_q4_exact<4>(x, weight, out, stream);
        return;
    case 5:
        launch_q4_exact<5>(x, weight, out, stream);
        return;
    case 6:
        launch_q4_exact<6>(x, weight, out, stream);
        return;
    case 7:
        launch_q4_exact<7>(x, weight, out, stream);
        return;
    case 8:
        launch_q4_exact<8>(x, weight, out, stream);
        return;
    default:
        break;
    }
    if (x.ne[1] <= 4) {
        launch_q4_simt_route<Q4GdnSimtR8C4Schedule>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q4_simt_route<Q4GdnSimtR8C8Schedule>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<kValueZRows, kHidden, kRowsPerBlock, 2, true, false, true, kValueRows>
        <<<kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                     cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    constexpr int kBlocks       = kValueZRows / kRowsPerBlock;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    q5_rowsplit_gemv_small_t_exact_kernel<kValueZRows, kHidden, kRowsPerBlock, 2, Cols,
                                           Q5GdnIndependentExactEpilogue>
        <<<kBlocks, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnIndependentExactEpilogue{out_ld});
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_small_t_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                             cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_exact<2>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_exact<3>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_exact<4>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_exact<5>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_exact<6>(x, weight, value, z, stream);
        return;
    case 7:
        launch_q5_exact<7>(x, weight, value, z, stream);
        return;
    case 8:
        launch_q5_exact<8>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 exact small-T requires T in [2,8]");
    }
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld, kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 8) {
        launch_q5_small_t_exact(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt_r8_c8(x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    launch_q4(x, qk_weight, qk, stream);
    launch_q5(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
