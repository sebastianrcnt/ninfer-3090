#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;
constexpr int kK    = 17408;
using Geometry      = Q5SmallTGeometry<kRows, kK>;

struct ResidualEpilogue {
    __nv_bfloat16* out;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            const std::int64_t top = static_cast<std::int64_t>(col0) * kRows + row;
            const std::int64_t bot = top + 8;
            out[top] = __float2bfloat16_rn(__bfloat162float(out[top]) + projected.x);
            out[bot] = __float2bfloat16_rn(__bfloat162float(out[bot]) + projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            const std::int64_t top = static_cast<std::int64_t>(col0 + 1) * kRows + row;
            const std::int64_t bot = top + 8;
            out[top] = __float2bfloat16_rn(__bfloat162float(out[top]) + projected.y);
            out[bot] = __float2bfloat16_rn(__bfloat162float(out[bot]) + projected.w);
        }
    }
};

using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_exact(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    constexpr int kBlocks = kRows / Q5SmallTMmaSchedule::kRowsPerCta;
    const ResidualEpilogue epilogue{static_cast<__nv_bfloat16*>(residual_out.data)};
    q5_small_t_mma_kernel<Geometry, ActiveCols, ResidualEpilogue>
        <<<kBlocks, Q5SmallTMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), epilogue);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launcher, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<7>{});

} // namespace

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    if (w.n != kRows || w.k != kK || w.padded_shape[1] != kK || x.ne[1] < 2 || x.ne[1] > 8) {
        throw std::invalid_argument("Q5 LinearAdd small-T MMA: unsupported exact problem");
    }
    kLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, w, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
