#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;
constexpr int kK    = 17408;

using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_exact(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    q5_rowsplit_gemv_small_t_exact_residual_launch_kernel<kRows, kK, 16, 2, ActiveCols>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), stream);
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
