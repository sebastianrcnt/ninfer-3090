#include "ninfer/ops/dflash_conv.h"

#include "ops/launcher/dflash_conv.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                         const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("dflash_conv: " + std::string(name) +
                                    " must be a contiguous BF16 matrix of the declared shape");
    }
}

} // namespace

void dflash_conv(const Tensor& input, const Tensor& coefficients, const Tensor& base_kernel,
                 std::int32_t block_size, std::int32_t group_size, Tensor& output,
                 cudaStream_t stream) {
    const std::int32_t hidden  = input.ne[0];
    const std::int32_t columns = input.ne[1];
    if (block_size < 1 || block_size > 16) {
        throw std::invalid_argument("dflash_conv: block_size must be 1..16");
    }
    if (hidden < 1 || group_size < 1 || hidden % group_size != 0) {
        throw std::invalid_argument("dflash_conv: group_size must divide the hidden width");
    }
    if (columns < 1 || columns % block_size != 0) {
        throw std::invalid_argument("dflash_conv: columns must be a positive multiple of the block");
    }
    const std::int32_t taps = base_kernel.ne[1];
    if (taps != 2) { throw std::invalid_argument("dflash_conv: only two taps are registered"); }

    require_bf16_matrix(input, hidden, columns, "input");
    require_bf16_matrix(output, hidden, columns, "output");
    require_bf16_matrix(base_kernel, hidden, taps, "base_kernel");
    require_bf16_matrix(coefficients, taps * (hidden / group_size), columns, "coefficients");
    if (input.data == output.data) {
        throw std::invalid_argument("dflash_conv: input and output must be distinct");
    }

    detail::dflash_conv_launch(input, coefficients, base_kernel, block_size, group_size, output,
                               stream);
}

} // namespace ninfer::ops
