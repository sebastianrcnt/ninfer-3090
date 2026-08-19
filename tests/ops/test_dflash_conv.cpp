// Numerical oracle for the DFlash 2 grouped dynamic convolution.
//
// The reference is z-lab/dflash's own PyTorch implementation, evaluated on CPU against the
// released Qwen3.8-27B-DFlash2 weights and dumped by tools/dflash2/dump_oracle.py. The dump
// rounds every input and weight to bfloat16 before computing, so a disagreement here is the
// kernel's arithmetic rather than input quantisation.
//
// Set NINFER_DFLASH2_ORACLE to the dump directory; the test skips without it.

#include "ninfer/ops/dflash_conv.h"
#include "ops/npy_fixture.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// bfloat16 keeps 8 mantissa bits, so a two-tap sum of products lands well inside 1e-3
// relative when both sides consume identical inputs.
constexpr PointwiseCriterion kCriterion{.absolute = 3e-3, .relative = 1e-3};

std::string oracle_root() {
    const char* root = std::getenv("NINFER_DFLASH2_ORACLE");
    return root == nullptr ? std::string() : std::string(root);
}

// The dump is [batch, length, hidden]; the Op reads [hidden, batch * length].
std::vector<float> to_column_major(const NpyArray& source) {
    const std::size_t batch = source.shape[0];
    const std::size_t length = source.shape[1];
    const std::size_t hidden = source.shape[2];
    std::vector<float> out(batch * length * hidden);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t l = 0; l < length; ++l) {
            const std::size_t column = b * length + l;
            for (std::size_t h = 0; h < hidden; ++h) {
                out[column * hidden + h] = source.f32[(b * length + l) * hidden + h];
            }
        }
    }
    return out;
}

// The dump is [batch, length, side, tap, group]; one side becomes [tap * group, column].
std::vector<float> coefficients_for_side(const NpyArray& source, std::size_t side) {
    const std::size_t batch = source.shape[0];
    const std::size_t length = source.shape[1];
    const std::size_t sides = source.shape[2];
    const std::size_t taps = source.shape[3];
    const std::size_t groups = source.shape[4];
    std::vector<float> out(batch * length * taps * groups);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t l = 0; l < length; ++l) {
            const std::size_t column = b * length + l;
            for (std::size_t t = 0; t < taps; ++t) {
                for (std::size_t g = 0; g < groups; ++g) {
                    const std::size_t from =
                        (((b * length + l) * sides + side) * taps + t) * groups + g;
                    out[column * taps * groups + t * groups + g] = source.f32[from];
                }
            }
        }
    }
    return out;
}

// The dump is [side, tap, hidden]; one side is already the [hidden, tap] slab the Op reads.
std::vector<float> base_for_side(const NpyArray& source, std::size_t side) {
    const std::size_t taps = source.shape[1];
    const std::size_t hidden = source.shape[2];
    std::vector<float> out(taps * hidden);
    for (std::size_t i = 0; i < taps * hidden; ++i) {
        out[i] = source.f32[side * taps * hidden + i];
    }
    return out;
}

int run_side(const std::string& root, const std::string& role, const std::string& input_name,
             std::size_t side, const std::string& expected_name) {
    const NpyArray input_array = read_npy(root + "/" + role + "_" + input_name + ".npy");
    const NpyArray dynamic = read_npy(root + "/" + role + "_dynamic.npy");
    const NpyArray base = read_npy(root + "/" + role + "_base_kernel.npy");
    const NpyArray expected = read_npy(root + "/" + role + "_" + expected_name + ".npy");

    const auto batch = static_cast<std::int32_t>(input_array.shape[0]);
    const auto block = static_cast<std::int32_t>(input_array.shape[1]);
    const auto hidden = static_cast<std::int32_t>(input_array.shape[2]);
    const auto taps = static_cast<std::int32_t>(base.shape[1]);
    const auto groups = static_cast<std::int32_t>(dynamic.shape[4]);
    const std::int32_t columns = batch * block;
    const std::int32_t group_size = hidden / groups;

    const std::vector<float> host_input = to_column_major(input_array);
    const std::vector<float> host_coefficients = coefficients_for_side(dynamic, side);
    const std::vector<float> host_base = base_for_side(base, side);

    DeviceBuffer device_input = to_device_bf16(host_input);
    DeviceBuffer device_coefficients = to_device_bf16(host_coefficients);
    DeviceBuffer device_base = to_device_bf16(host_base);
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(hidden) *
                                      static_cast<std::size_t>(columns) * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor input_tensor(device_input.p, DType::BF16, {hidden, columns});
    Tensor coefficient_tensor(device_coefficients.p, DType::BF16, {taps * groups, columns});
    Tensor base_tensor(device_base.p, DType::BF16, {hidden, taps});
    Tensor output_tensor(device_output.data(), DType::BF16, {hidden, columns});

    ops::dflash_conv(input_tensor, coefficient_tensor, base_tensor, block, group_size,
                     output_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> got = from_device_bf16(
        device_output.data(), static_cast<std::size_t>(hidden) * static_cast<std::size_t>(columns));
    const std::vector<float> reference_column_major = to_column_major(expected);
    std::vector<double> reference(reference_column_major.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        reference[i] = static_cast<double>(reference_column_major[i]);
    }

    const std::string label = "dflash_conv " + role + " " + expected_name;
    int failures = verify_pointwise(label, got, reference, kCriterion);
    failures += device_output.verify_guards(label.c_str());
    return failures;
}

} // namespace

int main() {
    const std::string root = oracle_root();
    if (root.empty()) {
        std::cerr << "skip: set NINFER_DFLASH2_ORACLE to the DFlash 2 oracle dump directory\n";
        return 77;
    }
    int failures = 0;
    try {
        for (const std::string& role : {"attention_conv", "mlp_conv"}) {
            // `prepare` convolves the sublayer input with side 0; `finish` convolves the
            // sublayer output with the side-1 coefficients the same projection produced.
            failures += run_side(root, role, "hidden", 0, "prepared");
            failures += run_side(root, role, "sublayer_out", 1, "finished");
        }
    } catch (const std::exception& error) {
        std::cerr << "dflash_conv oracle: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "FAIL dflash_conv against the DFlash 2 reference\n";
        return 1;
    }
    std::cout << "PASS dflash_conv against the DFlash 2 reference\n";
    return 0;
}
