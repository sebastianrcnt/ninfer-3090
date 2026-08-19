// Numerical oracle for the DFlash 2 candidate path selector.
//
// The reference is z-lab/dflash's CandidateSelector.select, run on CPU against the released
// Qwen3.8-27B-DFlash2 codebooks with every input rounded to bfloat16 first. Candidate order
// within a position is unspecified on both sides, so candidates are compared as sets and each
// score is matched to its own candidate token.
//
// Set NINFER_DFLASH2_ORACLE to the dump directory; the test skips without it.

#include "ninfer/ops/dflash_selector.h"
#include "ops/npy_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Scores accumulate 256 bfloat16 products in float32; the unary term is a bfloat16 logit.
constexpr PointwiseCriterion kCriterion{.absolute = 5e-3, .relative = 1e-3};

// [batch, positions, width] -> [width, batch * positions]
std::vector<float> to_column_major(const NpyArray& source) {
    const std::size_t batch = source.shape[0];
    const std::size_t positions = source.shape[1];
    const std::size_t width = source.shape[2];
    std::vector<float> out(batch * positions * width);
    for (std::size_t i = 0; i < batch * positions; ++i) {
        for (std::size_t w = 0; w < width; ++w) { out[i * width + w] = source.f32[i * width + w]; }
    }
    return out;
}

} // namespace

int main() {
    const char* root_env = std::getenv("NINFER_DFLASH2_ORACLE");
    if (root_env == nullptr || *root_env == '\0') {
        std::cerr << "skip: set NINFER_DFLASH2_ORACLE to the DFlash 2 oracle dump directory\n";
        return 77;
    }
    const std::string root(root_env);
    try {
        const NpyArray logits_array = read_npy(root + "/selector_logits.npy");
        const NpyArray projected_array = read_npy(root + "/selector_projected.npy");
        const NpyArray anchors_array = read_npy(root + "/selector_anchor_ids.npy");
        const NpyArray predecessor = read_npy(root + "/selector_predecessor_codebook.npy");
        const NpyArray successor = read_npy(root + "/selector_successor_codebook.npy");
        const NpyArray expected_path = read_npy(root + "/selector_path.npy");
        const NpyArray expected_candidates = read_npy(root + "/selector_candidates.npy");
        const NpyArray expected_scores = read_npy(root + "/selector_scores.npy");

        const auto batch = static_cast<std::int32_t>(logits_array.shape[0]);
        const auto positions = static_cast<std::int32_t>(logits_array.shape[1]);
        const auto vocabulary = static_cast<std::int32_t>(logits_array.shape[2]);
        const auto rank = static_cast<std::int32_t>(projected_array.shape[2]);
        const auto top_k = static_cast<std::int32_t>(expected_candidates.shape[2]);
        const std::int32_t columns = batch * positions;

        // The dumps are [.., vocab] and [.., rank] in C order, which is already the Op's
        // [V, column] and [R, column] memory layout.
        DeviceBuffer device_logits = to_device_bf16(to_column_major(logits_array));
        DeviceBuffer device_projected = to_device_bf16(to_column_major(projected_array));
        DeviceBuffer device_predecessor = to_device_bf16(predecessor.f32);
        DeviceBuffer device_successor = to_device_bf16(successor.f32);

        std::vector<std::int32_t> anchors(anchors_array.i64.begin(), anchors_array.i64.end());
        DeviceBuffer device_anchors = to_device<std::int32_t>(anchors);

        GuardedDeviceBuffer device_path(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
        GuardedDeviceBuffer device_candidates(static_cast<std::size_t>(columns) *
                                              static_cast<std::size_t>(top_k) *
                                              sizeof(std::int32_t));
        GuardedDeviceBuffer device_scores(static_cast<std::size_t>(columns) *
                                          static_cast<std::size_t>(top_k) * sizeof(float));
        device_path.fill(0xcd);
        device_candidates.fill(0xcd);
        device_scores.fill(0xcd);

        Tensor logits_tensor(device_logits.p, DType::BF16, {vocabulary, columns});
        Tensor projected_tensor(device_projected.p, DType::BF16, {rank, columns});
        Tensor anchors_tensor(device_anchors.p, DType::I32, {batch});
        Tensor predecessor_tensor(device_predecessor.p, DType::BF16, {rank, vocabulary});
        Tensor successor_tensor(device_successor.p, DType::BF16, {rank, vocabulary});
        Tensor path_tensor(device_path.data(), DType::I32, {positions, batch});
        Tensor candidates_tensor(device_candidates.data(), DType::I32, {top_k, positions, batch});
        Tensor scores_tensor(device_scores.data(), DType::FP32, {top_k, positions, batch});

        ops::dflash_selector(logits_tensor, projected_tensor, anchors_tensor, predecessor_tensor,
                             successor_tensor, top_k, positions, path_tensor, candidates_tensor,
                             scores_tensor, nullptr);
        cuda_synchronize();

        const auto got_path = from_device<std::int32_t>(device_path.data(),
                                                        static_cast<std::size_t>(columns));
        const auto got_candidates = from_device<std::int32_t>(
            device_candidates.data(),
            static_cast<std::size_t>(columns) * static_cast<std::size_t>(top_k));
        const auto got_scores = from_device<float>(
            device_scores.data(),
            static_cast<std::size_t>(columns) * static_cast<std::size_t>(top_k));

        int failures = 0;

        std::vector<std::int32_t> reference_path(expected_path.i64.begin(),
                                                 expected_path.i64.end());
        failures += verify_exact("dflash_selector path", got_path, reference_path);

        // Candidate sets, order-independent.
        for (std::int32_t column = 0; column < columns; ++column) {
            std::vector<std::int64_t> got(top_k);
            std::vector<std::int64_t> want(top_k);
            for (std::int32_t k = 0; k < top_k; ++k) {
                got[static_cast<std::size_t>(k)] = got_candidates[static_cast<std::size_t>(column * top_k + k)];
                want[static_cast<std::size_t>(k)] =
                    expected_candidates.i64[static_cast<std::size_t>(column * top_k + k)];
            }
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            if (got != want) {
                std::cerr << "dflash_selector: candidate set differs at column " << column << '\n';
                ++failures;
                break;
            }
        }

        // Scores, matched to the candidate token each belongs to.
        std::vector<double> got_matched;
        std::vector<double> want_matched;
        got_matched.reserve(static_cast<std::size_t>(columns) * static_cast<std::size_t>(top_k));
        want_matched.reserve(got_matched.capacity());
        for (std::int32_t column = 0; column < columns; ++column) {
            for (std::int32_t k = 0; k < top_k; ++k) {
                const std::int64_t token =
                    got_candidates[static_cast<std::size_t>(column * top_k + k)];
                std::int32_t match = -1;
                for (std::int32_t j = 0; j < top_k; ++j) {
                    if (expected_candidates.i64[static_cast<std::size_t>(column * top_k + j)] ==
                        token) {
                        match = j;
                        break;
                    }
                }
                if (match < 0) { continue; }
                got_matched.push_back(got_scores[static_cast<std::size_t>(column * top_k + k)]);
                want_matched.push_back(
                    expected_scores.f32[static_cast<std::size_t>(column * top_k + match)]);
            }
        }
        failures += verify_pointwise("dflash_selector scores", got_matched, want_matched,
                                     kCriterion);
        failures += device_path.verify_guards("dflash_selector path");
        failures += device_candidates.verify_guards("dflash_selector candidates");
        failures += device_scores.verify_guards("dflash_selector scores");

        if (failures != 0) {
            std::cerr << "FAIL dflash_selector against the DFlash 2 reference\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "dflash_selector oracle: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS dflash_selector against the DFlash 2 reference\n";
    return 0;
}
