// Losslessness of DFlash 2 speculative decoding against the target it drafts for.
//
// Greedy decoding must return exactly the tokens the target returns on its own: speculation
// changes how tokens are produced, never which ones. The check is token-identity between the
// same prompt decoded with no speculative backend and with DFlash 2, on one engine each.
//
// Set NINFER_QWEN3_8_27B_WEIGHTS and NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS; the test skips without
// them.

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::filesystem::path environment_path(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::filesystem::path() : std::filesystem::path(value);
}

std::uint32_t environment_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) { return fallback; }
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) { return fallback; }
    return static_cast<std::uint32_t>(parsed);
}

ninfer::EngineOptions base_options(const std::filesystem::path& artifact,
                                   ninfer::KvCacheStorage storage) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk = 1024;
    options.kv_cache      = storage;
    // The 27B target has no DFlash graph profiles yet; both arms run eager so the
    // comparison isolates speculation rather than graph capture.
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::PromptInput conversation(std::string text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage user;
    user.role = "user";
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

// One prompt from each family the throughput suite measures, so a divergence that
// only shows up on one kind of continuation is still caught.
const std::vector<std::string>& prompts() {
    static const std::vector<std::string> value{
        "Natalia sold clips to 48 friends in April and half as many in May. How many in total?",
        "Write a Python function that checks whether a list has two numbers closer than a "
        "threshold.",
        "Describe two cultural experiences worth seeking out on a trip to Hawaii.",
    };
    return value;
}

} // namespace

int main() {
    const std::filesystem::path target = environment_path("NINFER_QWEN3_8_27B_WEIGHTS");
    const std::filesystem::path drafter = environment_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS");
    if (!std::filesystem::is_regular_file(target) || !std::filesystem::is_regular_file(drafter)) {
        std::cerr << "skip: set NINFER_QWEN3_8_27B_WEIGHTS and "
                     "NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS to real artifacts\n";
        return 77;
    }

    const std::uint32_t outputs = environment_u32("NINFER_DFLASH2_REAL_OUTPUTS", 64);
    const std::uint32_t draft_tokens = environment_u32("NINFER_DFLASH2_REAL_DRAFT_TOKENS", 7);
    const std::uint32_t prompt_filter = environment_u32(
        "NINFER_DFLASH2_REAL_PROMPT", std::numeric_limits<std::uint32_t>::max());
    int failures = 0;
    const std::string storage_filter =
        std::getenv("NINFER_DFLASH2_REAL_KV") == nullptr
            ? "both"
            : std::getenv("NINFER_DFLASH2_REAL_KV");
    for (const ninfer::KvCacheStorage storage :
         {ninfer::KvCacheStorage::Int8Group64, ninfer::KvCacheStorage::TurboQuant}) {
        if ((storage_filter == "int8" && storage != ninfer::KvCacheStorage::Int8Group64) ||
            (storage_filter == "turboquant" &&
             storage != ninfer::KvCacheStorage::TurboQuant)) {
            continue;
        }
        std::vector<std::vector<ninfer::TokenId>> baseline;
        baseline.reserve(prompts().size());
        {
            ninfer::Engine engine(base_options(target, storage));
            for (std::size_t index = 0; index < prompts().size(); ++index) {
                if (prompt_filter != std::numeric_limits<std::uint32_t>::max() &&
                    index != prompt_filter) {
                    baseline.emplace_back();
                    continue;
                }
                const std::string& prompt = prompts()[index];
                baseline.push_back(engine.generate(engine.prepare(conversation(prompt)),
                                                   greedy(outputs))
                                       .generated_token_ids);
            }
        }

        ninfer::EngineOptions speculative = base_options(target, storage);
        speculative.draft_artifact_path   = drafter;
        speculative.speculative.backend   = ninfer::SpeculativeBackend::DFlash;
        speculative.speculative.draft_tokens = draft_tokens;

        ninfer::Engine speculative_engine(speculative);
        for (std::size_t index = 0; index < prompts().size(); ++index) {
            if (prompt_filter != std::numeric_limits<std::uint32_t>::max() &&
                index != prompt_filter) {
                continue;
            }
            const ninfer::GenerationResult result = speculative_engine.generate(
                speculative_engine.prepare(conversation(prompts()[index])), greedy(outputs));
            if (result.generated_token_ids != baseline[index]) {
                std::size_t first = 0;
                while (first < result.generated_token_ids.size() &&
                       first < baseline[index].size() &&
                       result.generated_token_ids[first] == baseline[index][first]) {
                    ++first;
                }
                std::cerr << "DFlash 2 greedy output diverged from the target on prompt " << index
                          << " with KV storage " << static_cast<int>(storage)
                          << " draft_tokens=" << draft_tokens << " first=" << first
                          << " baseline_size=" << baseline[index].size()
                          << " speculative_size=" << result.generated_token_ids.size();
                if (first < baseline[index].size()) {
                    std::cerr << " baseline_token=" << baseline[index][first];
                }
                if (first < result.generated_token_ids.size()) {
                    std::cerr << " speculative_token=" << result.generated_token_ids[first];
                }
                std::cerr << " rounds=" << result.speculative.rounds
                          << " accepted=" << result.speculative.accepted_tokens << '\n';
                ++failures;
                continue;
            }
            if (result.speculative.rounds == 0) {
                std::cerr << "DFlash 2 produced no speculative rounds on prompt " << index
                          << " with KV storage " << static_cast<int>(storage)
                          << "; the comparison would pass trivially\n";
                ++failures;
            }
        }
    }
    if (failures != 0) {
        std::cerr << "FAIL DFlash 2 losslessness\n";
        return 1;
    }
    std::cout << "PASS DFlash 2 greedy output is identical to the target for INT8 and "
                 "TurboQuant KV\n";
    return 0;
}
