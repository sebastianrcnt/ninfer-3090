#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    // Small enough for CI-class GPUs; the payloads are tiny at this context size.
    options.conversation_cache_ram_bytes = 256ULL << 20;
    options.speculative.backend          = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens     = 3;
    options.speculative.proposal_head    = ninfer::ProposalHead::Optimized;
    return options;
}

std::vector<ninfer::TokenId> conversation_prompt(ninfer::TokenId seed) {
    return {248045, 846, 198, 5834, seed, 248046, 198};
}

std::vector<ninfer::TokenId> resume_prompt(const std::vector<ninfer::TokenId>& prompt,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), generated.begin(), generated.end());
    continuation.push_back(198);
    return continuation;
}

ninfer::RequestOptions generate_options(std::uint32_t tokens, bool reuse) {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens = tokens;
    result.execution.sampling.temperature    = 0.0F;
    result.execution.allow_prefix_reuse      = reuse;
    result.stop.include_model_defaults       = false;
    return result;
}

// Runs one turn and returns its generated tokens; greedy throughout.
std::vector<ninfer::TokenId> run_turn(ninfer::Engine& engine,
                                      const std::vector<ninfer::TokenId>& prompt) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), generate_options(6, true));
    if (result.generated_token_ids.size() != 6 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "turn did not produce six tokens\n";
        return {};
    }
    return result.generated_token_ids;
}

// Alternates two conversations through one lane. The second conversation displaces the first
// from the only lane; resuming the first must restore from the cache instead of resetting.
int exercise_alternating_conversations(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt_a = conversation_prompt(9001);
    const std::vector<ninfer::TokenId> prompt_b = conversation_prompt(4242);

    const auto turn_a1 = run_turn(engine, prompt_a);
    if (turn_a1.empty()) { return 1; }
    const auto turn_b1 = run_turn(engine, prompt_b);
    if (turn_b1.empty()) { return 1; }

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt_a.size() + turn_a1.size() - 1);
    const ninfer::GenerationResult resumed = engine.generate(
        engine.prepare_tokens(resume_prompt(prompt_a, turn_a1)), generate_options(6, true));
    if (resumed.reused_prompt_tokens != expected_reuse) {
        std::cerr << "alternating resume reused " << resumed.reused_prompt_tokens
                  << " tokens, expected the cached frontier " << expected_reuse << '\n';
        return 1;
    }
    if (resumed.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier) {
        std::cerr << "cached restore did not continue as an append\n";
        return 1;
    }

    // Losslessness: the same resume without any reuse must produce identical greedy tokens.
    ninfer::RequestOptions baseline_options = generate_options(6, false);
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(resume_prompt(prompt_a, turn_a1)),
                        baseline_options);
    if (baseline.generated_token_ids != resumed.generated_token_ids) {
        std::cerr << "cache-restored prefill changed greedy output\n";
        return 1;
    }

    bool listed_a = false;
    for (const ninfer::ConversationSummary& conversation : engine.list_conversations()) {
        listed_a = listed_a || conversation.tokens >= expected_reuse;
    }
    if (!listed_a) {
        std::cerr << "completed conversation missing from the catalog listing\n";
        return 1;
    }

    bool erased = false;
    for (const ninfer::ConversationSummary& conversation : engine.list_conversations()) {
        erased = engine.erase_conversation(conversation.id);
    }
    if (!engine.list_conversations().empty() && !erased) {
        std::cerr << "erase_conversation did not clear the catalog\n";
        return 1;
    }
    return 0;
}

int exercise_durable_tier(const char* artifact, const std::filesystem::path& directory) {
    const std::vector<ninfer::TokenId> prompt = conversation_prompt(777);
    std::vector<ninfer::TokenId> turn;
    {
        ninfer::EngineOptions options            = engine_options(artifact);
        options.conversation_cache_dir           = directory;
        options.conversation_cache_disk_bytes    = 512ULL << 20;
        ninfer::Engine engine(options);
        turn                                     = run_turn(engine, prompt);
        if (turn.empty()) { return 1; }
        // The writer is asynchronous: wait briefly for the snapshot to land.
        for (int i = 0; i < 100 && std::filesystem::is_empty(directory); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (std::filesystem::is_empty(directory)) {
            std::cerr << "durable tier wrote no snapshot file\n";
            return 1;
        }
    } // Engine destroyed: RAM tier gone, only the disk tier remains.

    ninfer::EngineOptions options         = engine_options(artifact);
    options.conversation_cache_dir        = directory;
    options.conversation_cache_disk_bytes = 512ULL << 20;
    ninfer::Engine engine(options);
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + turn.size() - 1);
    const ninfer::GenerationResult resumed = engine.generate(
        engine.prepare_tokens(resume_prompt(prompt, turn)), generate_options(6, true));
    if (resumed.reused_prompt_tokens != expected_reuse) {
        std::cerr << "post-restart resume reused " << resumed.reused_prompt_tokens
                  << " tokens, expected the durable frontier " << expected_reuse << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS"); }
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: neither NINFER_QWEN3_8_27B_WEIGHTS nor NINFER_QWEN3_6_27B_WEIGHTS "
                     "is set\n";
        return 77;
    }

    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_alternating_conversations(engine); result != 0) {
            return result;
        }
    }

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ninfer-conversation-real-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    if (const int result = exercise_durable_tier(artifact, directory); result != 0) {
        std::filesystem::remove_all(directory);
        return result;
    }
    std::filesystem::remove_all(directory);

    std::cout << "ok\n";
    return 0;
}
