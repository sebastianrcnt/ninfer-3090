#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    // Invoked by wait()'s consumer thread after admission selects the request's lane and prefix
    // reuse path. If admission has already happened, the next wait() iteration invokes it.
    void set_plan_callback(std::function<void(GenerationPlan)> callback);

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// Observable state of the tiered conversation checkpoint cache.
struct ConversationCacheSummary {
    std::uint32_t conversations           = 0;
    std::uint32_t resident_conversations  = 0;
    std::uint32_t checkpoints             = 0;
    std::uint64_t resident_bytes          = 0;
    std::uint64_t disk_bytes              = 0;
    std::uint32_t pending_writes          = 0;
    std::uint32_t adopted_at_startup      = 0;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {},
           HostInputLease host_input                              = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    void reset_memory_peaks() noexcept;

    // Execution lanes, reported by llama.cpp's read-only /slots surface. A slot is a lane, never
    // a conversation: conversation state is owned by the checkpoint cache below and moves between
    // lanes, RAM, and local disk without a client naming it.
    [[nodiscard]] std::uint32_t slot_count() const;
    [[nodiscard]] std::vector<std::uint32_t> slot_token_counts() const;
    [[nodiscard]] ConversationCacheSummary conversation_cache_summary() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
