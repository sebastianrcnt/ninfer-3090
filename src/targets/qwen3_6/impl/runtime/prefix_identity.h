#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;

    [[nodiscard]] bool carries_vision() const noexcept { return !vision_items_.empty(); }

    // Round-trips the private representation for slot persistence, vision items included: they
    // take part in matches(), so an identity persisted without them would let a restored prefix
    // compare equal to a prompt carrying different media. deserialize() throws on truncated or
    // inconsistent input and leaves the object cleared.
    [[nodiscard]] std::vector<std::byte> serialize() const;
    void deserialize(std::span<const std::byte> bytes);

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

} // namespace ninfer::targets::qwen3_6::detail
