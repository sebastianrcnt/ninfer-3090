#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::targets::qwen3_6::charset_policy {

// The server-side policy equivalent of the deployed llama.cpp no-hanja.gbnf.
// It intentionally permits Korean, Latin, punctuation, and emoji while rejecting
// Han ideographs, Japanese kana, Cyrillic, and the Turkish-only letters listed in
// that grammar.
[[nodiscard]] bool permits_codepoint(std::uint32_t codepoint) noexcept;

// A complete UTF-8 string is permitted only when every scalar is permitted.
// Invalid UTF-8 is rejected: callers which operate on tokenizer byte fragments
// must retain their fragment state rather than treating an incomplete sequence as
// a character.
[[nodiscard]] bool permits_utf8(std::string_view text) noexcept;

inline constexpr std::uint32_t kUtf8InitialState = 0;
inline constexpr std::uint32_t kUtf8RejectedState = 0xffffffffU;

// Packed UTF-8 decoder state. It is deliberately compact enough to live one-per
// active lane on the device and is shared by ordinary and speculative sampling.
[[nodiscard]] std::uint32_t transition_utf8(std::uint32_t state, std::uint8_t byte) noexcept;
[[nodiscard]] std::uint32_t transition_token(std::uint32_t state, std::string_view bytes) noexcept;

} // namespace ninfer::targets::qwen3_6::charset_policy
