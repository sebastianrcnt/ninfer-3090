#include "targets/qwen3_6/impl/frontend/charset_policy.h"

namespace ninfer::targets::qwen3_6::charset_policy {
namespace {

constexpr bool in(std::uint32_t value, std::uint32_t first, std::uint32_t last) noexcept {
    return value >= first && value <= last;
}

} // namespace

bool permits_codepoint(std::uint32_t c) noexcept {
    // Keep this list in sync with /home/coolguy/.config/llama.cpp/no-hanja.gbnf.
    if (c == 0x00C7 || c == 0x00D6 || c == 0x00DC || c == 0x00E7 || c == 0x00F6 ||
        c == 0x00FC || in(c, 0x011E, 0x011F) || in(c, 0x0130, 0x0131) ||
        in(c, 0x015E, 0x015F)) {
        return false;
    }
    return !(in(c, 0x0400, 0x052F) || in(c, 0x1C80, 0x1C8F) || in(c, 0x2DE0, 0x2DFF) ||
             in(c, 0x3040, 0x30FF) || in(c, 0x31F0, 0x31FF) || in(c, 0x3400, 0x4DBF) ||
             in(c, 0xA640, 0xA69F) || in(c, 0xF900, 0xFAFF) || in(c, 0xFE2E, 0xFE2F) ||
             in(c, 0xFF65, 0xFF9F) || in(c, 0x4E00, 0x9FFF) || in(c, 0x1AFF0, 0x1AFFF) ||
             in(c, 0x1B000, 0x1B12F) || in(c, 0x20000, 0x2A6DF) ||
             in(c, 0x2A700, 0x2EE5F) || in(c, 0x2F800, 0x2FA1F) || in(c, 0x30000, 0x323AF));
}

bool permits_utf8(std::string_view text) noexcept {
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i++]);
        std::uint32_t c = 0;
        int extra = 0;
        if (lead < 0x80) { c = lead; }
        else if ((lead & 0xE0) == 0xC0) { c = lead & 0x1F; extra = 1; }
        else if ((lead & 0xF0) == 0xE0) { c = lead & 0x0F; extra = 2; }
        else if ((lead & 0xF8) == 0xF0) { c = lead & 0x07; extra = 3; }
        else { return false; }
        if (i + static_cast<std::size_t>(extra) > text.size()) { return false; }
        for (int j = 0; j < extra; ++j) {
            const unsigned char b = static_cast<unsigned char>(text[i++]);
            if ((b & 0xC0) != 0x80) { return false; }
            c = (c << 6) | (b & 0x3F);
        }
        if ((extra == 1 && c < 0x80) || (extra == 2 && c < 0x800) ||
            (extra == 3 && (c < 0x10000 || c > 0x10FFFF)) || in(c, 0xD800, 0xDFFF) ||
            !permits_codepoint(c)) {
            return false;
        }
    }
    return true;
}

std::uint32_t transition_utf8(std::uint32_t state, std::uint8_t byte) noexcept {
    if (state == kUtf8RejectedState) { return state; }
    const std::uint32_t remaining = state >> 24;
    const std::uint32_t width = (state >> 22) & 0x3U;
    std::uint32_t value = state & 0x003fffffU;
    if (remaining == 0) {
        if (byte < 0x80) { return permits_codepoint(byte) ? 0U : kUtf8RejectedState; }
        if (byte >= 0xc2 && byte <= 0xdf) { return (1U << 24) | (1U << 22) | (byte & 0x1fU); }
        if (byte >= 0xe0 && byte <= 0xef) { return (2U << 24) | (2U << 22) | (byte & 0x0fU); }
        if (byte >= 0xf0 && byte <= 0xf4) { return (3U << 24) | (3U << 22) | (byte & 0x07U); }
        return kUtf8RejectedState;
    }
    if ((byte & 0xc0U) != 0x80U) { return kUtf8RejectedState; }
    value = (value << 6) | (byte & 0x3fU);
    const std::uint32_t next = remaining - 1U;
    if (next != 0) { return (next << 24) | (width << 22) | value; }
    const std::uint32_t minimum = width == 1 ? 0x80U : width == 2 ? 0x800U : 0x10000U;
    return value < minimum || value > 0x10ffffU || in(value, 0xd800, 0xdfff) ||
                   !permits_codepoint(value)
               ? kUtf8RejectedState
               : 0U;
}

std::uint32_t transition_token(std::uint32_t state, std::string_view bytes) noexcept {
    for (const unsigned char byte : bytes) {
        state = transition_utf8(state, byte);
        if (state == kUtf8RejectedState) { break; }
    }
    return state;
}

} // namespace ninfer::targets::qwen3_6::charset_policy
