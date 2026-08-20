#include "targets/qwen3_6/impl/frontend/charset_policy.h"

#include <cassert>
#include <string_view>

namespace policy = ninfer::targets::qwen3_6::charset_policy;

int main() {
    assert(policy::permits_utf8("plain ASCII, 한국어, and emoji \xF0\x9F\x98\x80"));
    assert(!policy::permits_utf8("\xE4\xB8\xAD"));       // Han: 中
    assert(!policy::permits_utf8("\xE3\x81\x82"));       // Kana: あ
    assert(!policy::permits_utf8("\xD0\x96"));             // Cyrillic: Ж
    assert(!policy::permits_utf8("\xC3\xB6"));             // Turkish list: ö

    std::uint32_t state = policy::transition_token(0, "\xE4");
    assert(state != 0 && state != policy::kUtf8RejectedState);
    assert(policy::transition_token(state, "\xB8\xAD") == policy::kUtf8RejectedState);

    state = policy::transition_token(0, "\xEA");
    assert(state != 0 && state != policy::kUtf8RejectedState);
    state = policy::transition_token(state, "\xB0\x80"); // 가 split across token bytes
    assert(state == 0);

    assert(policy::transition_token(0, "\xE0\x80\x80") == policy::kUtf8RejectedState);
    assert(policy::transition_token(0, "\xF4\x90\x80\x80") == policy::kUtf8RejectedState);
    return 0;
}
