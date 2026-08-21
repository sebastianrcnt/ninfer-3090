#include "runtime/cache/conversation_snapshot.h"

#include <algorithm>

namespace ninfer::runtime {

void HostKvPayload::truncate(std::uint32_t pages) {
    for (HostPagePlane& plane : planes) {
        if (plane.pages.size() > pages) { plane.pages.resize(pages); }
    }
}

bool HostKvPayload::consistent() const noexcept {
    if (planes.empty()) { return true; }
    const std::size_t pages = planes.front().pages.size();
    return std::all_of(planes.begin(), planes.end(), [pages](const HostPagePlane& plane) {
        return plane.group_bytes != 0 && plane.pages.size() == pages;
    });
}

std::uint64_t ConversationSnapshot::state_bytes() const noexcept {
    std::uint64_t total = ledger.size() * sizeof(TokenId) + identity.size();
    for (const ConversationCheckpoint& checkpoint : checkpoints) {
        total += checkpoint.state.size();
    }
    return total;
}

std::uint32_t conversation_pages_for_tokens(std::uint32_t tokens,
                                            std::uint32_t page_size) noexcept {
    if (page_size == 0) { return 0; }
    return (tokens + page_size - 1U) / page_size;
}

} // namespace ninfer::runtime
