#pragma once

// ninfer::runtime - host-resident conversation state.
//
// A lane is a GPU execution resource, not a conversation. One conversation is represented here by
// its exact token ledger and prepared-prefix identity plus an ordered set of checkpoints taken
// over one shared packed-KV payload. No client session id takes part: two prompts belong to the
// same cached conversation exactly when their prepared prefixes agree token for token.
//
// The packed KV payload is stored once per conversation. A checkpoint names how much of that
// payload it covers and adds only its own fixed-size frontier state, so retaining 32 historical
// boundaries costs 32 small state blobs rather than 32 copies of a multi-gigabyte cache. Pages
// below every live frontier can never be rewritten, which is what makes both the sharing between
// a conversation's checkpoints and the copy-on-write sharing between branches sound.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::runtime {

// One packed KV page group, copied out of the device pool exactly once and immutable afterwards.
using HostPage = std::shared_ptr<const std::vector<std::byte>>;

struct HostPagePlane {
    std::uint64_t group_bytes = 0;
    std::vector<HostPage> pages;
};

// Planes are the pool's own storage planes; every plane holds the same page count, because a
// logical page group spans all of them.
struct HostKvPayload {
    std::vector<HostPagePlane> planes;

    [[nodiscard]] std::uint32_t page_count() const noexcept {
        return planes.empty() ? 0U : static_cast<std::uint32_t>(planes.front().pages.size());
    }

    void truncate(std::uint32_t pages);
    [[nodiscard]] bool consistent() const noexcept;
};

// Everything a restore must agree with before device memory is touched. A snapshot written by a
// different artifact, KV format, or speculative backend would otherwise be reinterpreted as valid
// attention state, so the comparison is exhaustive rather than advisory.
struct ConversationGeometry {
    std::uint64_t model_id_hash        = 0;
    std::uint32_t kv_dtype             = 0;
    std::uint32_t kv_storage           = 0;
    std::int32_t kv_quant_group        = 0;
    std::uint32_t speculative_backend  = 0;
    std::uint32_t kv_capacity          = 0;
    std::uint32_t page_size            = 0;
    std::uint32_t text_plane_count     = 0;
    std::uint32_t backend_plane_count  = 0;
    std::uint32_t checkpoint_state_bytes = 0;
    std::uint8_t kv_packed_v           = 0;
    std::uint8_t kv_rotate_k           = 0;
    std::uint8_t kv_rotate_v           = 0;
    std::uint8_t has_backend_kv        = 0;

    friend bool operator==(const ConversationGeometry&,
                           const ConversationGeometry&) noexcept = default;
};

// One retained boundary. `frontier` is the executed-token count the checkpoint continues from;
// the conversation ledger it implies is the first `frontier + 1` tokens, the extra entry being the
// token this state predicts but has not executed.
struct ConversationCheckpoint {
    std::uint32_t frontier      = 0;
    std::uint32_t text_pages    = 0;
    std::uint32_t backend_pages = 0;
    // A completed request boundary is always retained; a periodic grid point is a spacing-driven
    // extra and is the first thing dropped when the per-conversation cap binds.
    std::uint8_t turn_boundary = 0;
    // Target-opaque fixed-size continuation state: tail hidden state, GDN convolution/recurrent
    // state, the speculative frontier block, and the bookkeeping the target needs to resume.
    std::vector<std::byte> state;
};

struct ConversationSnapshot {
    ConversationGeometry geometry;
    // Ledger and identity of the newest checkpoint. Every earlier checkpoint uses the matching
    // prefix: a conversation only ever extends, and a divergent continuation becomes a branch.
    std::vector<TokenId> ledger;
    std::vector<std::byte> identity;
    HostKvPayload text;
    HostKvPayload backend;
    std::vector<ConversationCheckpoint> checkpoints; // strictly ascending frontier

    [[nodiscard]] std::uint32_t newest_frontier() const noexcept {
        return checkpoints.empty() ? 0U : checkpoints.back().frontier;
    }

    [[nodiscard]] std::uint64_t state_bytes() const noexcept;
};

// One capture handed back by the target: the lane's current boundary plus the packed pages that
// the caller did not already hold. `shared_*_pages` is where the new pages begin, so an append
// copies only what the append actually wrote.
struct ConversationCapture {
    ConversationGeometry geometry;
    std::vector<TokenId> ledger;
    std::vector<std::byte> identity;
    ConversationCheckpoint checkpoint;
    // Executed tokens the caller already holds for this conversation. Host pages below it are
    // byte-identical for every position an earlier checkpoint can read, which is what lets an
    // append reuse them and a branch share them copy-on-write.
    std::uint32_t shared_frontier      = 0;
    std::uint32_t shared_text_pages    = 0;
    std::uint32_t shared_backend_pages = 0;
    HostKvPayload new_text;    // pages [shared_text_pages, checkpoint.text_pages)
    HostKvPayload new_backend; // pages [shared_backend_pages, checkpoint.backend_pages)
};

[[nodiscard]] std::uint32_t conversation_pages_for_tokens(std::uint32_t tokens,
                                                          std::uint32_t page_size) noexcept;

} // namespace ninfer::runtime
