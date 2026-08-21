#pragma once

// ninfer::runtime - the tiered conversation checkpoint cache.
//
// A lane is an execution resource; a conversation is the cached unit. The active conversation
// keeps its live state and newest turn checkpoint on GPU. Displaced conversations and older
// checkpoints live here, in bounded host RAM, and optionally on disk behind the async writer.
// Ownership split: this file owns the catalog, byte budgets, LRU order, and payload bytes; the
// qwen3.6 family runtime owns what the bytes mean (capture/restore transactions, prefix matching,
// speculative state). No CUDA types may appear here.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ninfer/types.h>

namespace ninfer::runtime {

// Why the checkpoint exists; restore uses it to decide whether the lane's GPU turn-checkpoint
// slot is repopulated or invalidated.
enum class CheckpointKind : std::uint8_t {
    TurnBoundary,
    PeriodicGrid,
};

// One restorable point of one conversation, resident in host RAM. Fixed-size state plus the
// small host vectors; the KV pages are shared per conversation and live beside these.
struct ConversationCheckpoint {
    CheckpointKind kind = CheckpointKind::TurnBoundary;
    std::uint32_t frontier = 0;

    // Sequence continuation at the checkpoint, verbatim from the family state.
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier = 0;
    std::uint32_t text_kv_valid = 0;
    std::uint32_t mtp_kv_valid = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::int32_t rope_delta = 0;
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid = false;

    // Exact prepared-prefix identity truncated to the checkpoint frontier, serialized. Matching
    // compares against this blob, never against the conversation's newer ledger tail.
    std::vector<std::byte> prefix_identity;

    // Speculative drafts and hidden vectors, raw device images.
    std::vector<std::byte> mtp_drafts;
    std::vector<std::byte> tail_hidden;
    std::vector<std::byte> turn_checkpoint_hidden;

    // GDN conv/recurrent slices for one slot image, layer-major.
    std::vector<std::byte> linear_conv;
    std::vector<std::byte> linear_recurrent;

    [[nodiscard]] std::size_t state_bytes() const noexcept {
        return prefix_identity.size() + mtp_drafts.size() + tail_hidden.size() +
               turn_checkpoint_hidden.size() + linear_conv.size() + linear_recurrent.size();
    }
};

// The shared packed KV pages of one conversation, plane-major then page-order, exactly as
// PagedKVPool::read_page_group produces them. Pages are immutable once written, so a checkpoint
// at frontier F is valid while the payload covers at least F tokens' worth of pages. Pages are
// shared-immutable: a re-park installs fresh buffers rather than mutating existing ones, which
// lets the disk writer serialize a snapshot without blocking admission.
struct ConversationKvPayload {
    using PageBytes = std::shared_ptr<const std::vector<std::byte>>;

    std::uint32_t text_plane_count = 0;
    std::uint32_t backend_plane_count = 0;
    bool has_backend_kv = false;
    std::uint64_t text_page_group_bytes = 0;
    std::uint64_t backend_page_group_bytes = 0;
    // Plane-major page groups: text_pages[plane][page].
    std::vector<std::vector<PageBytes>> text_pages;
    std::vector<std::vector<PageBytes>> backend_pages;

    [[nodiscard]] std::size_t page_count() const noexcept {
        const auto count = [](const std::vector<std::vector<PageBytes>>& planes) {
            return planes.empty() ? 0 : planes.front().size();
        };
        return count(text_pages) + (has_backend_kv ? count(backend_pages) : 0);
    }

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& plane : text_pages) {
            for (const auto& page : plane) { total += page->size(); }
        }
        for (const auto& plane : backend_pages) {
            for (const auto& page : plane) { total += page->size(); }
        }
        return total;
    }
};

struct ConversationRecord;

// A stable handle handed out by the catalog. The id is engine-lifetime only; disk naming maps it
// through the catalog, not through this value's stability across restarts.
using ConversationId = std::uint64_t;

struct ConversationMatch {
    ConversationId id = 0;
    std::size_t checkpoint_index = 0;
    std::uint32_t frontier = 0;
};

// LRU bookkeeping is the catalog's own; entries move to the back on touch.
using ConversationList = std::list<ConversationRecord>;

struct ConversationRecord {
    ConversationId id = 0;
    std::string name; // stable disk name assigned at creation.
    std::uint64_t last_touch_tick = 0;

    // Newest conversation-wide continuation. Append-only while the conversation extends;
    // checkpoints reference frontiers inside this coverage.
    std::vector<TokenId> ledger;
    std::vector<std::byte> prefix_identity;

    ConversationKvPayload payload;
    std::vector<ConversationCheckpoint> checkpoints;

    // Highest page count the payload currently covers; append-only re-parks copy only pages
    // past this mark.
    std::size_t parked_text_pages = 0;
    std::size_t parked_backend_pages = 0;
};

// Global byte budget across every cached conversation. Eviction order when over budget:
// redundant/nearby historical checkpoints first, then least-recently-touched historical
// checkpoints, then the least-recently-touched inactive conversation. The caller guarantees the
// active lane state never enters this catalog.
class ConversationCache {
public:
    struct Options {
        std::size_t ram_budget_bytes = 0; // 0 disables automatic multi-conversation caching.
        std::size_t max_checkpoints_per_conversation = 32;
        std::size_t min_checkpoint_spacing_tokens = 8192;
    };

    ConversationCache() = default;
    explicit ConversationCache(Options options) : options_(options) {}

    // The Engine owns the cache lifetime and configures it once, after construction.
    void reconfigure(const Options& options) { options_ = options; }

    [[nodiscard]] const Options& options() const noexcept { return options_; }
    [[nodiscard]] bool enabled() const noexcept { return options_.ram_budget_bytes != 0; }
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] std::uint64_t tick() const noexcept { return tick_; }

    // Creates a conversation and returns its record. The caller populates payload and the first
    // checkpoint before admitting traffic against it.
    ConversationRecord& create(std::string name);

    // Exact-match lookup by catalog id.
    [[nodiscard]] ConversationRecord* find(ConversationId id);
    [[nodiscard]] const ConversationRecord* find(ConversationId id) const;

    // Touches LRU recency without mutating contents.
    void touch(ConversationRecord& record);

    // Drops checkpoints that carry no additional coverage: same-frontier duplicates and grid
    // points closer than the spacing to their retained neighbour. Returns evicted count.
    std::size_t prune_redundant_checkpoints(ConversationRecord& record);

    // Enforces the global RAM budget using the eviction order above. Returns freed bytes.
    std::size_t enforce_ram_budget();

    void erase(ConversationId id);

    [[nodiscard]] ConversationList& records() noexcept { return records_; }
    [[nodiscard]] const ConversationList& records() const noexcept { return records_; }

private:
    [[nodiscard]] std::size_t total_bytes() const noexcept;

    Options options_;
    ConversationList records_;
    ConversationId next_id_ = 1;
    std::uint64_t tick_ = 0;
};

} // namespace ninfer::runtime
