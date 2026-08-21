#pragma once

// ninfer::runtime - tiered conversation checkpoint cache.
//
// Ownership split: the target family owns checkpoint semantics, the capture/restore state
// transaction, and exact prepared-prefix divergence selection. Everything here is target-agnostic
// host bookkeeping - the conversation catalog, the global RAM and disk byte budgets, LRU policy,
// and the asynchronous disk worker.
//
// A lane is an execution resource, so no conversation is identified by a lane or by a client
// session id. `select` hands each candidate conversation to a family-provided selector and keeps
// the greatest exactly-matching frontier across the RAM tier first and the disk tier second.
//
// Tier roles follow the measured bandwidths on this host: RAM is the interactive tier (a full
// snapshot moves in a fraction of a second), while local disk is the restart-recovery and
// overflow tier (tens of seconds for a full snapshot). Nothing here reads a payload from disk
// until a prompt actually matches its catalog entry.

#include "runtime/cache/conversation_file.h"
#include "runtime/cache/conversation_snapshot.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

struct ConversationCachePolicy {
    // Global hot-cache payload budget across every cached conversation. Zero disables automatic
    // multi-conversation caching entirely.
    std::uint64_t ram_budget_bytes = 0;
    // Durable snapshot directory. Empty disables the disk tier.
    std::filesystem::path disk_dir;
    std::uint64_t disk_budget_bytes    = 0;
    std::uint32_t context_checkpoints  = 32;
    std::uint32_t checkpoint_min_step  = 8192;

    [[nodiscard]] bool ram_enabled() const noexcept { return ram_budget_bytes != 0; }
    [[nodiscard]] bool disk_enabled() const noexcept { return !disk_dir.empty(); }
};

struct ConversationCacheStats {
    std::uint32_t conversations         = 0;
    std::uint32_t resident_conversations = 0;
    std::uint32_t checkpoints           = 0;
    std::uint64_t resident_bytes        = 0;
    std::uint64_t disk_bytes            = 0;
    std::uint32_t pending_writes        = 0;
};

class ConversationCache {
public:
    using EntryId = std::uint64_t; // 0 means "no conversation"

    // Family-provided exact divergence selection over one candidate conversation. Returns the
    // index of the greatest checkpoint whose coverage the prompt still matches token for token,
    // or nullopt when the conversation is unusable for this prompt.
    using Selector = std::function<std::optional<std::size_t>(
        const std::vector<TokenId>& ledger, const std::vector<std::byte>& identity,
        const std::vector<ConversationCheckpoint>& checkpoints)>;

    struct Match {
        EntryId entry          = 0;
        std::size_t checkpoint = 0;
        std::uint32_t frontier = 0;
        bool from_disk         = false;
    };

    ConversationCache(ConversationCachePolicy options, ConversationGeometry geometry);
    ~ConversationCache();

    ConversationCache(const ConversationCache&)            = delete;
    ConversationCache& operator=(const ConversationCache&) = delete;

    [[nodiscard]] const ConversationCachePolicy& options() const noexcept { return options_; }

    // Adopts the durable catalog. Reads headers, ledgers, identities, and checkpoint records
    // only; payloads stay on disk. Files this build cannot use are removed, because the directory
    // is this server's own bounded cache. Returns the number of conversations adopted.
    std::uint32_t adopt_disk_catalog(const std::function<void(const std::string&)>& report);

    [[nodiscard]] std::optional<Match> select(const Selector& selector);

    // Brings a matched conversation into RAM when it is disk-resident and returns its payload.
    [[nodiscard]] std::shared_ptr<const ConversationSnapshot> acquire(const Match& match);

    // Merges one lane capture. `lane_entry` is the conversation the lane was continuing, or 0.
    // A capture whose ledger extends that conversation appends to it; a capture that diverges
    // becomes a branch sharing the parent's host pages copy-on-write. Returns the entry the lane
    // now belongs to, or 0 when caching is disabled.
    EntryId park(EntryId lane_entry, ConversationCapture&& capture);

    // A conversation a lane currently holds is never evicted.
    void pin(EntryId entry);
    void unpin(EntryId entry);

    void touch(EntryId entry);

    [[nodiscard]] ConversationCacheStats stats() const;

private:
    struct Entry {
        EntryId id = 0;
        std::shared_ptr<const ConversationSnapshot> snapshot; // null while disk-only
        std::optional<ConversationFileCatalog> disk;
        std::filesystem::path file;
        std::uint64_t file_bytes  = 0;
        std::uint64_t resident_bytes = 0;
        std::uint64_t last_use    = 0;
        std::uint32_t pins        = 0;
    };

    [[nodiscard]] Entry* find(EntryId id) noexcept;
    [[nodiscard]] const Entry* find(EntryId id) const noexcept;

    void adopt_pages(const HostKvPayload& payload);
    void release_pages(const HostKvPayload& payload);
    void account_resident(Entry& entry, const ConversationSnapshot& snapshot);
    void release_resident(Entry& entry);

    void trim_checkpoints(Entry& entry);
    void enforce_ram_budget();
    void enforce_disk_budget(std::uint64_t incoming_bytes);
    void erase_entry(EntryId id);

    [[nodiscard]] std::shared_ptr<const ConversationSnapshot>
    merged(const Entry* parent, ConversationCapture&& capture, bool branch) const;

    void schedule_write(EntryId id, std::shared_ptr<const ConversationSnapshot> snapshot);
    [[nodiscard]] ConversationFileCatalog
    catalog_of(const std::filesystem::path& path, const ConversationSnapshot& snapshot,
               std::uint64_t file_bytes) const;
    void writer_loop();

    ConversationCachePolicy options_;
    ConversationGeometry geometry_;

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    EntryId next_id_        = 1;
    std::uint64_t clock_    = 1;
    std::uint64_t resident_bytes_ = 0;
    std::uint64_t disk_bytes_     = 0;
    std::uint32_t next_file_serial_ = 0;
    // Distinct-page refcounts, so a page shared by several checkpoints or branches is charged to
    // the global RAM budget exactly once.
    std::unordered_map<const void*, std::uint32_t> page_refs_;

    // Disk worker: at most one write in flight and one coalesced newest snapshot per
    // conversation, so a fast conversation cannot queue an unbounded backlog of payloads.
    mutable std::mutex writer_mutex_;
    std::condition_variable writer_cv_;
    std::deque<EntryId> write_order_;
    std::unordered_map<EntryId, std::shared_ptr<const ConversationSnapshot>> pending_;
    std::unordered_map<EntryId, std::filesystem::path> write_paths_;
    bool writer_stopping_ = false;
    std::thread writer_;
};

} // namespace ninfer::runtime
