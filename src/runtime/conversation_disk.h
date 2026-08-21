#pragma once

// ninfer::runtime - the durable tier of the conversation checkpoint cache.
//
// NINFSLOT v4 stores one conversation per file: the shared packed KV payload once, then an
// ordered set of checkpoints whose fixed-size states follow. The identity block is inherited
// from v3 verbatim — a file written by a different artifact, KV format, or speculative backend
// is refused before any device memory is touched. Writes go through a sibling temporary and an
// atomic rename; the previous valid snapshot survives a crash mid-write.

#include "runtime/conversation_cache.h"

#include <condition_variable>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::runtime {

inline constexpr std::uint32_t kConversationFileFormat = 4;

// Fixed-size head of one checkpoint's on-disk record. Field order is part of the format.
struct CheckpointFileHeader {
    std::uint8_t kind = 0;
    std::uint32_t frontier = 0;
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier = 0;
    std::uint32_t text_kv_valid = 0;
    std::uint32_t mtp_kv_valid = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::int32_t rope_delta = 0;
    std::uint32_t mtp_draft_count = 0;
    std::uint8_t tail_hidden_valid = 0;
    std::uint8_t reserved[3] = {};
    std::uint64_t prefix_identity_bytes = 0;
    std::uint64_t mtp_drafts_bytes = 0;
    std::uint64_t tail_hidden_bytes = 0;
    std::uint64_t turn_checkpoint_hidden_bytes = 0;
    std::uint64_t linear_conv_bytes = 0;
    std::uint64_t linear_recurrent_bytes = 0;
};

static_assert(sizeof(CheckpointFileHeader) % 8 == 0,
              "CheckpointFileHeader must stay 8-byte aligned");

// A conversation snapshot as it travels to or from disk. Payload pages are shared with the RAM
// catalog on write and owned solely by the loader on read.
struct ConversationSnapshot {
    std::vector<TokenId> ledger;
    std::vector<std::byte> prefix_identity;
    ConversationKvPayload payload;
    std::vector<ConversationCheckpoint> checkpoints;

    [[nodiscard]] std::size_t bytes() const noexcept {
        return payload.payload_bytes() + prefix_identity.size();
    }
};

// What startup knows without reading payloads: which conversations exist and their frontiers.
struct DiskConversationEntry {
    std::string name;
    std::vector<std::uint32_t> checkpoint_frontiers;
    std::uint64_t file_bytes = 0;
};

class ConversationDiskCache {
public:
    ConversationDiskCache(std::filesystem::path directory, std::size_t budget_bytes,
                          std::string identity);
    ~ConversationDiskCache() noexcept;

    ConversationDiskCache(const ConversationDiskCache&)            = delete;
    ConversationDiskCache& operator=(const ConversationDiskCache&) = delete;

    // Scans the directory and validates headers only. Call once after construction; files that
    // fail validation are left in place but excluded from the catalog.
    void scan();

    [[nodiscard]] const std::map<std::string, DiskConversationEntry>& catalog() const noexcept {
        return catalog_;
    }

    // Queues one snapshot for durable write under the given conversation name. At most one
    // in-flight write and one coalesced newest pending job per name. The snapshot is consumed
    // by value; callers hand over shared page ownership rather than copying payload bytes.
    void schedule(std::string name, ConversationSnapshot snapshot);

    // Reads one conversation fully. Throws when the file is missing or fails validation.
    [[nodiscard]] ConversationSnapshot load(const std::string& name) const;

    // Drops the durable copy of one conversation.
    void erase(const std::string& name);

private:
    void worker_loop();
    void enforce_budget_locked(const std::string& just_written);
    [[nodiscard]] std::filesystem::path path_for(const std::string& name) const;

    std::filesystem::path directory_;
    std::size_t budget_bytes_ = 0;
    std::string identity_;
    std::map<std::string, DiskConversationEntry> catalog_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::map<std::string, ConversationSnapshot> pending_; // coalesced per conversation
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
