#include "runtime/conversation_disk.h"
#include "runtime/slot_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

namespace {

std::uint64_t identity_hash(std::string_view value) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char byte : value) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Fixed-size head of the v4 file. The identity block is byte-identical to SlotFileHeader's so
// one validation routine covers both formats.
struct ConversationFileHeader {
    char magic[8];
    std::uint32_t format;
    std::uint32_t header_bytes;

    std::uint64_t model_id_hash;
    std::uint32_t kv_dtype;
    std::uint32_t kv_storage;
    std::int32_t kv_quant_group;
    std::uint32_t speculative_backend;
    std::uint32_t kv_capacity;
    std::uint32_t page_size;
    std::uint32_t text_plane_count;
    std::uint32_t backend_plane_count;
    std::uint64_t text_page_group_bytes;
    std::uint64_t backend_page_group_bytes;
    std::uint8_t kv_packed_v;
    std::uint8_t kv_rotate_k;
    std::uint8_t kv_rotate_v;
    std::uint8_t has_backend_kv;

    std::uint32_t checkpoint_count;
    std::uint8_t reserved[4];
    std::uint64_t ledger_tokens;
    std::uint64_t prefix_identity_bytes;
    std::uint64_t parked_text_pages;
    std::uint64_t parked_backend_pages;
};

static_assert(sizeof(ConversationFileHeader) % 8 == 0,
              "ConversationFileHeader must stay 8-byte aligned");

void write_bytes(std::ofstream& out, const void* data, std::size_t bytes) {
    if (bytes == 0) { return; }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) { throw std::runtime_error("failed while writing a conversation snapshot"); }
}

void read_bytes(std::ifstream& in, void* data, std::size_t bytes, const char* what) {
    if (bytes == 0) { return; }
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) { throw std::invalid_argument(std::string("conversation file is truncated in ") + what); }
}

} // namespace

ConversationDiskCache::ConversationDiskCache(std::filesystem::path directory,
                                             std::size_t budget_bytes, std::string identity)
    : directory_(std::move(directory)), budget_bytes_(budget_bytes),
      identity_(std::move(identity)) {
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    worker_ = std::thread([this] { worker_loop(); });
}

ConversationDiskCache::~ConversationDiskCache() noexcept {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

std::filesystem::path ConversationDiskCache::path_for(const std::string& name) const {
    return directory_ / (name + ".conv");
}

void ConversationDiskCache::scan() {
    std::map<std::string, DiskConversationEntry> scanned;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (error || !entry.is_regular_file(error) || entry.path().extension() != ".conv") {
            continue;
        }
        try {
            std::ifstream in(entry.path(), std::ios::binary);
            if (!in) { continue; }
            ConversationFileHeader header{};
            read_bytes(in, &header, sizeof(header), "the file header");
            if (std::memcmp(header.magic, kSlotFileMagic, sizeof(header.magic)) != 0 ||
                header.format != kConversationFileFormat ||
                header.header_bytes != sizeof(ConversationFileHeader) ||
                header.model_id_hash != identity_hash(identity_)) {
                continue; // another artifact's snapshot stays on disk but is not offered.
            }
            DiskConversationEntry disk_entry;
            disk_entry.name       = entry.path().stem().string();
            disk_entry.file_bytes = entry.file_size(error);
            // Skip the payload sections between the file header and the checkpoint records:
            // ledger, prefix identity, then the shared KV pages.
            const std::uint64_t payload_bytes =
                header.ledger_tokens * sizeof(TokenId) + header.prefix_identity_bytes +
                header.text_plane_count * header.parked_text_pages *
                    header.text_page_group_bytes +
                (header.has_backend_kv
                     ? header.backend_plane_count * header.parked_backend_pages *
                           header.backend_page_group_bytes
                     : 0);
            in.seekg(static_cast<std::streamoff>(payload_bytes), std::ios::cur);
            for (std::uint32_t i = 0; i < header.checkpoint_count; ++i) {
                CheckpointFileHeader checkpoint{};
                read_bytes(in, &checkpoint, sizeof(checkpoint), "a checkpoint header");
                const auto skip = [&in](std::uint64_t bytes) {
                    in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
                };
                skip(checkpoint.prefix_identity_bytes + checkpoint.mtp_drafts_bytes +
                     checkpoint.tail_hidden_bytes + checkpoint.turn_checkpoint_hidden_bytes +
                     checkpoint.linear_conv_bytes + checkpoint.linear_recurrent_bytes);
                disk_entry.checkpoint_frontiers.push_back(checkpoint.frontier);
            }
            scanned.emplace(std::move(disk_entry.name), std::move(disk_entry));
        } catch (const std::exception&) {
            continue; // unreadable files are inert until they are overwritten or evicted.
        }
    }
    std::lock_guard lock(mutex_);
    catalog_ = std::move(scanned);
}

void ConversationDiskCache::schedule(std::string name, ConversationSnapshot snapshot) {
    std::lock_guard lock(mutex_);
    pending_[std::move(name)] = std::move(snapshot);
    cv_.notify_one();
}

void ConversationDiskCache::worker_loop() {
    for (;;) {
        ConversationSnapshot job;
        std::string name;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) { return; }
            // One coalesced write at a time; the newest pending snapshot per conversation wins.
            auto it      = pending_.begin();
            name         = it->first;
            job          = std::move(it->second);
            pending_.erase(it);
        }

        try {
            const std::filesystem::path final_path = path_for(name);
            const std::filesystem::path temp_path  = final_path.string() + ".partial";
            try {
                std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
                if (!out) { throw std::runtime_error("cannot open conversation file: " +
                                                     temp_path.string()); }

                ConversationFileHeader header{};
                std::memcpy(header.magic, kSlotFileMagic, sizeof(header.magic));
                header.format                 = kConversationFileFormat;
                header.header_bytes           = sizeof(ConversationFileHeader);
                header.model_id_hash          = identity_hash(identity_);
                header.text_plane_count       = job.payload.text_plane_count;
                header.backend_plane_count    = job.payload.backend_plane_count;
                header.text_page_group_bytes  = job.payload.text_page_group_bytes;
                header.backend_page_group_bytes = job.payload.backend_page_group_bytes;
                header.has_backend_kv         = job.payload.has_backend_kv ? 1U : 0U;
                header.checkpoint_count       = static_cast<std::uint32_t>(job.checkpoints.size());
                header.ledger_tokens          = job.ledger.size();
                header.prefix_identity_bytes  = job.prefix_identity.size();
                header.parked_text_pages =
                    job.payload.text_pages.empty() ? 0 : job.payload.text_pages.front().size();
                header.parked_backend_pages = job.payload.has_backend_kv &&
                                                      !job.payload.backend_pages.empty()
                                                  ? job.payload.backend_pages.front().size()
                                                  : 0;

                write_bytes(out, &header, sizeof(header));
                write_bytes(out, job.ledger.data(),
                            job.ledger.size() * sizeof(TokenId));
                write_bytes(out, job.prefix_identity.data(), job.prefix_identity.size());
                const auto write_pages =
                    [&out](const std::vector<std::vector<ConversationKvPayload::PageBytes>>&
                               planes) {
                        for (const auto& plane : planes) {
                            for (const auto& page : plane) {
                                write_bytes(out, page->data(), page->size());
                            }
                        }
                    };
                write_pages(job.payload.text_pages);
                if (job.payload.has_backend_kv) { write_pages(job.payload.backend_pages); }
                for (const ConversationCheckpoint& checkpoint : job.checkpoints) {
                    CheckpointFileHeader record{};
                    record.kind                       = static_cast<std::uint8_t>(checkpoint.kind);
                    record.frontier                   = checkpoint.frontier;
                    record.execution_frontier         = checkpoint.execution_frontier;
                    record.ledger_frontier            = checkpoint.ledger_frontier;
                    record.text_kv_valid              = checkpoint.text_kv_valid;
                    record.mtp_kv_valid               = checkpoint.mtp_kv_valid;
                    record.dflash_context_frontier    = checkpoint.dflash_context_frontier;
                    record.rope_delta                 = checkpoint.rope_delta;
                    record.mtp_draft_count            = checkpoint.mtp_draft_count;
                    record.tail_hidden_valid          = checkpoint.tail_hidden_valid ? 1U : 0U;
                    record.prefix_identity_bytes      = checkpoint.prefix_identity.size();
                    record.mtp_drafts_bytes           = checkpoint.mtp_drafts.size();
                    record.tail_hidden_bytes          = checkpoint.tail_hidden.size();
                    record.turn_checkpoint_hidden_bytes =
                        checkpoint.turn_checkpoint_hidden.size();
                    record.linear_conv_bytes          = checkpoint.linear_conv.size();
                    record.linear_recurrent_bytes     = checkpoint.linear_recurrent.size();
                    write_bytes(out, &record, sizeof(record));
                    write_bytes(out, checkpoint.prefix_identity.data(),
                                checkpoint.prefix_identity.size());
                    write_bytes(out, checkpoint.mtp_drafts.data(), checkpoint.mtp_drafts.size());
                    write_bytes(out, checkpoint.tail_hidden.data(), checkpoint.tail_hidden.size());
                    write_bytes(out, checkpoint.turn_checkpoint_hidden.data(),
                                checkpoint.turn_checkpoint_hidden.size());
                    write_bytes(out, checkpoint.linear_conv.data(), checkpoint.linear_conv.size());
                    write_bytes(out, checkpoint.linear_recurrent.data(),
                                checkpoint.linear_recurrent.size());
                }
                out.flush();
                if (!out) { throw std::runtime_error("failed while flushing a conversation file"); }
                out.close();
                std::filesystem::rename(temp_path, final_path);
            } catch (...) {
                std::error_code ignored;
                std::filesystem::remove(temp_path, ignored);
                throw;
            }

            std::lock_guard lock(mutex_);
            DiskConversationEntry disk_entry;
            disk_entry.name       = name;
            std::error_code size_error;
            disk_entry.file_bytes = std::filesystem::file_size(final_path, size_error);
            for (const ConversationCheckpoint& checkpoint : job.checkpoints) {
                disk_entry.checkpoint_frontiers.push_back(checkpoint.frontier);
            }
            catalog_[name] = std::move(disk_entry);
            enforce_budget_locked(name);
        } catch (const std::exception&) {
            // Durability is best-effort: the RAM tier still holds this conversation.
        }
    }
}

void ConversationDiskCache::enforce_budget_locked(const std::string&) {
    if (budget_bytes_ == 0) { return; }
    std::uint64_t total = 0;
    for (const auto& [name, entry] : catalog_) { total += entry.file_bytes; }
    while (total > budget_bytes_ && catalog_.size() > 1) {
        // Least-recently-written first: map order is name order, so pick the smallest mtime.
        auto victim_it   = catalog_.begin();
        auto oldest_time = std::filesystem::last_write_time(path_for(victim_it->first)).time_since_epoch().count();
        for (auto it = std::next(catalog_.begin()); it != catalog_.end(); ++it) {
            const auto time =
                std::filesystem::last_write_time(path_for(it->first)).time_since_epoch().count();
            if (time < oldest_time) {
                oldest_time = time;
                victim_it   = it;
            }
        }
        total -= victim_it->second.file_bytes;
        std::error_code ignored;
        std::filesystem::remove(path_for(victim_it->first), ignored);
        catalog_.erase(victim_it);
    }
}

ConversationSnapshot ConversationDiskCache::load(const std::string& name) const {
    const std::filesystem::path path = path_for(name);
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::invalid_argument("conversation file does not exist: " + path.string()); }

    ConversationSnapshot snapshot;
    ConversationFileHeader header{};
    read_bytes(in, &header, sizeof(header), "the file header");
    if (std::memcmp(header.magic, kSlotFileMagic, sizeof(header.magic)) != 0 ||
        header.format != kConversationFileFormat ||
        header.header_bytes != sizeof(ConversationFileHeader) ||
        header.model_id_hash != identity_hash(identity_)) {
        throw std::invalid_argument("conversation file describes a different model");
    }

    snapshot.payload.text_plane_count      = header.text_plane_count;
    snapshot.payload.backend_plane_count   = header.backend_plane_count;
    snapshot.payload.has_backend_kv        = header.has_backend_kv != 0;
    snapshot.payload.text_page_group_bytes = header.text_page_group_bytes;
    snapshot.payload.backend_page_group_bytes = header.backend_page_group_bytes;

    snapshot.ledger.resize(static_cast<std::size_t>(header.ledger_tokens));
    read_bytes(in, snapshot.ledger.data(), snapshot.ledger.size() * sizeof(TokenId),
               "the token ledger");
    snapshot.prefix_identity.resize(static_cast<std::size_t>(header.prefix_identity_bytes));
    read_bytes(in, snapshot.prefix_identity.data(), snapshot.prefix_identity.size(),
               "the prefix identity");

    const auto load_pages = [&](std::uint32_t planes, std::uint64_t group_bytes,
                                std::uint64_t pages,
                                std::vector<std::vector<ConversationKvPayload::PageBytes>>& out,
                                const char* what) {
        out.clear();
        out.resize(planes);
        for (std::uint32_t plane = 0; plane < planes; ++plane) {
            out[plane].resize(static_cast<std::size_t>(pages));
            for (std::uint64_t p = 0; p < pages; ++p) {
                auto buffer = std::make_shared<std::vector<std::byte>>(
                    static_cast<std::size_t>(group_bytes));
                read_bytes(in, buffer->data(), buffer->size(), what);
                out[plane][static_cast<std::size_t>(p)] = std::move(buffer);
            }
        }
    };
    load_pages(header.text_plane_count, header.text_page_group_bytes, header.parked_text_pages,
               snapshot.payload.text_pages, "the text KV payload");
    if (snapshot.payload.has_backend_kv) {
        load_pages(header.backend_plane_count, header.backend_page_group_bytes,
                   header.parked_backend_pages, snapshot.payload.backend_pages,
                   "the backend KV payload");
    }

    for (std::uint32_t i = 0; i < header.checkpoint_count; ++i) {
        CheckpointFileHeader record{};
        read_bytes(in, &record, sizeof(record), "a checkpoint header");
        ConversationCheckpoint checkpoint;
        checkpoint.kind                    = static_cast<CheckpointKind>(record.kind);
        checkpoint.frontier                = record.frontier;
        checkpoint.execution_frontier      = record.execution_frontier;
        checkpoint.ledger_frontier         = record.ledger_frontier;
        checkpoint.text_kv_valid           = record.text_kv_valid;
        checkpoint.mtp_kv_valid            = record.mtp_kv_valid;
        checkpoint.dflash_context_frontier = record.dflash_context_frontier;
        checkpoint.rope_delta              = record.rope_delta;
        checkpoint.mtp_draft_count         = record.mtp_draft_count;
        checkpoint.tail_hidden_valid       = record.tail_hidden_valid != 0;
        const auto read_block = [&](std::uint64_t bytes) {
            std::vector<std::byte> block(static_cast<std::size_t>(bytes));
            read_bytes(in, block.data(), block.size(), "a checkpoint section");
            return block;
        };
        checkpoint.prefix_identity          = read_block(record.prefix_identity_bytes);
        checkpoint.mtp_drafts               = read_block(record.mtp_drafts_bytes);
        checkpoint.tail_hidden              = read_block(record.tail_hidden_bytes);
        checkpoint.turn_checkpoint_hidden   = read_block(record.turn_checkpoint_hidden_bytes);
        checkpoint.linear_conv              = read_block(record.linear_conv_bytes);
        checkpoint.linear_recurrent         = read_block(record.linear_recurrent_bytes);
        snapshot.checkpoints.push_back(std::move(checkpoint));
    }
    in.peek();
    if (!in.eof()) { throw std::invalid_argument("conversation file has trailing bytes"); }
    return snapshot;
}

void ConversationDiskCache::erase(const std::string& name) {
    {
        std::lock_guard lock(mutex_);
        catalog_.erase(name);
        pending_.erase(name);
    }
    // A queued job would resurrect the file; the pending erase above runs before it, and any
    // in-flight write finishes into a file we then remove here.
    const std::filesystem::path path = path_for(name);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(path.string() + ".partial", ignored);
}

} // namespace ninfer::runtime
