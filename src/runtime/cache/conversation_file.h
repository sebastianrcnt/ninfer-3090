#pragma once

// ninfer::runtime - NINFSLOT v4: one conversation's shared KV payload plus its ordered checkpoint
// set, as a durable local file.
//
// v4 replaces the lane-oriented v3 slot file. v3 stored one lane's live state and its single turn
// checkpoint and could only be written by stopping the server; a v4 file stores a conversation,
// so it can be published after a completed request and restored onto whichever lane is free.
// v3 files are refused rather than read: their payload has no checkpoint set and their identity
// covers a lane rather than a conversation.
//
// The header is deliberately complete enough to serve as the startup catalog. A restart reads the
// header, ledger, identity, and checkpoint records - about a megabyte for a full 262,144-token
// conversation - and leaves the multi-gigabyte page payload on disk until a prompt actually
// matches it.

#include "runtime/cache/conversation_snapshot.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ninfer::runtime {

inline constexpr char kConversationFileMagic[8] = {'N', 'I', 'N', 'F', 'S', 'L', 'O', 'T'};
inline constexpr std::uint32_t kConversationFileFormat = 4;

// Trivially copyable and fixed-size; written verbatim ahead of the sections. Field order is part
// of the format, so append rather than reorder and raise kConversationFileFormat when the meaning
// of an existing field changes.
struct ConversationFileHeader {
    char magic[8];
    std::uint32_t format;
    std::uint32_t header_bytes;

    // Identity and geometry. A restore refuses on any disagreement.
    std::uint64_t model_id_hash;
    std::uint32_t kv_dtype;
    std::uint32_t kv_storage;
    std::int32_t kv_quant_group;
    std::uint32_t speculative_backend;
    std::uint32_t kv_capacity;
    std::uint32_t page_size;
    std::uint32_t text_plane_count;
    std::uint32_t backend_plane_count;
    std::uint32_t checkpoint_state_bytes;
    std::uint8_t kv_packed_v;
    std::uint8_t kv_rotate_k;
    std::uint8_t kv_rotate_v;
    std::uint8_t has_backend_kv;

    // Catalog sections, in the order they follow this header.
    std::uint32_t checkpoint_count;
    std::uint32_t text_page_count;
    std::uint32_t backend_page_count;
    std::uint32_t reserved;
    std::uint64_t ledger_bytes;
    std::uint64_t identity_bytes;
    std::uint64_t text_payload_bytes;
    std::uint64_t backend_payload_bytes;
};

static_assert(sizeof(ConversationFileHeader) % 8 == 0,
              "ConversationFileHeader must stay 8-byte aligned");

struct ConversationCheckpointRecord {
    std::uint32_t frontier;
    std::uint32_t text_pages;
    std::uint32_t backend_pages;
    std::uint8_t turn_boundary;
    std::uint8_t reserved[3];
};

static_assert(sizeof(ConversationCheckpointRecord) == 16,
              "ConversationCheckpointRecord must stay 16 bytes");

// Everything a restart needs to decide whether a prompt matches, without reading the payload.
struct ConversationFileCatalog {
    std::filesystem::path path;
    ConversationGeometry geometry;
    std::vector<TokenId> ledger;
    std::vector<std::byte> identity;
    std::vector<ConversationCheckpoint> checkpoints; // state blobs are left empty here
    std::uint64_t file_bytes = 0;
};

// Reads header, ledger, identity, and checkpoint records only. Throws std::invalid_argument on a
// magic/format/geometry mismatch or on a truncated or internally inconsistent file.
[[nodiscard]] ConversationFileCatalog read_conversation_catalog(const std::filesystem::path& path,
                                                                const ConversationGeometry& expect);

// Reads the checkpoint state blobs plus the KV pages the newest retained checkpoint covers. The
// catalog must have come from read_conversation_catalog for the same file.
[[nodiscard]] ConversationSnapshot
read_conversation_payload(const ConversationFileCatalog& catalog);

// Publishes through a sibling temporary and an atomic rename, so a crash mid-write cannot leave a
// half file that would later pass the header check. The previous valid file stays readable until
// its replacement is complete. Returns the byte size of the published file.
std::uint64_t write_conversation_file(const std::filesystem::path& path,
                                      const ConversationSnapshot& snapshot);

} // namespace ninfer::runtime
