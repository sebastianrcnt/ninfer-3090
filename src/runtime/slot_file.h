#pragma once

// ninfer::runtime - on-disk format for a persisted KV slot.
//
// The external API this serves is llama.cpp's (`--slot-save-path`, POST /slots/{id}?action=...),
// but the payload is not llama.cpp's: NInfer stores TurboQuant/INT8 KV in its packed form, plus
// GDN recurrent state and the frontier bookkeeping the prefix matcher needs. Compatibility is at
// the HTTP surface only, so this header is free to describe NInfer state directly.
//
// A restore validates every identity field below before touching device memory. Refusing a
// mismatched file is the whole point: a slot written by a different model, KV format, or
// speculative backend would otherwise be reinterpreted as valid attention state.

#include <cstdint>

namespace ninfer::runtime {

inline constexpr char kSlotFileMagic[8]        = {'N', 'I', 'N', 'F', 'S', 'L', 'O', 'T'};
// v2 describes vision items inside the prefix identity blob; v1 files carried none and are
// refused rather than read, since a v1 identity for a media prompt cannot exist.
// v3 binds model_id_hash to the artifact's build_id as well. A v2 file's hash covers only the
// target name, which every artifact of one architecture shares, so a v2 file cannot be shown to
// have come from the weights now loaded. Refusing them is the point of the bump.
inline constexpr std::uint32_t kSlotFileFormat = 3;

// Trivially copyable and fixed-size; written verbatim ahead of the payload sections. Field order
// is part of the format, so append rather than reorder and raise kSlotFileFormat when the meaning
// of an existing field changes.
struct SlotFileHeader {
    char magic[8];
    std::uint32_t format;
    std::uint32_t header_bytes;

    // Identity. Restore refuses on any disagreement.
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

    // Sequence continuation.
    std::uint32_t execution_frontier;
    std::uint32_t ledger_frontier;
    std::uint32_t text_kv_valid;
    std::uint32_t mtp_kv_valid;
    std::uint32_t dflash_context_frontier;
    std::int32_t rope_delta;
    std::uint32_t mtp_draft_count;
    std::uint32_t turn_checkpoint_frontier;
    std::uint8_t turn_checkpoint_valid;
    std::uint8_t tail_hidden_valid;
    std::uint8_t reserved[2];

    // Payload section sizes, in the order the sections follow this header.
    std::uint64_t ledger_bytes;
    std::uint64_t prefix_identity_bytes;
    std::uint64_t mtp_drafts_bytes;
    std::uint64_t tail_hidden_bytes;
    std::uint64_t turn_checkpoint_hidden_bytes;
    std::uint64_t text_kv_bytes;
    std::uint64_t backend_kv_bytes;
    std::uint64_t linear_conv_bytes;
    std::uint64_t linear_recurrent_bytes;
};

static_assert(sizeof(SlotFileHeader) % 8 == 0, "SlotFileHeader must stay 8-byte aligned");

// Reported back to the HTTP layer, which renders llama.cpp's field names (n_saved/n_written and
// n_restored/n_read). Tokens are ledger length, bytes are the file size actually transferred.
struct SlotTransferResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
};

} // namespace ninfer::runtime
