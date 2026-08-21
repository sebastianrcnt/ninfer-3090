#pragma once

// Qwen3.6 conversation checkpoint semantics: capturing one lane boundary into host-owned state,
// restoring one onto a free lane, and selecting the greatest exactly-matching frontier of a
// cached conversation.
//
// The runtime owns the catalog, the byte budgets, and the disk worker; nothing here knows how
// many conversations exist or where they are stored. What lives here is the part that only the
// target can be right about: which device state a continuation needs, the order the state has to
// be written back in, and what "the prompt still matches this frontier" means for a prepared
// prefix that carries token types, MRoPE positions, and media identity.
//
// A checkpoint is always taken at a round boundary, where the lane's ledger holds exactly
// `execution_frontier + 1` tokens and its tail hidden state predicts the last one. That is the
// same shape a completed request leaves behind, so one capture path serves both the completed
// turn boundary and the periodic grid points inside a long turn.

#include "core/device.h"
#include "runtime/cache/conversation_snapshot.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

// Identity is compared, never trusted from a snapshot, so a cheap stable hash is enough; the
// point is to reject state produced by a different artifact, not to authenticate it.
std::uint64_t conversation_identity_hash(std::string_view value) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char byte : value) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// A lane's linear-attention state is two slices per layer: the live slot and its turn checkpoint.
// The slot extent is outermost in both pools, but they have different rank - conv is
// {channels, width, slots} and recurrent is {key_dim, value_dim, heads, slots} - so the dimension
// is named by the caller rather than inferred.
constexpr int kLinearConvSlotDim      = 2;
constexpr int kLinearRecurrentSlotDim = 3;
constexpr int kCyclicLaneDim          = 3;

// Fixed-size head of one checkpoint's target-opaque state blob. Field order is part of the
// NINFSLOT v4 payload, so append rather than reorder.
struct CheckpointStateHeader {
    std::uint32_t execution_frontier;
    std::uint32_t ledger_frontier;
    std::uint32_t text_kv_valid;
    std::uint32_t mtp_kv_valid;
    std::uint32_t dflash_context_frontier;
    std::int32_t rope_delta;
    std::uint32_t mtp_draft_count;
    std::uint8_t tail_hidden_valid;
    std::uint8_t reserved[3];
    std::uint64_t tail_hidden_bytes;
    std::uint64_t linear_conv_bytes;
    std::uint64_t linear_recurrent_bytes;
    std::uint64_t dflash_local_bytes;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts;
};

static_assert(sizeof(CheckpointStateHeader) % 8 == 0,
              "CheckpointStateHeader must stay 8-byte aligned");

Tensor linear_slot_view(const Tensor& pool, std::int32_t slot, int slot_dim) {
    return pool.slice(slot_dim, slot, 1);
}

// Every device region one checkpoint copies, in the exact order the blob stores them. Keeping the
// enumeration in one place is what makes capture and restore provably symmetric.
class ConversationStateLayout {
public:
    ConversationStateLayout(const ProgramImplCore& program, std::uint32_t lane)
        : live_(LinearStateSlots::current_state_slot(lane, program.max_concurrency)) {
        tail_ = program.sequences[lane].tail_hidden;
        for (const Tensor& layer : program.decoder->linear_attention.conv) {
            conv_.push_back(linear_slot_view(layer, live_, kLinearConvSlotDim));
        }
        for (const Tensor& layer : program.decoder->linear_attention.recurrent) {
            recurrent_.push_back(linear_slot_view(layer, live_, kLinearRecurrentSlotDim));
        }
        if (program.dflash) {
            const auto lane_index = static_cast<std::int32_t>(lane);
            const auto local_layers = static_cast<std::uint32_t>(DFlashConfig::local_layers);
            for (std::uint32_t layer = 0; layer < local_layers; ++layer) {
                const CyclicKVCacheLayerView view = program.dflash->local_layer(layer);
                dflash_.push_back(view.k.slice(kCyclicLaneDim, lane_index, 1));
                dflash_.push_back(view.v.slice(kCyclicLaneDim, lane_index, 1));
            }
        }
    }

    [[nodiscard]] const Tensor& tail() const noexcept { return tail_; }
    [[nodiscard]] const std::vector<Tensor>& conv() const noexcept { return conv_; }
    [[nodiscard]] const std::vector<Tensor>& recurrent() const noexcept { return recurrent_; }
    [[nodiscard]] const std::vector<Tensor>& dflash() const noexcept { return dflash_; }

    [[nodiscard]] std::uint64_t bytes(const std::vector<Tensor>& group) const noexcept {
        std::uint64_t total = 0;
        for (const Tensor& tensor : group) { total += tensor.bytes(); }
        return total;
    }

    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        return sizeof(CheckpointStateHeader) + tail_.bytes() + bytes(conv_) + bytes(recurrent_) +
               bytes(dflash_);
    }

private:
    std::int32_t live_ = 0;
    Tensor tail_;
    std::vector<Tensor> conv_;
    std::vector<Tensor> recurrent_;
    std::vector<Tensor> dflash_;
};

} // namespace

std::uint32_t ProgramImplCore::conversation_checkpoint_state_bytes() const {
    return static_cast<std::uint32_t>(ConversationStateLayout(*this, 0).total_bytes());
}

runtime::ConversationGeometry
ProgramImplCore::conversation_geometry(std::string_view identity) const {
    const PagedKVPool& text_pool         = decoder->text_kv.pool();
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    return runtime::ConversationGeometry{
        .model_id_hash       = conversation_identity_hash(identity),
        .kv_dtype            = static_cast<std::uint32_t>(kv_dtype),
        .kv_storage          = static_cast<std::uint32_t>(kv_storage),
        .kv_quant_group      = kv_quant_group,
        .speculative_backend = static_cast<std::uint32_t>(speculative_backend),
        .kv_capacity         = kv_capacity,
        .page_size           = static_cast<std::uint32_t>(kPagedKVPageSize),
        .text_plane_count    = static_cast<std::uint32_t>(text_pool.plane_count()),
        .backend_plane_count =
            backend != nullptr ? static_cast<std::uint32_t>(backend->pool().plane_count()) : 0U,
        .checkpoint_state_bytes = conversation_checkpoint_state_bytes(),
        .kv_packed_v            = static_cast<std::uint8_t>(kv_packed_v ? 1 : 0),
        .kv_rotate_k            = static_cast<std::uint8_t>(kv_rotate_k ? 1 : 0),
        .kv_rotate_v            = static_cast<std::uint8_t>(kv_rotate_v ? 1 : 0),
        .has_backend_kv         = static_cast<std::uint8_t>(backend != nullptr ? 1 : 0),
    };
}

std::byte* ProgramImplCore::conversation_staging(std::size_t bytes) {
    if (!conversation_staging_ || conversation_staging_->size() < bytes) {
        conversation_staging_.emplace(bytes);
    }
    return static_cast<std::byte*>(conversation_staging_->data());
}

runtime::ConversationCapture
ProgramImplCore::capture_conversation_lane(std::uint32_t lane, std::string_view identity,
                                           std::uint32_t shared_frontier, bool turn_boundary) {
    if (lane >= max_concurrency) { throw std::out_of_range("conversation lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    if (!sequence.kv) { throw std::logic_error("conversation lane has no KV allocation bundle"); }
    if (sequence.execution_frontier == 0 ||
        sequence.ledger.size() != static_cast<std::size_t>(sequence.execution_frontier) + 1ULL ||
        sequence.prefix_identity.size() != sequence.ledger.size() ||
        sequence.text_kv_valid < sequence.execution_frontier || !sequence.tail_hidden_valid) {
        throw std::logic_error("conversation capture requires a lane at a round boundary");
    }
    if (speculative_backend == SpeculativeBackend::DFlash &&
        sequence.dflash_context_frontier != sequence.execution_frontier) {
        // DFlash's committed local context lags its execution frontier between rounds. A
        // checkpoint taken there could never serve as an append base, so it is not taken.
        throw std::logic_error("DFlash local context is behind the capture frontier");
    }

    PagedKVPool& text_pool         = decoder->text_kv.pool();
    qwen3_6::PagedKVCache* backend = sequence.kv->backend ? backend_kv_cache() : nullptr;
    if (sequence.kv->backend && backend == nullptr) {
        throw std::logic_error("backend KV bundle without a backend cache");
    }

    const auto pages_for = [](std::uint32_t tokens) {
        return runtime::conversation_pages_for_tokens(tokens,
                                                       static_cast<std::uint32_t>(kPagedKVPageSize));
    };
    const std::uint32_t text_pages = pages_for(sequence.execution_frontier);
    const std::uint32_t backend_pages =
        backend != nullptr ? std::max(1U, pages_for(backend_kv_valid(sequence))) : 0U;
    if (text_pages > sequence.kv->text.mapped_page_count() ||
        (backend != nullptr && backend_pages > sequence.kv->backend->mapped_page_count())) {
        throw std::logic_error("conversation capture exceeds the lane's mapped KV pages");
    }

    runtime::ConversationCapture capture;
    capture.geometry             = conversation_geometry(identity);
    capture.shared_frontier      = std::min(shared_frontier, sequence.execution_frontier);
    capture.shared_text_pages    = std::min(text_pages, capture.shared_frontier /
                                                            static_cast<std::uint32_t>(
                                                                kPagedKVPageSize));
    capture.shared_backend_pages = std::min(backend_pages, capture.shared_text_pages);
    capture.ledger.assign(sequence.ledger.begin(), sequence.ledger.end());
    capture.identity = sequence.prefix_identity.serialize();

    // Queued device work must land before the host reads those buffers.
    device.synchronize();

    const ConversationStateLayout layout(*this, lane);
    CheckpointStateHeader header{};
    header.execution_frontier      = sequence.execution_frontier;
    header.ledger_frontier         = sequence.ledger_frontier;
    header.text_kv_valid           = sequence.text_kv_valid;
    header.mtp_kv_valid            = sequence.mtp_kv_valid;
    header.dflash_context_frontier = sequence.dflash_context_frontier;
    header.rope_delta              = sequence.rope_delta;
    header.mtp_draft_count         = sequence.mtp_draft_count;
    header.tail_hidden_valid       = 1;
    header.tail_hidden_bytes       = layout.tail().bytes();
    header.linear_conv_bytes       = layout.bytes(layout.conv());
    header.linear_recurrent_bytes  = layout.bytes(layout.recurrent());
    header.dflash_local_bytes      = layout.bytes(layout.dflash());
    header.mtp_drafts              = sequence.mtp_drafts;

    std::vector<std::byte> state(static_cast<std::size_t>(layout.total_bytes()));
    std::memcpy(state.data(), &header, sizeof(header));
    std::size_t cursor = sizeof(header);
    const auto read_device = [&](const Tensor& tensor) {
        if (tensor.bytes() == 0 || tensor.data == nullptr) { return; }
        CUDA_CHECK(cudaMemcpy(state.data() + cursor, tensor.data, tensor.bytes(),
                              cudaMemcpyDeviceToHost));
        cursor += tensor.bytes();
    };
    read_device(layout.tail());
    for (const Tensor& tensor : layout.conv()) { read_device(tensor); }
    for (const Tensor& tensor : layout.recurrent()) { read_device(tensor); }
    for (const Tensor& tensor : layout.dflash()) { read_device(tensor); }
    if (cursor != state.size()) {
        throw std::logic_error("conversation checkpoint state layout is inconsistent");
    }

    capture.checkpoint = runtime::ConversationCheckpoint{
        .frontier      = sequence.execution_frontier,
        .text_pages    = text_pages,
        .backend_pages = backend_pages,
        .turn_boundary = static_cast<std::uint8_t>(turn_boundary ? 1 : 0),
        .state         = std::move(state),
    };

    // Packed KV moves through one bounded pinned staging page rather than a full-size host mirror:
    // a 262,144-token conversation carries several GiB, and materializing that twice to hand it to
    // the cache would defeat the point of caching it.
    const auto copy_pages = [&](PagedKVPool& pool, const PagedKVAllocation& allocation,
                                std::uint32_t first, std::uint32_t last,
                                runtime::HostKvPayload& out) {
        out.planes.resize(pool.plane_count());
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            const std::size_t group_bytes = pool.page_group_bytes(plane);
            out.planes[plane].group_bytes = group_bytes;
            out.planes[plane].pages.reserve(last - first);
            std::byte* staging = conversation_staging(group_bytes);
            for (std::uint32_t page = first; page < last; ++page) {
                pool.read_page_group(plane, allocation.page_ids()[page], staging, device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                out.planes[plane].pages.push_back(std::make_shared<const std::vector<std::byte>>(
                    staging, staging + group_bytes));
            }
        }
    };
    copy_pages(text_pool, sequence.kv->text, capture.shared_text_pages, text_pages,
               capture.new_text);
    if (backend != nullptr) {
        copy_pages(backend->pool(), *sequence.kv->backend, capture.shared_backend_pages,
                   backend_pages, capture.new_backend);
    }
    return capture;
}

void ProgramImplCore::restore_conversation_lane(std::uint32_t lane,
                                                const runtime::ConversationSnapshot& snapshot,
                                                std::size_t checkpoint_index,
                                                std::string_view identity) {
    if (lane >= max_concurrency) { throw std::out_of_range("conversation lane is out of range"); }
    if (checkpoint_index >= snapshot.checkpoints.size()) {
        throw std::out_of_range("conversation checkpoint is out of range");
    }
    if (!(snapshot.geometry == conversation_geometry(identity))) {
        throw std::invalid_argument(
            "cached conversation describes a different model or KV configuration");
    }
    const runtime::ConversationCheckpoint& checkpoint = snapshot.checkpoints[checkpoint_index];
    if (checkpoint.state.size() != snapshot.geometry.checkpoint_state_bytes ||
        checkpoint.frontier == 0 ||
        static_cast<std::size_t>(checkpoint.frontier) + 1ULL > snapshot.ledger.size() ||
        checkpoint.text_pages > snapshot.text.page_count() ||
        checkpoint.backend_pages > snapshot.backend.page_count()) {
        throw std::invalid_argument("cached conversation checkpoint is not self-consistent");
    }

    CheckpointStateHeader header{};
    std::memcpy(&header, checkpoint.state.data(), sizeof(header));
    if (header.execution_frontier != checkpoint.frontier ||
        header.ledger_frontier != checkpoint.frontier + 1U || header.tail_hidden_valid == 0U ||
        header.mtp_draft_count > qwen3_6::kMtpDecodeMaximumDrafts) {
        throw std::invalid_argument("cached conversation checkpoint state is inconsistent");
    }

    PagedKVPool& text_pool         = decoder->text_kv.pool();
    qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if ((backend != nullptr) != (snapshot.geometry.has_backend_kv != 0U) ||
        (backend != nullptr && checkpoint.backend_pages == 0)) {
        throw std::invalid_argument("cached conversation backend KV does not match this Engine");
    }

    SequenceState& sequence = sequences[lane];
    clear_lane(sequence, requests[lane]);
    try {
        const ConversationStateLayout layout(*this, lane);
        if (layout.total_bytes() != checkpoint.state.size() ||
            header.tail_hidden_bytes != layout.tail().bytes() ||
            header.linear_conv_bytes != layout.bytes(layout.conv()) ||
            header.linear_recurrent_bytes != layout.bytes(layout.recurrent()) ||
            header.dflash_local_bytes != layout.bytes(layout.dflash())) {
            throw std::invalid_argument("cached conversation state does not match this model");
        }

        sequence.ledger.assign(snapshot.ledger.begin(),
                               snapshot.ledger.begin() +
                                   static_cast<std::ptrdiff_t>(checkpoint.frontier) + 1);
        sequence.prefix_identity.deserialize(snapshot.identity);
        if (sequence.prefix_identity.size() < sequence.ledger.size()) {
            throw std::invalid_argument("cached conversation identity is shorter than its ledger");
        }
        sequence.prefix_identity.truncate(sequence.ledger.size());

        // Deliberately left unbound, exactly as a lane that finished a request and stayed
        // retained is: admission binds the table row again on the next request.
        reserve_sequence_kv(sequence, checkpoint.text_pages, checkpoint.backend_pages);
        sequence.kv->text.materialize_pages(checkpoint.text_pages, device.stream);
        if (sequence.kv->backend) {
            sequence.kv->backend->materialize_pages(checkpoint.backend_pages, device.stream);
        }

        const auto write_pages = [&](PagedKVPool& pool, const PagedKVAllocation& allocation,
                                     const runtime::HostKvPayload& payload, std::uint32_t pages,
                                     const char* what) {
            if (payload.planes.size() != pool.plane_count()) {
                throw std::invalid_argument(std::string("cached conversation ") + what +
                                            " plane count does not match this pool");
            }
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                const std::size_t group_bytes = pool.page_group_bytes(plane);
                if (payload.planes[plane].group_bytes != group_bytes) {
                    throw std::invalid_argument(std::string("cached conversation ") + what +
                                                " page geometry does not match this pool");
                }
                std::byte* staging = conversation_staging(group_bytes);
                for (std::uint32_t page = 0; page < pages; ++page) {
                    const runtime::HostPage& source = payload.planes[plane].pages[page];
                    if (source == nullptr || source->size() != group_bytes) {
                        throw std::invalid_argument(std::string("cached conversation ") + what +
                                                    " page has the wrong size");
                    }
                    std::memcpy(staging, source->data(), group_bytes);
                    pool.write_page_group(plane, allocation.page_ids()[page], staging,
                                          device.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device.stream));
                }
            }
        };
        write_pages(text_pool, sequence.kv->text, snapshot.text, checkpoint.text_pages,
                    "Text KV");
        if (sequence.kv->backend) {
            write_pages(backend->pool(), *sequence.kv->backend, snapshot.backend,
                        checkpoint.backend_pages, "backend KV");
        }

        std::size_t cursor = sizeof(header);
        const auto write_device = [&](const Tensor& tensor) {
            if (tensor.bytes() == 0 || tensor.data == nullptr) { return; }
            CUDA_CHECK(cudaMemcpy(tensor.data, checkpoint.state.data() + cursor, tensor.bytes(),
                                  cudaMemcpyHostToDevice));
            cursor += tensor.bytes();
        };
        write_device(layout.tail());
        for (const Tensor& tensor : layout.conv()) { write_device(tensor); }
        for (const Tensor& tensor : layout.recurrent()) { write_device(tensor); }
        for (const Tensor& tensor : layout.dflash()) { write_device(tensor); }
        device.synchronize();

        sequence.execution_frontier = header.execution_frontier;
        sequence.ledger_frontier    = header.ledger_frontier;
        // A checkpoint restores exactly the KV its stored pages cover, never the larger extent the
        // lane happened to have valid when the capture was taken.
        sequence.text_kv_valid = std::min(header.text_kv_valid, header.execution_frontier);
        sequence.mtp_kv_valid  = std::min(header.mtp_kv_valid, header.execution_frontier);
        sequence.dflash_context_frontier =
            std::min(header.dflash_context_frontier, header.execution_frontier);
        sequence.rope_delta              = header.rope_delta;
        sequence.mtp_drafts              = header.mtp_drafts;
        sequence.mtp_draft_count         = header.mtp_draft_count;
        sequence.tail_hidden_valid       = true;
        // The GPU turn checkpoint belongs to the lane's previous occupant. The restored
        // conversation's own history lives in the cache, so the slot starts invalid.
        sequence.turn_checkpoint = {};
        sequence.retained        = true;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, requests[lane]);
        throw;
    }
}

std::uint32_t ProgramImplCore::retained_token_count_lane(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) ? static_cast<std::uint32_t>(sequences[lane].ledger.size()) : 0U;
}

std::optional<std::size_t> ProgramImplCore::select_conversation_checkpoint(
    const PreparedPromptData& prompt, const std::vector<TokenId>& ledger,
    const std::vector<std::byte>& identity_blob,
    const std::vector<runtime::ConversationCheckpoint>& checkpoints) const {
    if (!prompt.identity.reusable || checkpoints.empty() || ledger.empty()) {
        return std::nullopt;
    }
    qwen3_6::detail::ResidentPrefixIdentity identity;
    try {
        identity.deserialize(identity_blob);
    } catch (const std::exception&) { return std::nullopt; }
    if (identity.size() != ledger.size()) { return std::nullopt; }

    const auto prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    for (std::size_t index = checkpoints.size(); index-- > 0;) {
        const std::uint32_t frontier = checkpoints[index].frontier;
        // A checkpoint past the prompt cannot be a prefix of it, and one past the ledger is not
        // describable by this conversation at all.
        if (frontier == 0 || frontier > prompt_tokens ||
            static_cast<std::size_t>(frontier) + 1ULL > ledger.size()) {
            continue;
        }
        if (speculative_backend == SpeculativeBackend::Mtp) {
            CheckpointStateHeader header{};
            if (checkpoints[index].state.size() >= sizeof(header)) {
                std::memcpy(&header, checkpoints[index].state.data(), sizeof(header));
                if (frontier != 0 && header.mtp_kv_valid + 1U < frontier) { continue; }
            }
        }
        if (qwen3_6::detail::prefix_matches(prompt, ledger, identity, frontier)) { return index; }
    }
    return std::nullopt;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
