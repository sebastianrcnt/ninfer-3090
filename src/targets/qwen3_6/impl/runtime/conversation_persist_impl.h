#pragma once

// Qwen3.6 conversation-tiered checkpoint capture and restore.
//
// The byte composition of a checkpoint mirrors the v3 slot payload exactly — ledger, prefix
// identity, drafts, hidden vectors, GDN slot image, packed KV page groups — but lands in host
// memory owned by the runtime catalog instead of a file, and carries one GDN slot image (the
// live role) rather than both. Capture is only valid at a retention boundary, where the family
// invariants hold: ledger.size() == ledger_frontier == execution_frontier + 1 and
// mtp_draft_count == 0. The caller holds the executor lock; these calls synchronize the device
// stream themselves.
//
// A lane's linear-attention state slice per layer carries the slot extent on a rank-dependent
// dimension: conv is {channels, width, slots}, recurrent is {key_dim, value_dim, heads, slots}.

#include "runtime/conversation_cache.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

namespace {

constexpr int kLinearConvSlotDim      = 2;
constexpr int kLinearRecurrentSlotDim = 3;

Tensor linear_slot_view(const Tensor& pool, std::int32_t slot, int slot_dim) {
    return pool.slice(slot_dim, slot, 1);
}

std::vector<std::byte> device_bytes(const Tensor& tensor) {
    const std::size_t bytes = tensor.bytes();
    if (bytes == 0 || tensor.data == nullptr) { return {}; }
    std::vector<std::byte> out(bytes);
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, bytes, cudaMemcpyDeviceToHost));
    return out;
}

void write_device_bytes(const Tensor& tensor, const std::vector<std::byte>& bytes,
                        const char* what) {
    if (bytes.size() != tensor.bytes()) {
        throw std::invalid_argument(std::string("conversation checkpoint ") + what +
                                    " has the wrong size for this model");
    }
    if (bytes.empty()) { return; }
    CUDA_CHECK(cudaMemcpy(tensor.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}

} // namespace

std::vector<TokenId> ProgramImplCore::retained_lane_ledger_copy(std::uint32_t lane) const {
    if (!has_retained_lane(lane)) { return {}; }
    return sequences[lane].ledger;
}

std::vector<std::byte> ProgramImplCore::retained_lane_identity_copy(std::uint32_t lane) const {
    if (!has_retained_lane(lane)) { return {}; }
    return sequences[lane].prefix_identity.serialize();
}

runtime::ConversationCheckpoint
ProgramImplCore::capture_retained_lane_checkpoint(std::uint32_t lane,
                                                  runtime::CheckpointKind kind) {
    if (!has_retained_lane(lane)) {
        throw std::invalid_argument("conversation capture requires a retained lane");
    }
    const SequenceState& sequence = sequences[lane];
    // The GPU's intra-prompt turn checkpoint is optional here: single-shot requests retain a
    // complete conversation boundary without ever capturing one.
    if (!sequence.kv || sequence.mtp_draft_count != 0 ||
        sequence.ledger.size() != sequence.execution_frontier + 1U ||
        sequence.prefix_identity.size() != sequence.ledger.size()) {
        throw std::logic_error("retained lane does not hold a complete turn boundary");
    }

    device.synchronize();

    runtime::ConversationCheckpoint checkpoint;
    checkpoint.kind                    = kind;
    checkpoint.frontier                = sequence.execution_frontier;
    checkpoint.execution_frontier      = sequence.execution_frontier;
    checkpoint.ledger_frontier         = sequence.ledger_frontier;
    checkpoint.text_kv_valid           = sequence.text_kv_valid;
    checkpoint.mtp_kv_valid            = sequence.mtp_kv_valid;
    checkpoint.dflash_context_frontier = sequence.dflash_context_frontier;
    checkpoint.rope_delta              = sequence.rope_delta;
    checkpoint.mtp_draft_count         = sequence.mtp_draft_count;
    checkpoint.tail_hidden_valid       = sequence.tail_hidden_valid;
    checkpoint.prefix_identity         = sequence.prefix_identity.serialize();
    checkpoint.mtp_drafts.resize(sizeof(sequence.mtp_drafts));
    std::memcpy(checkpoint.mtp_drafts.data(), sequence.mtp_drafts.data(),
                sizeof(sequence.mtp_drafts));
    checkpoint.tail_hidden            = device_bytes(sequence.tail_hidden);
    checkpoint.turn_checkpoint_hidden = device_bytes(sequence.turn_checkpoint_hidden);

    // One live-role GDN image per layer, flattened layer-major into single buffers.
    const std::int32_t live_slot = LinearStateSlots::current_state_slot(lane, max_concurrency);
    for (const Tensor& layer : decoder->linear_attention.conv) {
        const auto slice = device_bytes(linear_slot_view(layer, live_slot, kLinearConvSlotDim));
        checkpoint.linear_conv.insert(checkpoint.linear_conv.end(), slice.begin(), slice.end());
    }
    for (const Tensor& layer : decoder->linear_attention.recurrent) {
        const auto slice =
            device_bytes(linear_slot_view(layer, live_slot, kLinearRecurrentSlotDim));
        checkpoint.linear_recurrent.insert(checkpoint.linear_recurrent.end(), slice.begin(),
                                           slice.end());
    }
    return checkpoint;
}

void ProgramImplCore::capture_lane_kv_payload(std::uint32_t lane,
                                              runtime::ConversationKvPayload& payload,
                                              std::size_t& text_parked,
                                              std::size_t& backend_parked) {
    const SequenceState& sequence = sequences[lane];
    if (!sequence.kv) { throw std::invalid_argument("conversation capture requires KV state"); }

    PagedKVPool& text_pool         = decoder->text_kv.pool();
    qwen3_6::PagedKVCache* backend = sequence.kv->backend ? backend_kv_cache() : nullptr;

    // Pages are immutable once written, so a re-park copies only pages past the parked mark and
    // keeps the already-captured page pointers for the unchanged prefix.
    const auto stream_pages =
        [&](PagedKVPool& pool, const PagedKVAllocation& allocation,
            std::vector<std::vector<runtime::ConversationKvPayload::PageBytes>>& planes,
            std::size_t& parked_pages) {
            if (planes.size() != pool.plane_count()) { planes.assign(pool.plane_count(), {}); }
            const std::size_t pages = allocation.page_ids().size();
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                auto& row = planes[plane];
                row.resize(pages);
                const std::size_t begin = std::min(parked_pages, pages);
                for (std::size_t p = begin; p < pages; ++p) {
                    auto buffer =
                        std::make_shared<std::vector<std::byte>>(pool.page_group_bytes(plane));
                    pool.read_page_group(plane, allocation.page_ids()[p], buffer->data(),
                                         device.stream);
                    row[p] = std::move(buffer);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            parked_pages = pages;
        };

    device.synchronize();
    payload.has_backend_kv        = backend != nullptr;
    payload.text_plane_count      = static_cast<std::uint32_t>(text_pool.plane_count());
    payload.text_page_group_bytes = text_pool.page_group_bytes(0);
    stream_pages(text_pool, sequence.kv->text, payload.text_pages, text_parked);
    if (backend != nullptr) {
        payload.backend_plane_count = static_cast<std::uint32_t>(backend->pool().plane_count());
        payload.backend_page_group_bytes = backend->pool().page_group_bytes(0);
        stream_pages(backend->pool(), *sequence.kv->backend, payload.backend_pages,
                     backend_parked);
    } else {
        payload.backend_plane_count      = 0;
        payload.backend_page_group_bytes = 0;
        payload.backend_pages.clear();
        backend_parked = 0;
    }
}

bool ProgramImplCore::conversation_checkpoint_matches(
    const runtime::ConversationCheckpoint& checkpoint, const std::vector<TokenId>& ledger,
    const std::vector<std::byte>& identity_blob, const PreparedPromptData& prompt) const {
    if (checkpoint.frontier > ledger.size()) { return false; }
    qwen3_6::detail::ResidentPrefixIdentity identity;
    try {
        identity.deserialize(identity_blob);
    } catch (const std::invalid_argument&) { return false; }
    return prefix_matches(prompt, ledger, identity, checkpoint.frontier);
}

void ProgramImplCore::restore_lane_from_conversation(
    std::uint32_t lane, const runtime::ConversationCheckpoint& checkpoint,
    std::span<const TokenId> ledger, const runtime::ConversationKvPayload& payload) {
    if (lane >= max_concurrency) { throw std::out_of_range("conversation lane is out of range"); }
    if (checkpoint.frontier > ledger.size() || checkpoint.prefix_identity.empty()) {
        throw std::invalid_argument("conversation checkpoint disagrees with its conversation");
    }
    PagedKVPool& text_pool = decoder->text_kv.pool();
    if (payload.text_plane_count != static_cast<std::uint32_t>(text_pool.plane_count()) ||
        payload.text_page_group_bytes == 0) {
        throw std::invalid_argument("conversation KV geometry does not match this pool");
    }
    qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (payload.has_backend_kv && backend == nullptr) {
        throw std::invalid_argument("conversation carries backend KV this engine does not load");
    }
    if (backend != nullptr &&
        (payload.backend_plane_count !=
             static_cast<std::uint32_t>(backend->pool().plane_count()) ||
         payload.backend_page_group_bytes == 0)) {
        throw std::invalid_argument("conversation backend KV geometry does not match this pool");
    }

    SequenceState& sequence = sequences[lane];
    clear_lane(sequence, requests[lane]);
    try {
        sequence.ledger.assign(ledger.begin(), ledger.end());
        sequence.prefix_identity.deserialize(checkpoint.prefix_identity);
        if (sequence.prefix_identity.size() != sequence.ledger.size()) {
            throw std::invalid_argument("checkpoint identity length disagrees with its ledger");
        }

        const auto plane_pages =
            [](const std::vector<std::vector<runtime::ConversationKvPayload::PageBytes>>& planes) {
                return planes.empty()
                           ? std::uint32_t{0}
                           : static_cast<std::uint32_t>(planes.front().size());
            };
        const std::uint32_t text_pages    = plane_pages(payload.text_pages);
        const std::uint32_t backend_pages = payload.has_backend_kv
                                                ? plane_pages(payload.backend_pages)
                                                : 0U;
        reserve_sequence_kv(sequence, text_pages, backend_pages);
        sequence.kv->text.materialize_pages(text_pages, device.stream);
        if (sequence.kv->backend) {
            sequence.kv->backend->materialize_pages(backend_pages, device.stream);
        }

        write_device_bytes(sequence.tail_hidden, checkpoint.tail_hidden, "tail hidden state");
        write_device_bytes(sequence.turn_checkpoint_hidden, checkpoint.turn_checkpoint_hidden,
                           "turn checkpoint hidden state");

        const auto load_kv =
            [&](PagedKVPool& pool, const PagedKVAllocation& allocation,
                const std::vector<std::vector<runtime::ConversationKvPayload::PageBytes>>& planes,
                const char* what) {
                for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                    const auto& pages = planes.at(plane);
                    if (pages.size() != allocation.page_ids().size()) {
                        throw std::invalid_argument(std::string("conversation ") + what +
                                                    " page count disagrees with the reservation");
                    }
                    for (std::size_t p = 0; p < pages.size(); ++p) {
                        pool.write_page_group(plane, allocation.page_ids()[p], pages[p]->data(),
                                              device.stream);
                    }
                }
            };
        load_kv(text_pool, sequence.kv->text, payload.text_pages, "text KV payload");
        if (sequence.kv->backend && backend != nullptr) {
            load_kv(backend->pool(), *sequence.kv->backend, payload.backend_pages,
                    "backend KV payload");
        }

        // One live-role GDN image per layer; a turn-boundary checkpoint also repopulates the GPU
        // turn-checkpoint role so post-restore state matches a lane that just finished a request
        // on this machine.
        const std::int32_t live_slot =
            LinearStateSlots::current_state_slot(lane, max_concurrency);
        const std::int32_t check_slot =
            LinearStateSlots::turn_checkpoint_state_slot(lane, max_concurrency);
        const auto load_linear = [&](const std::vector<Tensor>& layers,
                                     const std::vector<std::byte>& bytes, int slot_dim,
                                     const char* what) {
            if (!layers.empty() && bytes.size() % layers.size() != 0) {
                throw std::invalid_argument(std::string("conversation checkpoint ") + what +
                                            " size disagrees with this model");
            }
            std::size_t offset = 0;
            for (const Tensor& layer : layers) {
                const Tensor view = linear_slot_view(layer, live_slot, slot_dim);
                if (offset + view.bytes() > bytes.size()) {
                    throw std::invalid_argument(std::string("conversation checkpoint ") + what +
                                                " is truncated");
                }
                CUDA_CHECK(cudaMemcpy(view.data, bytes.data() + offset, view.bytes(),
                                      cudaMemcpyHostToDevice));
                if (checkpoint.kind == runtime::CheckpointKind::TurnBoundary) {
                    const Tensor check_view = linear_slot_view(layer, check_slot, slot_dim);
                    CUDA_CHECK(cudaMemcpy(check_view.data, view.data, check_view.bytes(),
                                          cudaMemcpyDeviceToDevice));
                }
                offset += view.bytes();
            }
        };
        load_linear(decoder->linear_attention.conv, checkpoint.linear_conv, kLinearConvSlotDim,
                    "GDN convolution state");
        load_linear(decoder->linear_attention.recurrent, checkpoint.linear_recurrent,
                    kLinearRecurrentSlotDim, "GDN recurrent state");

        device.synchronize();

        sequence.execution_frontier       = checkpoint.execution_frontier;
        sequence.ledger_frontier          = checkpoint.ledger_frontier;
        sequence.text_kv_valid            = checkpoint.text_kv_valid;
        sequence.mtp_kv_valid             = checkpoint.mtp_kv_valid;
        sequence.dflash_context_frontier  = checkpoint.dflash_context_frontier;
        sequence.rope_delta               = checkpoint.rope_delta;
        sequence.mtp_draft_count          = checkpoint.mtp_draft_count;
        sequence.tail_hidden_valid        = checkpoint.tail_hidden_valid;
        sequence.retained                 = true;
        sequence.turn_checkpoint.valid    = checkpoint.kind == runtime::CheckpointKind::TurnBoundary;
        sequence.turn_checkpoint.frontier =
            sequence.turn_checkpoint.valid ? checkpoint.frontier : 0U;
    } catch (...) {
        clear_lane(sequence, requests[lane]);
        throw;
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
