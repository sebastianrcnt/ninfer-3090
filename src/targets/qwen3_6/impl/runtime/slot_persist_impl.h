#pragma once

// Qwen3.6 slot persistence: one retained lane to a file and back.
//
// The lane must be idle; the executor holds its scheduler lock across these calls. Everything
// here synchronizes the device stream before touching host buffers, because the payload is read
// straight out of the live pools rather than through a staging copy on the GPU.

#include "core/device.h"
#include "runtime/slot_file.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <span>
#include <utility>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

// Identity is compared, never trusted from the file, so a cheap stable hash is enough; the point
// is to reject a slot written by a different artifact, not to authenticate one.
std::uint64_t slot_identity_hash(std::string_view value) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char byte : value) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// A lane's linear-attention state is two slices per layer: the live slot and its turn checkpoint.
// The slot extent is outermost in both pools, but they have different rank — conv is
// {channels, width, slots} and recurrent is {key_dim, value_dim, heads, slots} — so the dimension
// is named by the caller rather than inferred.
constexpr int kLinearConvSlotDim      = 2;
constexpr int kLinearRecurrentSlotDim = 3;

// Device payload moves through a fixed staging buffer rather than a full-size host copy. A 262K
// slot carries several GiB of KV; materializing that in RAM to write it to disk would defeat the
// point of persisting it. 8 MiB is comfortably above one KV page group and one linear-state slice.
constexpr std::size_t kSlotStagingBytes = 8UL << 20;

Tensor linear_slot_view(const Tensor& pool, std::int32_t slot, int slot_dim) {
    return pool.slice(slot_dim, slot, 1);
}

// Streams a slot to disk. Every device read lands in `staging_` first, so host residency stays
// bounded by kSlotStagingBytes no matter how large the slot is.
class SlotWriter {
public:
    explicit SlotWriter(const std::filesystem::path& path)
        : out_(path, std::ios::binary | std::ios::trunc), staging_(kSlotStagingBytes) {
        if (!out_) {
            throw std::runtime_error("cannot open slot file for writing: " + path.string());
        }
    }

    void host(const void* data, std::size_t bytes) {
        if (bytes == 0) { return; }
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        check();
    }

    std::uint64_t device(const Tensor& tensor) {
        const std::size_t bytes = tensor.bytes();
        if (bytes == 0 || tensor.data == nullptr) { return 0; }
        const auto* source = static_cast<const std::byte*>(tensor.data);
        for (std::size_t offset = 0; offset < bytes; offset += staging_.size()) {
            const std::size_t chunk = std::min(staging_.size(), bytes - offset);
            CUDA_CHECK(cudaMemcpy(staging_.data(), source + offset, chunk, cudaMemcpyDeviceToHost));
            host(staging_.data(), chunk);
        }
        return bytes;
    }

    void page_group(PagedKVPool& pool, std::size_t plane, std::int32_t page, cudaStream_t stream) {
        const std::size_t bytes = pool.page_group_bytes(plane);
        if (bytes > staging_.size()) {
            throw std::runtime_error("KV page group exceeds the slot staging buffer");
        }
        pool.read_page_group(plane, page, staging_.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        host(staging_.data(), bytes);
    }

    [[nodiscard]] std::uint64_t written() { return static_cast<std::uint64_t>(out_.tellp()); }

    void finish() {
        out_.flush();
        check();
    }

private:
    void check() {
        if (!out_) { throw std::runtime_error("failed while writing the slot file"); }
    }

    std::ofstream out_;
    std::vector<std::byte> staging_;
};

// Symmetric reader. Sections are consumed in the order the header declares them.
class SlotReader {
public:
    explicit SlotReader(const std::filesystem::path& path)
        : in_(path, std::ios::binary), staging_(kSlotStagingBytes) {
        if (!in_) { throw std::invalid_argument("slot file does not exist: " + path.string()); }
    }

    void host(void* data, std::size_t bytes, const char* what) {
        if (bytes == 0) { return; }
        in_.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
        check(what);
    }

    std::vector<std::byte> host_block(std::uint64_t bytes, const char* what) {
        std::vector<std::byte> block(static_cast<std::size_t>(bytes));
        host(block.data(), block.size(), what);
        return block;
    }

    void device(const Tensor& tensor, std::uint64_t bytes, const char* what) {
        if (bytes == 0) { return; }
        if (tensor.data == nullptr || tensor.bytes() != bytes) {
            throw std::invalid_argument(std::string("slot file ") + what +
                                        " has the wrong size for this model");
        }
        auto* destination = static_cast<std::byte*>(tensor.data);
        for (std::uint64_t offset = 0; offset < bytes; offset += staging_.size()) {
            const auto chunk =
                static_cast<std::size_t>(std::min<std::uint64_t>(staging_.size(), bytes - offset));
            host(staging_.data(), chunk, what);
            CUDA_CHECK(cudaMemcpy(destination + offset, staging_.data(), chunk,
                                  cudaMemcpyHostToDevice));
        }
    }

    void page_group(PagedKVPool& pool, std::size_t plane, std::int32_t page, cudaStream_t stream,
                    const char* what) {
        const std::size_t bytes = pool.page_group_bytes(plane);
        if (bytes > staging_.size()) {
            throw std::runtime_error("KV page group exceeds the slot staging buffer");
        }
        host(staging_.data(), bytes, what);
        pool.write_page_group(plane, page, staging_.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void expect_end() {
        in_.peek();
        if (!in_.eof()) { throw std::invalid_argument("slot file has trailing bytes"); }
    }

private:
    void check(const char* what) {
        if (!in_) { throw std::invalid_argument(std::string("slot file is truncated in ") + what); }
    }

    std::ifstream in_;
    std::vector<std::byte> staging_;
};

} // namespace

runtime::SlotTransferResult ProgramImplCore::save_retained_lane(std::uint32_t lane,
                                                                const std::string& path,
                                                                std::string_view identity) {
    if (!has_retained_lane(lane)) {
        throw std::invalid_argument("slot lane holds no retained sequence");
    }
    SequenceState& sequence = sequences[lane];
    if (!sequence.kv) { throw std::logic_error("retained lane has no KV allocation bundle"); }

    PagedKVPool& text_pool         = decoder->text_kv.pool();
    qwen3_6::PagedKVCache* backend = sequence.kv->backend ? backend_kv_cache() : nullptr;
    if (sequence.kv->backend && backend == nullptr) {
        throw std::logic_error("backend KV bundle without a backend cache");
    }

    const std::int32_t live_slot = LinearStateSlots::current_state_slot(lane, max_concurrency);
    const std::int32_t check_slot =
        LinearStateSlots::turn_checkpoint_state_slot(lane, max_concurrency);

    // Sizes are derived before anything is written, so the header is complete on the first pass
    // and the payload can stream straight through afterwards.
    const auto pool_bytes = [](PagedKVPool& pool, const PagedKVAllocation& allocation) {
        std::uint64_t group = 0;
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            group += pool.page_group_bytes(plane);
        }
        return std::pair<std::uint64_t, std::uint64_t>{group,
                                                       group * allocation.page_ids().size()};
    };
    const auto linear_bytes = [&](const std::vector<Tensor>& layers, int slot_dim) {
        std::uint64_t total = 0;
        for (const Tensor& layer : layers) {
            total += 2ULL * linear_slot_view(layer, live_slot, slot_dim).bytes();
        }
        return total;
    };
    const auto [text_group, text_total] = pool_bytes(text_pool, sequence.kv->text);
    std::uint64_t backend_group = 0;
    std::uint64_t backend_total = 0;
    if (backend != nullptr) {
        std::tie(backend_group, backend_total) = pool_bytes(backend->pool(), *sequence.kv->backend);
    }
    const std::vector<std::byte> identity_blob = sequence.prefix_identity.serialize();

    runtime::SlotFileHeader header{};
    std::memcpy(header.magic, runtime::kSlotFileMagic, sizeof(header.magic));
    header.format              = runtime::kSlotFileFormat;
    header.header_bytes        = static_cast<std::uint32_t>(sizeof(runtime::SlotFileHeader));
    header.model_id_hash       = slot_identity_hash(identity);
    header.kv_dtype            = static_cast<std::uint32_t>(kv_dtype);
    header.kv_storage          = static_cast<std::uint32_t>(kv_storage);
    header.kv_quant_group      = kv_quant_group;
    header.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    header.kv_capacity         = kv_capacity;
    header.page_size           = static_cast<std::uint32_t>(kPagedKVPageSize);
    header.kv_packed_v         = kv_packed_v ? 1U : 0U;
    header.kv_rotate_k         = kv_rotate_k ? 1U : 0U;
    header.kv_rotate_v         = kv_rotate_v ? 1U : 0U;
    header.has_backend_kv      = backend != nullptr ? 1U : 0U;
    header.text_plane_count    = static_cast<std::uint32_t>(text_pool.plane_count());
    header.backend_plane_count =
        backend != nullptr ? static_cast<std::uint32_t>(backend->pool().plane_count()) : 0U;
    header.text_page_group_bytes    = text_group;
    header.backend_page_group_bytes = backend_group;

    header.execution_frontier       = sequence.execution_frontier;
    header.ledger_frontier          = sequence.ledger_frontier;
    header.text_kv_valid            = sequence.text_kv_valid;
    header.mtp_kv_valid             = sequence.mtp_kv_valid;
    header.dflash_context_frontier  = sequence.dflash_context_frontier;
    header.rope_delta               = sequence.rope_delta;
    header.mtp_draft_count          = sequence.mtp_draft_count;
    header.turn_checkpoint_valid    = sequence.turn_checkpoint.valid ? 1U : 0U;
    header.turn_checkpoint_frontier = sequence.turn_checkpoint.frontier;
    header.tail_hidden_valid        = sequence.tail_hidden_valid ? 1U : 0U;

    header.ledger_bytes          = sequence.ledger.size() * sizeof(TokenId);
    header.prefix_identity_bytes = identity_blob.size();
    header.mtp_drafts_bytes      = sizeof(sequence.mtp_drafts);
    header.tail_hidden_bytes     = sequence.tail_hidden.bytes();
    header.turn_checkpoint_hidden_bytes = sequence.turn_checkpoint_hidden.bytes();
    header.text_kv_bytes                = text_total;
    header.backend_kv_bytes             = backend_total;
    header.linear_conv_bytes = linear_bytes(decoder->linear_attention.conv, kLinearConvSlotDim);
    header.linear_recurrent_bytes =
        linear_bytes(decoder->linear_attention.recurrent, kLinearRecurrentSlotDim);

    // Queued device work must land before the writer starts reading those buffers.
    device.synchronize();

    // Written through a sibling temporary so a crash mid-write cannot leave a half file that
    // would later pass the header check and restore as plausible attention state.
    const std::filesystem::path final_path(path);
    const std::filesystem::path temp_path = final_path.string() + ".partial";
    std::uint64_t written                 = 0;
    try {
        SlotWriter writer(temp_path);
        writer.host(&header, sizeof(header));
        writer.host(sequence.ledger.data(), static_cast<std::size_t>(header.ledger_bytes));
        writer.host(identity_blob.data(), identity_blob.size());
        writer.host(sequence.mtp_drafts.data(), sizeof(sequence.mtp_drafts));
        writer.device(sequence.tail_hidden);
        writer.device(sequence.turn_checkpoint_hidden);

        const auto stream_kv = [&](PagedKVPool& pool, const PagedKVAllocation& allocation) {
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                for (const std::int32_t page : allocation.page_ids()) {
                    writer.page_group(pool, plane, page, device.stream);
                }
            }
        };
        stream_kv(text_pool, sequence.kv->text);
        if (backend != nullptr) { stream_kv(backend->pool(), *sequence.kv->backend); }

        const auto stream_linear = [&](const std::vector<Tensor>& layers, int slot_dim) {
            for (const Tensor& layer : layers) {
                writer.device(linear_slot_view(layer, live_slot, slot_dim));
                writer.device(linear_slot_view(layer, check_slot, slot_dim));
            }
        };
        stream_linear(decoder->linear_attention.conv, kLinearConvSlotDim);
        stream_linear(decoder->linear_attention.recurrent, kLinearRecurrentSlotDim);

        writer.finish();
        written = writer.written();
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_path, ignored);
        throw;
    }
    std::filesystem::rename(temp_path, final_path);

    return runtime::SlotTransferResult{
        .tokens = static_cast<std::uint32_t>(sequence.ledger.size()),
        .bytes  = written,
    };
}

runtime::SlotTransferResult ProgramImplCore::restore_retained_lane(std::uint32_t lane,
                                                                   const std::string& path,
                                                                   std::string_view identity) {
    if (lane >= max_concurrency) { throw std::out_of_range("slot lane is out of range"); }

    SlotReader reader{std::filesystem::path(path)};
    runtime::SlotFileHeader header{};
    reader.host(&header, sizeof(header), "the file header");

    if (std::memcmp(header.magic, runtime::kSlotFileMagic, sizeof(header.magic)) != 0) {
        throw std::invalid_argument("slot file magic does not match");
    }
    if (header.format != runtime::kSlotFileFormat ||
        header.header_bytes != sizeof(runtime::SlotFileHeader)) {
        throw std::invalid_argument("slot file was written by an incompatible format version");
    }
    // Every one of these would reinterpret unrelated bytes as attention state, so the check is
    // exhaustive rather than advisory, and it runs before the lane is disturbed.
    if (header.model_id_hash != slot_identity_hash(identity) ||
        header.kv_dtype != static_cast<std::uint32_t>(kv_dtype) ||
        header.kv_storage != static_cast<std::uint32_t>(kv_storage) ||
        header.kv_quant_group != kv_quant_group ||
        header.speculative_backend != static_cast<std::uint32_t>(speculative_backend) ||
        header.kv_capacity != kv_capacity ||
        header.page_size != static_cast<std::uint32_t>(kPagedKVPageSize) ||
        header.kv_packed_v != (kv_packed_v ? 1U : 0U) ||
        header.kv_rotate_k != (kv_rotate_k ? 1U : 0U) ||
        header.kv_rotate_v != (kv_rotate_v ? 1U : 0U) ||
        (header.has_backend_kv != 0U) != (backend_kv_cache() != nullptr)) {
        throw std::invalid_argument("slot file describes a different model or KV configuration");
    }

    PagedKVPool& text_pool = decoder->text_kv.pool();
    if (header.text_plane_count != static_cast<std::uint32_t>(text_pool.plane_count()) ||
        header.text_page_group_bytes == 0 ||
        header.text_kv_bytes % header.text_page_group_bytes != 0) {
        throw std::invalid_argument("slot file KV geometry does not match this pool");
    }
    const auto text_pages =
        static_cast<std::uint32_t>(header.text_kv_bytes / header.text_page_group_bytes);
    std::uint32_t backend_pages    = 0;
    qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (header.has_backend_kv != 0U) {
        if (backend == nullptr || header.backend_page_group_bytes == 0 ||
            header.backend_kv_bytes % header.backend_page_group_bytes != 0 ||
            header.backend_plane_count !=
                static_cast<std::uint32_t>(backend->pool().plane_count())) {
            throw std::invalid_argument("slot file backend KV geometry does not match this pool");
        }
        backend_pages =
            static_cast<std::uint32_t>(header.backend_kv_bytes / header.backend_page_group_bytes);
    }

    // Past this point the lane is being rewritten, so any failure must leave it cleared rather
    // than half restored.
    SequenceState& sequence = sequences[lane];
    clear_lane(sequence, requests[lane]);
    try {
        if (header.ledger_bytes % sizeof(TokenId) != 0) {
            throw std::invalid_argument("slot file token ledger is misaligned");
        }
        sequence.ledger.resize(static_cast<std::size_t>(header.ledger_bytes / sizeof(TokenId)));
        reader.host(sequence.ledger.data(), static_cast<std::size_t>(header.ledger_bytes),
                    "the token ledger");
        sequence.prefix_identity.deserialize(
            reader.host_block(header.prefix_identity_bytes, "the prefix identity"));
        if (sequence.prefix_identity.size() != sequence.ledger.size()) {
            throw std::invalid_argument("slot file identity length disagrees with its ledger");
        }
        if (header.mtp_drafts_bytes != sizeof(sequence.mtp_drafts)) {
            throw std::invalid_argument("slot file speculative draft block has the wrong size");
        }
        reader.host(sequence.mtp_drafts.data(), sizeof(sequence.mtp_drafts),
                    "the speculative drafts");

        // Deliberately left unbound. A lane that finished a request and stayed retained has
        // released its table row, and admission binds again on the next request; binding here
        // would make that second bind fail on an already-bound bundle.
        reserve_sequence_kv(sequence, text_pages, backend_pages);
        sequence.kv->text.materialize_pages(text_pages, device.stream);
        if (sequence.kv->backend) {
            sequence.kv->backend->materialize_pages(backend_pages, device.stream);
        }

        reader.device(sequence.tail_hidden, header.tail_hidden_bytes, "the tail hidden state");
        reader.device(sequence.turn_checkpoint_hidden, header.turn_checkpoint_hidden_bytes,
                      "the turn checkpoint hidden state");

        const auto load_kv = [&](PagedKVPool& pool, const PagedKVAllocation& allocation,
                                 const char* what) {
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                for (const std::int32_t page : allocation.page_ids()) {
                    reader.page_group(pool, plane, page, device.stream, what);
                }
            }
        };
        load_kv(text_pool, sequence.kv->text, "the text KV payload");
        if (sequence.kv->backend) {
            load_kv(backend->pool(), *sequence.kv->backend, "the backend KV payload");
        }

        const auto load_linear = [&](const std::vector<Tensor>& layers, int slot_dim,
                                     std::uint64_t declared, const char* what) {
            const std::int32_t live = LinearStateSlots::current_state_slot(lane, max_concurrency);
            const std::int32_t check =
                LinearStateSlots::turn_checkpoint_state_slot(lane, max_concurrency);
            std::uint64_t consumed = 0;
            for (const Tensor& layer : layers) {
                for (const std::int32_t slot : {live, check}) {
                    const Tensor view = linear_slot_view(layer, slot, slot_dim);
                    reader.device(view, view.bytes(), what);
                    consumed += view.bytes();
                }
            }
            if (consumed != declared) {
                throw std::invalid_argument(std::string("slot file ") + what +
                                            " size disagrees with this model");
            }
        };
        load_linear(decoder->linear_attention.conv, kLinearConvSlotDim, header.linear_conv_bytes,
                    "the GDN convolution state");
        load_linear(decoder->linear_attention.recurrent, kLinearRecurrentSlotDim,
                    header.linear_recurrent_bytes, "the GDN recurrent state");
        reader.expect_end();

        device.synchronize();

        sequence.execution_frontier       = header.execution_frontier;
        sequence.ledger_frontier          = header.ledger_frontier;
        sequence.text_kv_valid            = header.text_kv_valid;
        sequence.mtp_kv_valid             = header.mtp_kv_valid;
        sequence.dflash_context_frontier  = header.dflash_context_frontier;
        sequence.rope_delta               = header.rope_delta;
        sequence.mtp_draft_count          = header.mtp_draft_count;
        sequence.turn_checkpoint.valid    = header.turn_checkpoint_valid != 0U;
        sequence.turn_checkpoint.frontier = header.turn_checkpoint_frontier;
        sequence.tail_hidden_valid        = header.tail_hidden_valid != 0U;
        sequence.retained                 = true;
    } catch (...) {
        clear_lane(sequence, requests[lane]);
        throw;
    }

    return runtime::SlotTransferResult{
        .tokens = static_cast<std::uint32_t>(sequence.ledger.size()),
        .bytes  = sizeof(header) + header.ledger_bytes + header.prefix_identity_bytes +
                 header.mtp_drafts_bytes + header.tail_hidden_bytes +
                 header.turn_checkpoint_hidden_bytes + header.text_kv_bytes +
                 header.backend_kv_bytes + header.linear_conv_bytes +
                 header.linear_recurrent_bytes,
    };
}

std::uint32_t ProgramImplCore::erase_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return 0; }
    const auto tokens = static_cast<std::uint32_t>(sequences[lane].ledger.size());
    evict_retained_lane(lane);
    return tokens;
}

std::uint32_t ProgramImplCore::retained_token_count_lane(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) ? static_cast<std::uint32_t>(sequences[lane].ledger.size()) : 0U;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
