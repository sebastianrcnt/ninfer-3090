#include "runtime/cache/conversation_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer::runtime {
namespace {

void read_exact(std::ifstream& in, void* destination, std::uint64_t bytes, const char* what) {
    if (bytes == 0) { return; }
    in.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!in) {
        throw std::invalid_argument(std::string("conversation snapshot is truncated in ") + what);
    }
}

void write_exact(std::ofstream& out, const void* source, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    out.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
    if (!out) { throw std::runtime_error("failed while writing a conversation snapshot"); }
}

ConversationGeometry geometry_of(const ConversationFileHeader& header) noexcept {
    return ConversationGeometry{
        .model_id_hash          = header.model_id_hash,
        .kv_dtype               = header.kv_dtype,
        .kv_storage             = header.kv_storage,
        .kv_quant_group         = header.kv_quant_group,
        .speculative_backend    = header.speculative_backend,
        .kv_capacity            = header.kv_capacity,
        .page_size              = header.page_size,
        .text_plane_count       = header.text_plane_count,
        .backend_plane_count    = header.backend_plane_count,
        .checkpoint_state_bytes = header.checkpoint_state_bytes,
        .kv_packed_v            = header.kv_packed_v,
        .kv_rotate_k            = header.kv_rotate_k,
        .kv_rotate_v            = header.kv_rotate_v,
        .has_backend_kv         = header.has_backend_kv,
    };
}

std::vector<std::uint64_t> read_plane_bytes(std::ifstream& in, std::uint32_t planes,
                                            const char* what) {
    std::vector<std::uint64_t> out(planes);
    read_exact(in, out.data(), static_cast<std::uint64_t>(planes) * sizeof(std::uint64_t), what);
    for (const std::uint64_t bytes : out) {
        if (bytes == 0) {
            throw std::invalid_argument("conversation snapshot declares an empty KV plane");
        }
    }
    return out;
}

std::uint64_t payload_bytes(const std::vector<std::uint64_t>& plane_bytes, std::uint32_t pages) {
    std::uint64_t total = 0;
    for (const std::uint64_t bytes : plane_bytes) { total += bytes * pages; }
    return total;
}

void read_pages(std::ifstream& in, HostKvPayload& payload,
                const std::vector<std::uint64_t>& plane_bytes, std::uint32_t pages,
                const char* what) {
    payload.planes.clear();
    payload.planes.reserve(plane_bytes.size());
    for (const std::uint64_t group_bytes : plane_bytes) {
        HostPagePlane plane;
        plane.group_bytes = group_bytes;
        plane.pages.reserve(pages);
        for (std::uint32_t page = 0; page < pages; ++page) {
            auto block = std::make_shared<std::vector<std::byte>>(
                static_cast<std::size_t>(group_bytes));
            read_exact(in, block->data(), group_bytes, what);
            plane.pages.push_back(std::move(block));
        }
        payload.planes.push_back(std::move(plane));
    }
}

} // namespace

ConversationFileCatalog read_conversation_catalog(const std::filesystem::path& path,
                                                  const ConversationGeometry& expect) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::invalid_argument("conversation snapshot does not exist: " + path.string());
    }

    ConversationFileHeader header{};
    read_exact(in, &header, sizeof(header), "the file header");
    if (std::memcmp(header.magic, kConversationFileMagic, sizeof(header.magic)) != 0) {
        throw std::invalid_argument("conversation snapshot magic does not match");
    }
    if (header.format != kConversationFileFormat ||
        header.header_bytes != sizeof(ConversationFileHeader)) {
        throw std::invalid_argument(
            "conversation snapshot was written by an incompatible format version");
    }

    ConversationFileCatalog catalog;
    catalog.path     = path;
    catalog.geometry = geometry_of(header);
    if (!(catalog.geometry == expect)) {
        throw std::invalid_argument(
            "conversation snapshot describes a different model or KV configuration");
    }
    if (header.checkpoint_count == 0 || header.checkpoint_state_bytes == 0 ||
        header.text_plane_count == 0 || header.text_page_count == 0 ||
        header.ledger_bytes % sizeof(TokenId) != 0) {
        throw std::invalid_argument("conversation snapshot header is internally inconsistent");
    }
    if ((header.has_backend_kv != 0U) != (header.backend_plane_count != 0U)) {
        throw std::invalid_argument("conversation snapshot backend KV geometry is inconsistent");
    }

    const std::vector<std::uint64_t> text_planes =
        read_plane_bytes(in, header.text_plane_count, "the Text KV plane table");
    const std::vector<std::uint64_t> backend_planes =
        read_plane_bytes(in, header.backend_plane_count, "the backend KV plane table");
    if (payload_bytes(text_planes, header.text_page_count) != header.text_payload_bytes ||
        payload_bytes(backend_planes, header.backend_page_count) !=
            header.backend_payload_bytes) {
        throw std::invalid_argument("conversation snapshot KV payload size disagrees with its "
                                    "declared page geometry");
    }

    catalog.ledger.resize(static_cast<std::size_t>(header.ledger_bytes / sizeof(TokenId)));
    read_exact(in, catalog.ledger.data(), header.ledger_bytes, "the token ledger");
    catalog.identity.resize(static_cast<std::size_t>(header.identity_bytes));
    read_exact(in, catalog.identity.data(), header.identity_bytes, "the prefix identity");

    catalog.checkpoints.resize(header.checkpoint_count);
    std::uint32_t previous = 0;
    for (std::uint32_t index = 0; index < header.checkpoint_count; ++index) {
        ConversationCheckpointRecord record{};
        read_exact(in, &record, sizeof(record), "the checkpoint records");
        if (index != 0 && record.frontier <= previous) {
            throw std::invalid_argument("conversation snapshot checkpoints are not ascending");
        }
        if (record.frontier == 0 || record.text_pages == 0 ||
            record.text_pages > header.text_page_count ||
            record.backend_pages > header.backend_page_count) {
            throw std::invalid_argument("conversation snapshot checkpoint coverage is invalid");
        }
        previous                            = record.frontier;
        catalog.checkpoints[index].frontier = record.frontier;
        catalog.checkpoints[index].text_pages     = record.text_pages;
        catalog.checkpoints[index].backend_pages  = record.backend_pages;
        catalog.checkpoints[index].turn_boundary  = record.turn_boundary;
    }
    if (static_cast<std::uint64_t>(previous) + 1ULL != catalog.ledger.size()) {
        throw std::invalid_argument(
            "conversation snapshot ledger length disagrees with its newest checkpoint");
    }
    if (catalog.checkpoints.back().text_pages != header.text_page_count ||
        catalog.checkpoints.back().backend_pages != header.backend_page_count) {
        throw std::invalid_argument(
            "conversation snapshot stores pages its newest checkpoint does not cover");
    }

    std::error_code ec;
    catalog.file_bytes = std::filesystem::file_size(path, ec);
    return catalog;
}

ConversationSnapshot read_conversation_payload(const ConversationFileCatalog& catalog) {
    std::ifstream in(catalog.path, std::ios::binary);
    if (!in) {
        throw std::invalid_argument("conversation snapshot disappeared: " +
                                    catalog.path.string());
    }
    ConversationFileHeader header{};
    read_exact(in, &header, sizeof(header), "the file header");
    if (std::memcmp(header.magic, kConversationFileMagic, sizeof(header.magic)) != 0 ||
        header.format != kConversationFileFormat ||
        !(geometry_of(header) == catalog.geometry)) {
        throw std::invalid_argument("conversation snapshot changed under the catalog");
    }

    const std::vector<std::uint64_t> text_planes =
        read_plane_bytes(in, header.text_plane_count, "the Text KV plane table");
    const std::vector<std::uint64_t> backend_planes =
        read_plane_bytes(in, header.backend_plane_count, "the backend KV plane table");

    ConversationSnapshot snapshot;
    snapshot.geometry = catalog.geometry;
    snapshot.ledger.resize(static_cast<std::size_t>(header.ledger_bytes / sizeof(TokenId)));
    read_exact(in, snapshot.ledger.data(), header.ledger_bytes, "the token ledger");
    snapshot.identity.resize(static_cast<std::size_t>(header.identity_bytes));
    read_exact(in, snapshot.identity.data(), header.identity_bytes, "the prefix identity");

    snapshot.checkpoints = catalog.checkpoints;
    in.seekg(static_cast<std::streamoff>(sizeof(ConversationCheckpointRecord)) *
                 static_cast<std::streamoff>(header.checkpoint_count),
             std::ios::cur);
    for (ConversationCheckpoint& checkpoint : snapshot.checkpoints) {
        checkpoint.state.resize(header.checkpoint_state_bytes);
        read_exact(in, checkpoint.state.data(), header.checkpoint_state_bytes,
                   "the checkpoint state");
    }

    read_pages(in, snapshot.text, text_planes, header.text_page_count, "the Text KV payload");
    read_pages(in, snapshot.backend, backend_planes, header.backend_page_count,
               "the backend KV payload");

    in.peek();
    if (!in.eof()) { throw std::invalid_argument("conversation snapshot has trailing bytes"); }
    return snapshot;
}

std::uint64_t write_conversation_file(const std::filesystem::path& path,
                                      const ConversationSnapshot& snapshot) {
    if (snapshot.checkpoints.empty() || !snapshot.text.consistent() ||
        !snapshot.backend.consistent()) {
        throw std::logic_error("conversation snapshot is not publishable");
    }
    const ConversationGeometry& geometry = snapshot.geometry;

    ConversationFileHeader header{};
    std::memcpy(header.magic, kConversationFileMagic, sizeof(header.magic));
    header.format                 = kConversationFileFormat;
    header.header_bytes           = static_cast<std::uint32_t>(sizeof(ConversationFileHeader));
    header.model_id_hash          = geometry.model_id_hash;
    header.kv_dtype               = geometry.kv_dtype;
    header.kv_storage             = geometry.kv_storage;
    header.kv_quant_group         = geometry.kv_quant_group;
    header.speculative_backend    = geometry.speculative_backend;
    header.kv_capacity            = geometry.kv_capacity;
    header.page_size              = geometry.page_size;
    header.text_plane_count       = geometry.text_plane_count;
    header.backend_plane_count    = geometry.backend_plane_count;
    header.checkpoint_state_bytes = geometry.checkpoint_state_bytes;
    header.kv_packed_v            = geometry.kv_packed_v;
    header.kv_rotate_k            = geometry.kv_rotate_k;
    header.kv_rotate_v            = geometry.kv_rotate_v;
    header.has_backend_kv         = geometry.has_backend_kv;
    header.checkpoint_count       = static_cast<std::uint32_t>(snapshot.checkpoints.size());
    header.text_page_count        = snapshot.text.page_count();
    header.backend_page_count     = snapshot.backend.page_count();
    header.ledger_bytes           = snapshot.ledger.size() * sizeof(TokenId);
    header.identity_bytes         = snapshot.identity.size();
    for (const HostPagePlane& plane : snapshot.text.planes) {
        header.text_payload_bytes += plane.group_bytes * header.text_page_count;
    }
    for (const HostPagePlane& plane : snapshot.backend.planes) {
        header.backend_payload_bytes += plane.group_bytes * header.backend_page_count;
    }

    const std::filesystem::path temp = path.string() + ".partial";
    std::uint64_t written            = 0;
    try {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot open a conversation snapshot for writing: " +
                                     temp.string());
        }
        write_exact(out, &header, sizeof(header));
        for (const HostPagePlane& plane : snapshot.text.planes) {
            write_exact(out, &plane.group_bytes, sizeof(plane.group_bytes));
        }
        for (const HostPagePlane& plane : snapshot.backend.planes) {
            write_exact(out, &plane.group_bytes, sizeof(plane.group_bytes));
        }
        write_exact(out, snapshot.ledger.data(), header.ledger_bytes);
        write_exact(out, snapshot.identity.data(), header.identity_bytes);
        for (const ConversationCheckpoint& checkpoint : snapshot.checkpoints) {
            const ConversationCheckpointRecord record{
                .frontier      = checkpoint.frontier,
                .text_pages    = checkpoint.text_pages,
                .backend_pages = checkpoint.backend_pages,
                .turn_boundary = checkpoint.turn_boundary,
                .reserved      = {},
            };
            write_exact(out, &record, sizeof(record));
        }
        for (const ConversationCheckpoint& checkpoint : snapshot.checkpoints) {
            if (checkpoint.state.size() != geometry.checkpoint_state_bytes) {
                throw std::logic_error("conversation checkpoint state has the wrong size");
            }
            write_exact(out, checkpoint.state.data(), checkpoint.state.size());
        }
        const auto write_pages = [&](const HostKvPayload& payload) {
            for (const HostPagePlane& plane : payload.planes) {
                for (const HostPage& page : plane.pages) {
                    if (page == nullptr || page->size() != plane.group_bytes) {
                        throw std::logic_error("conversation KV page has the wrong size");
                    }
                    write_exact(out, page->data(), plane.group_bytes);
                }
            }
        };
        write_pages(snapshot.text);
        write_pages(snapshot.backend);
        out.flush();
        if (!out) { throw std::runtime_error("failed while writing a conversation snapshot"); }
        written = static_cast<std::uint64_t>(out.tellp());
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw;
    }
    std::filesystem::rename(temp, path);
    return written;
}

} // namespace ninfer::runtime
