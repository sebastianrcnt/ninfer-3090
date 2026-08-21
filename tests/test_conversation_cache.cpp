// Conversation checkpoint cache: retention policy, host-page sharing, budget eviction, divergence
// selection, and the NINFSLOT v4 container.
//
// Everything exercised here is the target-agnostic host half of the feature, so it runs without a
// GPU. The device half - what a checkpoint's state blob contains and how it is written back - is
// owned by the target and is checked against a full-prefix execution instead.

#include "runtime/cache/conversation_cache.h"
#include "runtime/cache/conversation_file.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::runtime::ConversationCache;
using ninfer::runtime::ConversationCachePolicy;
using ninfer::runtime::ConversationCapture;
using ninfer::runtime::ConversationCheckpoint;
using ninfer::runtime::ConversationFileCatalog;
using ninfer::runtime::ConversationGeometry;
using ninfer::runtime::ConversationSnapshot;
using ninfer::runtime::HostKvPayload;
using ninfer::runtime::HostPagePlane;

constexpr std::uint32_t kPageSize    = 64;
constexpr std::uint64_t kGroupBytes  = 4096;
constexpr std::uint32_t kStateBytes  = 256;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ConversationGeometry geometry() {
    ConversationGeometry out;
    out.model_id_hash          = 0x1234'5678'9abc'def0ULL;
    out.kv_capacity            = 65536;
    out.page_size              = kPageSize;
    out.text_plane_count       = 2;
    out.backend_plane_count    = 0;
    out.checkpoint_state_bytes = kStateBytes;
    return out;
}

std::uint32_t pages_for(std::uint32_t tokens) {
    return ninfer::runtime::conversation_pages_for_tokens(tokens, kPageSize);
}

// A capture whose ledger is a deterministic function of (seed, frontier), so an extension of the
// same seed continues the same conversation and a different seed diverges at `branch_at`.
ConversationCapture make_capture(std::uint32_t frontier, std::uint32_t shared_frontier,
                                 std::uint32_t seed, std::uint32_t branch_at, bool turn_boundary) {
    ConversationCapture capture;
    capture.geometry        = geometry();
    capture.shared_frontier = shared_frontier;
    capture.ledger.resize(static_cast<std::size_t>(frontier) + 1);
    for (std::size_t index = 0; index < capture.ledger.size(); ++index) {
        // Below `branch_at` the ledger matches the trunk (seed 0) exactly; above it the seed
        // makes the continuation diverge.
        capture.ledger[index] =
            static_cast<TokenId>(index * 10 + (index < branch_at ? 0U : seed));
    }
    capture.identity.assign(capture.ledger.size(), std::byte{0x5a});

    const std::uint32_t text_pages = pages_for(frontier);
    capture.shared_text_pages      = std::min(text_pages, shared_frontier / kPageSize);
    capture.new_text.planes.resize(2);
    for (auto& plane : capture.new_text.planes) {
        plane.group_bytes = kGroupBytes;
        for (std::uint32_t page = capture.shared_text_pages; page < text_pages; ++page) {
            plane.pages.push_back(std::make_shared<const std::vector<std::byte>>(
                static_cast<std::size_t>(kGroupBytes), std::byte{static_cast<unsigned char>(page)}));
        }
    }
    capture.checkpoint = ConversationCheckpoint{
        .frontier      = frontier,
        .text_pages    = text_pages,
        .backend_pages = 0,
        .turn_boundary = static_cast<std::uint8_t>(turn_boundary ? 1 : 0),
        .state         = std::vector<std::byte>(kStateBytes, std::byte{0x11}),
    };
    return capture;
}

// Selector standing in for the target's exact prepared-prefix comparison: the greatest checkpoint
// whose ledger prefix agrees with the prompt token for token.
ConversationCache::Selector selector_for(const std::vector<TokenId>& prompt) {
    return [prompt](const std::vector<TokenId>& ledger, const std::vector<std::byte>&,
                    const std::vector<ConversationCheckpoint>& checkpoints)
               -> std::optional<std::size_t> {
        for (std::size_t index = checkpoints.size(); index-- > 0;) {
            const std::uint32_t frontier = checkpoints[index].frontier;
            if (frontier == 0 || frontier > prompt.size() ||
                static_cast<std::size_t>(frontier) + 1 > ledger.size()) {
                continue;
            }
            if (std::equal(ledger.begin(), ledger.begin() + frontier, prompt.begin())) {
                return index;
            }
        }
        return std::nullopt;
    };
}

std::vector<TokenId> prompt_from(const std::vector<TokenId>& ledger, std::size_t tokens) {
    return std::vector<TokenId>(ledger.begin(), ledger.begin() + static_cast<std::ptrdiff_t>(tokens));
}

std::uint64_t big_budget() { return 1ULL << 40; }

int test_retention_policy() {
    int failures = 0;
    ConversationCachePolicy options;
    options.ram_budget_bytes    = big_budget();
    options.context_checkpoints = 3;
    options.checkpoint_min_step = 1000;
    ConversationCache cache(options, geometry());

    ConversationCache::EntryId entry = 0;
    std::uint32_t shared             = 0;
    // Boundaries at 1000, 1100 (too close), 2000, 3000, 4000, 5000.
    for (const std::uint32_t frontier : {1000U, 1100U, 2000U, 3000U, 4000U, 5000U}) {
        entry  = cache.park(entry, make_capture(frontier, shared, 0, 0, true));
        shared = frontier;
        failures += check(entry != 0, "park did not return a conversation");
    }

    const auto ledger = make_capture(5000, 0, 0, 0, true).ledger;
    const auto match  = cache.select(selector_for(prompt_from(ledger, 5001)));
    failures += check(match.has_value(), "the newest boundary was not selectable");
    failures += check(match && match->frontier == 5000, "the newest boundary was not retained");

    const auto snapshot = cache.acquire(*match);
    failures += check(snapshot != nullptr, "the resident conversation could not be acquired");
    if (snapshot == nullptr) { return failures + 1; }
    failures += check(snapshot->checkpoints.size() == 4,
                      "the per-conversation checkpoint cap was not enforced");
    for (const ConversationCheckpoint& checkpoint : snapshot->checkpoints) {
        failures += check(checkpoint.frontier != 1100,
                          "a boundary inside checkpoint_min_step was retained");
    }
    failures += check(snapshot->checkpoints.back().frontier == 5000,
                      "the newest boundary was dropped by trimming");
    failures += check(snapshot->text.page_count() == pages_for(5000),
                      "the shared payload does not cover the newest boundary");

    // A mid-history prompt lands on the greatest eligible retained boundary, not on token zero.
    const auto mid = cache.select(selector_for(prompt_from(ledger, 3500)));
    failures += check(mid.has_value() && mid->frontier == 3000,
                      "mid-history selection did not choose the greatest eligible frontier");
    return failures;
}

int test_page_sharing_and_branching() {
    int failures = 0;
    ConversationCachePolicy options;
    options.ram_budget_bytes    = big_budget();
    options.context_checkpoints = 8;
    options.checkpoint_min_step = 0;
    ConversationCache cache(options, geometry());

    const ConversationCache::EntryId first  = cache.park(0, make_capture(2000, 0, 0, 0, true));
    const std::uint64_t after_first         = cache.stats().resident_bytes;
    const ConversationCache::EntryId second = cache.park(first, make_capture(2100, 2000, 0, 0, true));
    failures += check(first == second, "an extension of the same conversation created a new entry");

    const std::uint64_t after_append = cache.stats().resident_bytes;
    const std::uint64_t appended_pages =
        static_cast<std::uint64_t>(pages_for(2100) - 2000 / kPageSize) * 2ULL * kGroupBytes;
    failures += check(after_append - after_first < appended_pages + kGroupBytes,
                      "an append recopied host pages it already shared");

    // A divergence below the shared frontier becomes a branch that shares the parent's pages.
    const ConversationCache::EntryId branch =
        cache.park(second, make_capture(2200, 1000, 7, 1000, true));
    failures += check(branch != 0 && branch != second, "a divergent capture did not branch");
    failures += check(cache.stats().conversations == 2, "the branch did not become its own entry");

    const std::uint64_t after_branch = cache.stats().resident_bytes;
    const std::uint64_t unshared_branch =
        static_cast<std::uint64_t>(pages_for(2200)) * 2ULL * kGroupBytes;
    const std::uint64_t shared_prefix =
        static_cast<std::uint64_t>(1000 / kPageSize) * 2ULL * kGroupBytes;
    // The slack covers the branch's own ledger, identity, and checkpoint state; what the check
    // establishes is that the shared page payload was not charged a second time.
    failures +=
        check(after_branch - after_append < unshared_branch - shared_prefix + (64ULL << 10),
              "a branch was charged for host pages it shares with its parent");

    // Both continuations remain selectable at their own frontiers.
    const auto trunk_ledger  = make_capture(2100, 0, 0, 0, true).ledger;
    const auto branch_ledger = make_capture(2200, 0, 7, 1000, true).ledger;
    const auto trunk_match   = cache.select(selector_for(prompt_from(trunk_ledger, 2101)));
    const auto branch_match  = cache.select(selector_for(prompt_from(branch_ledger, 2201)));
    failures += check(trunk_match && trunk_match->frontier == 2100,
                      "the trunk conversation stopped matching after branching");
    failures += check(branch_match && branch_match->frontier == 2200,
                      "the branch conversation did not match its own frontier");
    if (trunk_match && branch_match) {
        const auto trunk_snapshot  = cache.acquire(*trunk_match);
        const auto branch_snapshot = cache.acquire(*branch_match);
        bool shared_objects        = trunk_snapshot != nullptr && branch_snapshot != nullptr;
        for (std::uint32_t page = 0; shared_objects && page < 1000 / kPageSize; ++page) {
            shared_objects = trunk_snapshot->text.planes[0].pages[page].get() ==
                             branch_snapshot->text.planes[0].pages[page].get();
        }
        failures += check(shared_objects,
                          "a branch copied its parent's host pages instead of sharing them");
    }
    return failures;
}

int test_ram_budget_eviction() {
    int failures = 0;
    ConversationCachePolicy options;
    // Room for roughly one conversation's payload.
    options.ram_budget_bytes    = 40ULL * kGroupBytes;
    options.context_checkpoints = 4;
    options.checkpoint_min_step = 0;
    ConversationCache cache(options, geometry());

    const ConversationCache::EntryId first = cache.park(0, make_capture(1000, 0, 1, 0, true));
    cache.pin(first);
    for (std::uint32_t seed = 2; seed < 6; ++seed) {
        (void)cache.park(0, make_capture(1000, 0, seed, 0, true));
    }
    failures += check(cache.stats().resident_bytes <= options.ram_budget_bytes,
                      "the global RAM budget was exceeded");

    const auto pinned_ledger = make_capture(1000, 0, 1, 0, true).ledger;
    const auto pinned_match  = cache.select(selector_for(prompt_from(pinned_ledger, 1001)));
    failures += check(pinned_match.has_value(),
                      "a conversation held by a lane was evicted by the budget");
    cache.unpin(first);
    return failures;
}

int test_file_container(const std::filesystem::path& root) {
    int failures = 0;
    ConversationSnapshot snapshot;
    snapshot.geometry = geometry();
    snapshot.ledger.assign(129, 0);
    for (std::size_t index = 0; index < snapshot.ledger.size(); ++index) {
        snapshot.ledger[index] = static_cast<TokenId>(index);
    }
    snapshot.identity.assign(8, std::byte{0x33});
    snapshot.text.planes.resize(2);
    for (auto& plane : snapshot.text.planes) {
        plane.group_bytes = kGroupBytes;
        for (std::uint32_t page = 0; page < pages_for(128); ++page) {
            plane.pages.push_back(std::make_shared<const std::vector<std::byte>>(
                static_cast<std::size_t>(kGroupBytes), std::byte{0x7e}));
        }
    }
    snapshot.checkpoints.push_back(ConversationCheckpoint{
        .frontier      = 64,
        .text_pages    = 1,
        .backend_pages = 0,
        .turn_boundary = 1,
        .state         = std::vector<std::byte>(kStateBytes, std::byte{0xa1}),
    });
    snapshot.checkpoints.push_back(ConversationCheckpoint{
        .frontier      = 128,
        .text_pages    = 2,
        .backend_pages = 0,
        .turn_boundary = 1,
        .state         = std::vector<std::byte>(kStateBytes, std::byte{0xa2}),
    });

    const std::filesystem::path path = root / "round-trip.ninfslot";
    const std::uint64_t written      = ninfer::runtime::write_conversation_file(path, snapshot);
    failures += check(written == std::filesystem::file_size(path),
                      "the published file size disagrees with the writer");
    failures += check(!std::filesystem::exists(path.string() + ".partial"),
                      "the sibling temporary survived a successful publish");

    const ConversationFileCatalog catalog =
        ninfer::runtime::read_conversation_catalog(path, geometry());
    failures += check(catalog.ledger == snapshot.ledger, "the catalog ledger did not round-trip");
    failures += check(catalog.checkpoints.size() == 2 && catalog.checkpoints[1].frontier == 128,
                      "the catalog checkpoint records did not round-trip");
    failures += check(catalog.checkpoints[0].state.empty(),
                      "reading the catalog also read the checkpoint payload");

    const ConversationSnapshot loaded = ninfer::runtime::read_conversation_payload(catalog);
    failures += check(loaded.checkpoints.size() == 2 &&
                          loaded.checkpoints[0].state == snapshot.checkpoints[0].state &&
                          loaded.checkpoints[1].state == snapshot.checkpoints[1].state,
                      "the checkpoint state did not round-trip");
    failures += check(loaded.text.page_count() == snapshot.text.page_count() &&
                          *loaded.text.planes[1].pages[1] == *snapshot.text.planes[1].pages[1],
                      "the KV payload did not round-trip");

    // A snapshot from another build must be refused rather than reinterpreted.
    ConversationGeometry other = geometry();
    other.model_id_hash ^= 1ULL;
    bool refused = false;
    try {
        (void)ninfer::runtime::read_conversation_catalog(path, other);
    } catch (const std::exception&) { refused = true; }
    failures += check(refused, "a snapshot from a different build was accepted");

    // Truncation must be refused, not read as a shorter conversation.
    const std::filesystem::path truncated = root / "truncated.ninfslot";
    std::filesystem::copy_file(path, truncated,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) / 2);
    refused = false;
    try {
        const ConversationFileCatalog partial =
            ninfer::runtime::read_conversation_catalog(truncated, geometry());
        (void)ninfer::runtime::read_conversation_payload(partial);
    } catch (const std::exception&) { refused = true; }
    failures += check(refused, "a truncated snapshot was accepted");

    // A corrupt header must be refused before any payload is touched.
    const std::filesystem::path corrupt = root / "corrupt.ninfslot";
    std::filesystem::copy_file(path, corrupt, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream out(corrupt, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(0);
        const char bad[8] = {'N', 'O', 'T', 'S', 'L', 'O', 'T', '!'};
        out.write(bad, sizeof(bad));
    }
    refused = false;
    try {
        (void)ninfer::runtime::read_conversation_catalog(corrupt, geometry());
    } catch (const std::exception&) { refused = true; }
    failures += check(refused, "a snapshot with a foreign magic was accepted");
    return failures;
}

int test_disk_tier(const std::filesystem::path& root) {
    int failures = 0;
    const std::filesystem::path dir = root / "disk";
    std::filesystem::create_directories(dir);

    const auto ledger = make_capture(2000, 0, 3, 0, true).ledger;
    {
        ConversationCachePolicy options;
        options.ram_budget_bytes    = big_budget();
        options.disk_dir            = dir;
        options.disk_budget_bytes   = 1ULL << 30;
        options.context_checkpoints = 4;
        options.checkpoint_min_step = 0;
        ConversationCache cache(options, geometry());
        (void)cache.park(0, make_capture(2000, 0, 3, 0, true));
        // Draining the writer is what the destructor does; give it the chance here.
    }

    std::uint32_t files = 0;
    for (const auto& item : std::filesystem::directory_iterator(dir)) {
        if (item.path().extension() == ".ninfslot") { ++files; }
        failures += check(item.path().extension() != ".partial",
                          "an unfinished snapshot temporary was left behind");
    }
    failures += check(files == 1, "the completed conversation was not published to disk");

    // A leftover temporary from a crash during replacement must not survive adoption, and the
    // previous valid file must still be adopted.
    { std::ofstream leftover(dir / "conv-0000ffff.ninfslot.partial", std::ios::binary); }

    ConversationCachePolicy options;
    options.ram_budget_bytes    = big_budget();
    options.disk_dir            = dir;
    options.disk_budget_bytes   = 1ULL << 30;
    options.context_checkpoints = 4;
    options.checkpoint_min_step = 0;
    ConversationCache restarted(options, geometry());
    const std::uint32_t adopted = restarted.adopt_disk_catalog(nullptr);
    failures += check(adopted == 1, "the durable catalog was not adopted at startup");
    failures += check(restarted.stats().resident_conversations == 0,
                      "startup read a conversation payload it did not need");
    failures += check(!std::filesystem::exists(dir / "conv-0000ffff.ninfslot.partial"),
                      "an unfinished snapshot temporary survived adoption");

    const auto match = restarted.select(selector_for(prompt_from(ledger, 2001)));
    failures += check(match && match->from_disk && match->frontier == 2000,
                      "the adopted conversation was not selectable from its catalog");
    if (match) {
        const auto snapshot = restarted.acquire(*match);
        failures += check(snapshot != nullptr && snapshot->newest_frontier() == 2000,
                          "the adopted conversation payload could not be read lazily");
        failures += check(restarted.stats().resident_conversations == 1,
                          "a lazily read conversation was not promoted to the hot tier");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "ninfer-conversation-cache-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root);

    failures += test_retention_policy();
    failures += test_page_sharing_and_branching();
    failures += test_ram_budget_eviction();
    failures += test_file_container(root);
    failures += test_disk_tier(root);

    std::filesystem::remove_all(root, ec);
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
