#include "runtime/conversation_cache.h"

using namespace ninfer::runtime;

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ConversationCheckpoint make_checkpoint(std::uint32_t frontier,
                                                CheckpointKind kind) {
    ConversationCheckpoint checkpoint;
    checkpoint.kind     = kind;
    checkpoint.frontier = frontier;
    // Small representative state blobs so byte accounting has something to count.
    checkpoint.linear_conv.assign(frontier % 7 + 1, std::byte{0});
    return checkpoint;
}

ConversationKvPayload make_payload(std::size_t pages) {
    ConversationKvPayload payload;
    payload.has_backend_kv        = false;
    payload.text_plane_count      = 1;
    payload.text_page_group_bytes = 64;
    payload.text_pages.resize(1);
    for (std::size_t p = 0; p < pages; ++p) {
        payload.text_pages.front().push_back(
            std::make_shared<const std::vector<std::byte>>(64, std::byte{0}));
    }
    return payload;
}

} // namespace

int main() {
    int failures = 0;

    ConversationCache::Options options;
    options.ram_budget_bytes                 = 700;
    options.max_checkpoints_per_conversation = 4;
    options.min_checkpoint_spacing_tokens    = 100;
    ConversationCache cache(options);
    failures += check(cache.enabled(), "cache with a positive budget reports disabled");
    failures += check(!ConversationCache().enabled(), "default cache reports enabled");

    ConversationRecord& first = cache.create("conversation-1");
    first.payload             = make_payload(8);
    first.checkpoints.push_back(make_checkpoint(10, CheckpointKind::TurnBoundary));
    const std::uint64_t first_id = first.id;

    ConversationRecord& second = cache.create("conversation-2");
    second.payload             = make_payload(8);
    second.checkpoints.push_back(make_checkpoint(20, CheckpointKind::TurnBoundary));
    const std::uint64_t second_id = second.id;

    failures += check(cache.find(first_id) != nullptr && cache.find(second_id) != nullptr,
                      "created conversations are not findable by id");

    // Redundant pruning: a grid checkpoint closer than the spacing to its newer neighbour goes;
    // turn boundaries always survive.
    ConversationRecord& pruned = cache.create("conversation-3");
    pruned.checkpoints.push_back(make_checkpoint(500, CheckpointKind::TurnBoundary));
    pruned.checkpoints.push_back(make_checkpoint(480, CheckpointKind::PeriodicGrid));
    pruned.checkpoints.push_back(make_checkpoint(300, CheckpointKind::PeriodicGrid));
    pruned.checkpoints.push_back(make_checkpoint(200, CheckpointKind::TurnBoundary));
    const std::size_t evicted = cache.prune_redundant_checkpoints(pruned);
    failures += check(evicted == 1 && pruned.checkpoints.size() == 3 &&
                          pruned.checkpoints[1].frontier == 300,
                      "pruning kept a redundant near-neighbour grid checkpoint");

    // Budget enforcement: conversation-2 stays coldest by touch order, so it is dropped whole.
    cache.touch(first);
    cache.touch(pruned);
    const std::size_t freed = cache.enforce_ram_budget();
    const ConversationRecord* survivor = cache.find(second_id);
    failures += check(freed > 0 && survivor == nullptr,
                      "budget enforcement did not drop the coldest conversation");
    for (const auto& record : cache.records()) { (void)record; }

    cache.erase(first_id);
    failures += check(cache.find(first_id) == nullptr && cache.size() == 1,
                      "erase did not remove exactly the named conversation");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
