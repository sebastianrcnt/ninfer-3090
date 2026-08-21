#include "runtime/conversation_cache.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

ConversationRecord& ConversationCache::create(std::string name) {
    if (!enabled()) { throw std::logic_error("conversation cache is disabled"); }
    ConversationRecord record;
    record.id   = next_id_++;
    record.name = std::move(name);
    records_.push_back(std::move(record));
    return records_.back();
}

ConversationRecord* ConversationCache::find(ConversationId id) {
    for (auto& record : records_) {
        if (record.id == id) { return &record; }
    }
    return nullptr;
}

const ConversationRecord* ConversationCache::find(ConversationId id) const {
    for (const auto& record : records_) {
        if (record.id == id) { return &record; }
    }
    return nullptr;
}

void ConversationCache::touch(ConversationRecord& record) { record.last_touch_tick = ++tick_; }

std::size_t ConversationCache::prune_redundant_checkpoints(ConversationRecord& record) {
    const auto spacing = options_.min_checkpoint_spacing_tokens;
    std::size_t evicted = 0;
    // Newest-first walk: a grid checkpoint is redundant when the next-newer retained boundary is
    // within the spacing. Turn boundaries always survive.
    for (std::size_t i = 1; i < record.checkpoints.size();) {
        const std::size_t newer = i - 1;
        ConversationCheckpoint& checkpoint = record.checkpoints[i];
        const bool redundant = checkpoint.kind == CheckpointKind::PeriodicGrid &&
                               record.checkpoints[newer].frontier - checkpoint.frontier <
                                   spacing;
        if (redundant) {
            record.checkpoints.erase(record.checkpoints.begin() + static_cast<long>(i));
            ++evicted;
            continue;
        }
        ++i;
    }
    // Per-conversation cap: drop the oldest periodic checkpoints until it holds.
    while (record.checkpoints.size() > options_.max_checkpoints_per_conversation) {
        auto victim = std::find_if(record.checkpoints.rbegin(), record.checkpoints.rend(),
                                   [](const ConversationCheckpoint& checkpoint) {
                                       return checkpoint.kind ==
                                              CheckpointKind::PeriodicGrid;
                                   });
        if (victim == record.checkpoints.rend()) {
            // Only turn boundaries remain; keep the newest and its predecessor.
            while (record.checkpoints.size() > 2) { record.checkpoints.pop_back(); }
            break;
        }
        record.checkpoints.erase(std::next(victim).base());
        ++evicted;
    }
    return evicted;
}

std::size_t ConversationCache::total_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& record : records_) {
        total += record.payload.payload_bytes();
        for (const auto& checkpoint : record.checkpoints) { total += checkpoint.state_bytes(); }
    }
    return total;
}

std::size_t ConversationCache::enforce_ram_budget() {
    const std::size_t before = total_bytes();
    if (before <= options_.ram_budget_bytes) { return 0; }

    // Pass 1: prune redundant historical checkpoints everywhere, cheapest first.
    for (auto& record : records_) { prune_redundant_checkpoints(record); }
    std::size_t freed = before - total_bytes();
    if (total_bytes() <= options_.ram_budget_bytes) { return freed; }

    // Pass 2: least-recently-touched historical checkpoints across conversations.
    struct Victim {
        ConversationRecord* record;
        std::size_t index;
        std::size_t bytes;
    };
    const auto collect_victims = [&]() {
        std::vector<Victim> victims;
        for (auto& record : records_) {
            for (std::size_t i = 0; i < record.checkpoints.size(); ++i) {
                if (record.checkpoints[i].kind != CheckpointKind::TurnBoundary || i == 0) {
                    continue;
                }
                victims.push_back(Victim{&record, i, record.checkpoints[i].state_bytes()});
            }
        }
        std::sort(victims.begin(), victims.end(), [](const Victim& a, const Victim& b) {
            return a.record->last_touch_tick < b.record->last_touch_tick;
        });
        return victims;
    };
    for (const Victim& victim : collect_victims()) {
        if (total_bytes() <= options_.ram_budget_bytes) { break; }
        if (victim.index < victim.record->checkpoints.size()) {
            freed += victim.bytes;
            victim.record->checkpoints.erase(
                victim.record->checkpoints.begin() + static_cast<long>(victim.index));
        }
    }
    if (total_bytes() <= options_.ram_budget_bytes) { return freed; }

    // Pass 3: drop whole least-recently-touched conversations (front of the list is maintained
    // by touch(), which re-splices entries to the back).
    while (!records_.empty() && total_bytes() > options_.ram_budget_bytes) {
        auto& oldest = records_.front();
        const std::size_t bytes = oldest.payload.payload_bytes();
        for (const auto& checkpoint : oldest.checkpoints) { freed += checkpoint.state_bytes(); }
        freed += bytes;
        records_.pop_front();
    }
    return freed;
}

void ConversationCache::erase(ConversationId id) {
    for (auto it = records_.begin(); it != records_.end(); ++it) {
        if (it->id == id) {
            records_.erase(it);
            return;
        }
    }
}

} // namespace ninfer::runtime
