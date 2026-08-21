#include "runtime/cache/conversation_cache.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::runtime {
namespace {

constexpr const char* kSnapshotExtension = ".ninfslot";

std::string snapshot_file_name(std::uint32_t serial) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "conv-%08x%s", serial, kSnapshotExtension);
    return buffer;
}

// A checkpoint is "redundant" when its two neighbours already bracket it closely: dropping it
// costs the least reachable frontier of any historical boundary in the conversation.
std::size_t least_valuable_historical(const std::vector<ConversationCheckpoint>& checkpoints) {
    // checkpoints.size() >= 2; the newest is never a candidate.
    std::size_t best        = 0;
    std::uint32_t best_gain = 0;
    bool have               = false;
    for (std::size_t index = 0; index + 1 < checkpoints.size(); ++index) {
        const std::uint32_t previous = index == 0 ? 0U : checkpoints[index - 1].frontier;
        const std::uint32_t gain     = checkpoints[index].frontier - previous;
        // A periodic grid point loses to a completed turn boundary at equal spacing.
        const std::uint32_t weighted =
            checkpoints[index].turn_boundary != 0 ? gain : gain / 2U;
        if (!have || weighted < best_gain) {
            have      = true;
            best      = index;
            best_gain = weighted;
        }
    }
    return best;
}

} // namespace

ConversationCache::ConversationCache(ConversationCachePolicy options,
                                     ConversationGeometry geometry)
    : options_(std::move(options)), geometry_(geometry) {
    if (options_.disk_enabled() && options_.disk_budget_bytes == 0) {
        throw std::invalid_argument(
            "a conversation cache directory requires a positive disk budget");
    }
    if (options_.disk_enabled() && !options_.ram_enabled()) {
        throw std::invalid_argument(
            "the durable conversation cache tier requires a hot-cache budget");
    }
    if (options_.disk_enabled()) {
        std::filesystem::create_directories(options_.disk_dir);
        writer_ = std::thread([this] { writer_loop(); });
    }
}

ConversationCache::~ConversationCache() {
    if (writer_.joinable()) {
        {
            std::lock_guard lock(writer_mutex_);
            writer_stopping_ = true;
        }
        writer_cv_.notify_all();
        writer_.join();
    }
}

ConversationCache::Entry* ConversationCache::find(EntryId id) noexcept {
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [id](const Entry& entry) { return entry.id == id; });
    return it == entries_.end() ? nullptr : &*it;
}

const ConversationCache::Entry* ConversationCache::find(EntryId id) const noexcept {
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [id](const Entry& entry) { return entry.id == id; });
    return it == entries_.end() ? nullptr : &*it;
}

void ConversationCache::adopt_pages(const HostKvPayload& payload) {
    for (const HostPagePlane& plane : payload.planes) {
        for (const HostPage& page : plane.pages) {
            if (page == nullptr) { continue; }
            if (++page_refs_[page.get()] == 1U) { resident_bytes_ += page->size(); }
        }
    }
}

void ConversationCache::release_pages(const HostKvPayload& payload) {
    for (const HostPagePlane& plane : payload.planes) {
        for (const HostPage& page : plane.pages) {
            if (page == nullptr) { continue; }
            const auto it = page_refs_.find(page.get());
            if (it == page_refs_.end()) { continue; }
            if (--it->second == 0U) {
                resident_bytes_ -= std::min<std::uint64_t>(resident_bytes_, page->size());
                page_refs_.erase(it);
            }
        }
    }
}

void ConversationCache::account_resident(Entry& entry, const ConversationSnapshot& snapshot) {
    adopt_pages(snapshot.text);
    adopt_pages(snapshot.backend);
    entry.resident_bytes = snapshot.state_bytes();
    resident_bytes_ += entry.resident_bytes;
}

void ConversationCache::release_resident(Entry& entry) {
    if (entry.snapshot == nullptr) { return; }
    release_pages(entry.snapshot->text);
    release_pages(entry.snapshot->backend);
    resident_bytes_ -= std::min(resident_bytes_, entry.resident_bytes);
    entry.resident_bytes = 0;
    entry.snapshot.reset();
}

std::uint32_t
ConversationCache::adopt_disk_catalog(const std::function<void(const std::string&)>& report) {
    if (!options_.disk_enabled()) { return 0; }
    std::lock_guard lock(mutex_);

    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (const auto& item : std::filesystem::directory_iterator(options_.disk_dir, ec)) {
        if (!item.is_regular_file()) { continue; }
        const std::filesystem::path& path = item.path();
        if (path.extension() == ".partial") {
            // A write that never completed. Its target, if any, is still the previous valid file.
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            continue;
        }
        if (path.extension() == kSnapshotExtension) { files.push_back(path); }
    }
    std::sort(files.begin(), files.end());

    std::uint32_t adopted = 0;
    std::uint32_t refused = 0;
    for (const std::filesystem::path& path : files) {
        std::optional<ConversationFileCatalog> catalog;
        try {
            catalog = read_conversation_catalog(path, geometry_);
        } catch (const std::exception& error) {
            ++refused;
            if (report) {
                report("dropping unusable conversation snapshot " + path.filename().string() +
                       ": " + error.what());
            }
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            continue;
        }
        Entry entry;
        entry.id         = next_id_++;
        entry.file       = path;
        entry.file_bytes = catalog->file_bytes;
        entry.last_use   = clock_++;
        entry.disk       = std::move(catalog);
        disk_bytes_ += entry.file_bytes;
        entries_.push_back(std::move(entry));
        ++adopted;

        const std::string stem = path.stem().string();
        if (stem.rfind("conv-", 0) == 0) {
            try {
                const auto serial = static_cast<std::uint32_t>(std::stoul(stem.substr(5), nullptr, 16));
                next_file_serial_ = std::max(next_file_serial_, serial + 1U);
            } catch (const std::exception&) {}
        }
    }
    if (refused != 0 && report) {
        report("conversation cache refused " + std::to_string(refused) + " snapshot file(s)");
    }
    return adopted;
}

std::optional<ConversationCache::Match> ConversationCache::select(const Selector& selector) {
    if (!options_.ram_enabled()) { return std::nullopt; }
    std::lock_guard lock(mutex_);

    std::optional<Match> best;
    const auto consider = [&](const Entry& entry, const std::vector<TokenId>& ledger,
                              const std::vector<std::byte>& identity,
                              const std::vector<ConversationCheckpoint>& checkpoints,
                              bool from_disk) {
        const std::optional<std::size_t> index = selector(ledger, identity, checkpoints);
        if (!index) { return; }
        const std::uint32_t frontier = checkpoints[*index].frontier;
        // RAM wins ties: the interactive tier restores a full snapshot in a fraction of the time
        // the same payload takes to come back off local disk.
        if (best && (best->frontier > frontier ||
                     (best->frontier == frontier && !best->from_disk))) {
            return;
        }
        best = Match{
            .entry      = entry.id,
            .checkpoint = *index,
            .frontier   = frontier,
            .from_disk  = from_disk,
        };
    };

    for (const Entry& entry : entries_) {
        if (entry.snapshot == nullptr) { continue; }
        consider(entry, entry.snapshot->ledger, entry.snapshot->identity,
                 entry.snapshot->checkpoints, false);
    }
    for (const Entry& entry : entries_) {
        if (entry.snapshot != nullptr || !entry.disk) { continue; }
        consider(entry, entry.disk->ledger, entry.disk->identity, entry.disk->checkpoints, true);
    }
    return best;
}

std::shared_ptr<const ConversationSnapshot>
ConversationCache::acquire(const Match& match) {
    std::shared_ptr<const ConversationSnapshot> loaded;
    ConversationFileCatalog catalog;
    {
        std::lock_guard lock(mutex_);
        Entry* entry = find(match.entry);
        if (entry == nullptr) { return nullptr; }
        entry->last_use = clock_++;
        if (entry->snapshot != nullptr) { return entry->snapshot; }
        if (!entry->disk) { return nullptr; }
        catalog = *entry->disk;
    }

    // Read outside the catalog lock: a full payload takes tens of seconds off local disk, and
    // nothing about it needs the catalog held.
    loaded = std::make_shared<const ConversationSnapshot>(read_conversation_payload(catalog));

    std::lock_guard lock(mutex_);
    Entry* entry = find(match.entry);
    if (entry == nullptr) { return nullptr; }
    if (entry->snapshot == nullptr) {
        entry->snapshot = loaded;
        account_resident(*entry, *loaded);
        entry->last_use = clock_++;
        enforce_ram_budget();
        entry = find(match.entry);
        if (entry == nullptr || entry->snapshot == nullptr) { return loaded; }
    }
    return entry->snapshot;
}

std::shared_ptr<const ConversationSnapshot>
ConversationCache::merged(const Entry* parent, ConversationCapture&& capture, bool branch) const {
    auto out      = std::make_shared<ConversationSnapshot>();
    out->geometry = capture.geometry;
    out->ledger   = std::move(capture.ledger);
    out->identity = std::move(capture.identity);

    if (parent != nullptr && parent->snapshot != nullptr) {
        const ConversationSnapshot& base = *parent->snapshot;
        out->text                        = base.text;
        out->backend                     = base.backend;
        out->text.truncate(capture.shared_text_pages);
        out->backend.truncate(capture.shared_backend_pages);
        for (const ConversationCheckpoint& checkpoint : base.checkpoints) {
            // Every position an earlier checkpoint reads lies below the shared frontier, so the
            // merged page array serves it whether the lane appended to the parent or branched.
            if (checkpoint.frontier < capture.checkpoint.frontier &&
                checkpoint.frontier <= (branch ? capture.shared_frontier
                                               : capture.checkpoint.frontier) &&
                checkpoint.text_pages <= capture.checkpoint.text_pages &&
                checkpoint.backend_pages <= capture.checkpoint.backend_pages) {
                out->checkpoints.push_back(checkpoint);
            }
        }
    }

    const auto extend = [](HostKvPayload& destination, const HostKvPayload& addition) {
        if (destination.planes.empty()) {
            destination.planes.resize(addition.planes.size());
            for (std::size_t plane = 0; plane < addition.planes.size(); ++plane) {
                destination.planes[plane].group_bytes = addition.planes[plane].group_bytes;
            }
        }
        if (destination.planes.size() != addition.planes.size()) {
            throw std::logic_error("conversation capture plane count changed");
        }
        for (std::size_t plane = 0; plane < addition.planes.size(); ++plane) {
            HostPagePlane& target = destination.planes[plane];
            target.group_bytes    = addition.planes[plane].group_bytes;
            target.pages.insert(target.pages.end(), addition.planes[plane].pages.begin(),
                                addition.planes[plane].pages.end());
        }
    };
    extend(out->text, capture.new_text);
    if (!capture.new_backend.planes.empty()) { extend(out->backend, capture.new_backend); }

    out->checkpoints.push_back(std::move(capture.checkpoint));
    if (out->text.page_count() != out->checkpoints.back().text_pages ||
        out->backend.page_count() != out->checkpoints.back().backend_pages ||
        !out->text.consistent() || !out->backend.consistent()) {
        throw std::logic_error("merged conversation payload does not cover its newest checkpoint");
    }
    return out;
}

ConversationCache::EntryId ConversationCache::park(EntryId lane_entry,
                                                   ConversationCapture&& capture) {
    if (!options_.ram_enabled()) { return 0; }
    if (capture.checkpoint.frontier == 0 || capture.ledger.empty()) { return 0; }
    if (!(capture.geometry == geometry_)) {
        throw std::logic_error("conversation capture does not match this Engine");
    }

    std::shared_ptr<const ConversationSnapshot> published;
    EntryId id = 0;
    {
        std::lock_guard lock(mutex_);
        Entry* parent = lane_entry == 0 ? nullptr : find(lane_entry);
        if (parent != nullptr && parent->snapshot == nullptr) {
            // The lane's conversation was evicted from RAM while the lane went on using it. Its
            // pages are no longer here to share, so this capture starts a fresh conversation.
            parent = nullptr;
        }
        // A capture that extends the lane's conversation appends to it; one that diverges below
        // the shared frontier becomes a branch over the same host pages.
        bool branch = false;
        if (parent != nullptr &&
            (capture.shared_text_pages > parent->snapshot->text.page_count() ||
             capture.shared_backend_pages > parent->snapshot->backend.page_count())) {
            parent = nullptr;
        }
        if (parent != nullptr) {
            const std::vector<TokenId>& base = parent->snapshot->ledger;
            const auto shared = static_cast<std::size_t>(capture.shared_frontier);
            const bool extends =
                capture.ledger.size() >= base.size() &&
                std::equal(base.begin(), base.end(), capture.ledger.begin());
            if (!extends) {
                branch = shared != 0 && shared < base.size() && shared < capture.ledger.size() &&
                         std::equal(base.begin(),
                                    base.begin() + static_cast<std::ptrdiff_t>(shared),
                                    capture.ledger.begin());
                if (!branch) { parent = nullptr; }
            }
        }
        if (parent == nullptr &&
            (capture.shared_text_pages != 0 || capture.shared_backend_pages != 0)) {
            // The capture only carries the pages above its shared boundary, so without the parent
            // that supplied the rest there is no complete payload to cache. Refusing here keeps a
            // stale lane association from producing a snapshot with a hole in it.
            return 0;
        }

        std::shared_ptr<const ConversationSnapshot> snapshot =
            merged(parent, std::move(capture), branch);

        if (parent != nullptr && !branch) {
            id = parent->id;
            release_resident(*parent);
            parent->snapshot = snapshot;
            account_resident(*parent, *snapshot);
            parent->last_use = clock_++;
            trim_checkpoints(*parent);
        } else {
            Entry entry;
            entry.id       = next_id_++;
            entry.snapshot = snapshot;
            entry.last_use = clock_++;
            account_resident(entry, *snapshot);
            id = entry.id;
            entries_.push_back(std::move(entry));
            trim_checkpoints(*find(id));
        }
        Entry* owner = find(id);
        published    = owner->snapshot;
        enforce_ram_budget();
        if (find(id) == nullptr) {
            // The budget could not keep even this conversation; nothing is cached for the lane.
            return 0;
        }
    }
    if (options_.disk_enabled() && published != nullptr) { schedule_write(id, published); }
    return id;
}

void ConversationCache::trim_checkpoints(Entry& entry) {
    if (entry.snapshot == nullptr) { return; }
    const std::uint32_t cap = options_.context_checkpoints;
    const std::uint32_t step = options_.checkpoint_min_step;

    std::vector<ConversationCheckpoint> kept = entry.snapshot->checkpoints;
    if (kept.size() <= 1) { return; }

    // Spacing first: a boundary that sits inside `checkpoint_min_step` of the boundary before it
    // adds almost no reachable frontier, so it never earns a retained slot.
    if (step != 0) {
        std::vector<ConversationCheckpoint> spaced;
        spaced.reserve(kept.size());
        std::uint32_t previous = 0;
        for (std::size_t index = 0; index + 1 < kept.size(); ++index) {
            if (kept[index].frontier - previous < step) { continue; }
            previous = kept[index].frontier;
            spaced.push_back(kept[index]);
        }
        spaced.push_back(kept.back()); // the newest boundary is always retained
        kept = std::move(spaced);
    }
    while (kept.size() > static_cast<std::size_t>(cap) + 1ULL) {
        kept.erase(kept.begin() + static_cast<std::ptrdiff_t>(least_valuable_historical(kept)));
    }
    if (kept.size() == entry.snapshot->checkpoints.size()) { return; }

    auto replacement         = std::make_shared<ConversationSnapshot>(*entry.snapshot);
    replacement->checkpoints = std::move(kept);
    release_resident(entry);
    entry.snapshot = replacement;
    account_resident(entry, *replacement);
}

void ConversationCache::enforce_ram_budget() {
    while (resident_bytes_ > options_.ram_budget_bytes) {
        // Oldest inactive conversation first: strip its historical checkpoints, then release it
        // entirely. Page payload is what the budget is actually made of, and only dropping a
        // whole conversation releases pages.
        Entry* victim = nullptr;
        for (Entry& entry : entries_) {
            if (entry.pins != 0 || entry.snapshot == nullptr) { continue; }
            if (victim == nullptr || entry.last_use < victim->last_use) { victim = &entry; }
        }
        if (victim == nullptr) { return; }
        if (victim->snapshot->checkpoints.size() > 1) {
            auto replacement = std::make_shared<ConversationSnapshot>(*victim->snapshot);
            replacement->checkpoints.erase(
                replacement->checkpoints.begin() +
                static_cast<std::ptrdiff_t>(least_valuable_historical(replacement->checkpoints)));
            release_resident(*victim);
            victim->snapshot = replacement;
            account_resident(*victim, *replacement);
            continue;
        }
        const EntryId id = victim->id;
        release_resident(*victim);
        if (!victim->disk) { erase_entry(id); }
    }
}

void ConversationCache::erase_entry(EntryId id) {
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [id](const Entry& entry) { return entry.id == id; });
    if (it == entries_.end()) { return; }
    release_resident(*it);
    if (!it->file.empty()) {
        disk_bytes_ -= std::min(disk_bytes_, it->file_bytes);
        std::error_code ignored;
        std::filesystem::remove(it->file, ignored);
    }
    entries_.erase(it);
    {
        std::lock_guard lock(writer_mutex_);
        pending_.erase(id);
        write_paths_.erase(id);
    }
}

void ConversationCache::enforce_disk_budget(std::uint64_t incoming_bytes) {
    while (disk_bytes_ + incoming_bytes > options_.disk_budget_bytes) {
        Entry* victim = nullptr;
        for (Entry& entry : entries_) {
            if (entry.pins != 0 || entry.file.empty()) { continue; }
            if (victim == nullptr || entry.last_use < victim->last_use) { victim = &entry; }
        }
        if (victim == nullptr) { return; }
        const EntryId id = victim->id;
        if (victim->snapshot != nullptr) {
            disk_bytes_ -= std::min(disk_bytes_, victim->file_bytes);
            std::error_code ignored;
            std::filesystem::remove(victim->file, ignored);
            victim->file.clear();
            victim->file_bytes = 0;
            victim->disk.reset();
            std::lock_guard lock(writer_mutex_);
            pending_.erase(id);
            write_paths_.erase(id);
            continue;
        }
        erase_entry(id);
    }
}

void ConversationCache::pin(EntryId entry) {
    if (entry == 0) { return; }
    std::lock_guard lock(mutex_);
    Entry* found = find(entry);
    if (found != nullptr) {
        ++found->pins;
        found->last_use = clock_++;
    }
}

void ConversationCache::unpin(EntryId entry) {
    if (entry == 0) { return; }
    std::lock_guard lock(mutex_);
    Entry* found = find(entry);
    if (found != nullptr && found->pins != 0) { --found->pins; }
    enforce_ram_budget();
}

void ConversationCache::touch(EntryId entry) {
    if (entry == 0) { return; }
    std::lock_guard lock(mutex_);
    Entry* found = find(entry);
    if (found != nullptr) { found->last_use = clock_++; }
}

ConversationCacheStats ConversationCache::stats() const {
    std::lock_guard lock(mutex_);
    ConversationCacheStats out;
    out.conversations  = static_cast<std::uint32_t>(entries_.size());
    out.resident_bytes = resident_bytes_;
    out.disk_bytes     = disk_bytes_;
    for (const Entry& entry : entries_) {
        if (entry.snapshot == nullptr) { continue; }
        ++out.resident_conversations;
        out.checkpoints += static_cast<std::uint32_t>(entry.snapshot->checkpoints.size());
    }
    std::lock_guard writer_lock(writer_mutex_);
    out.pending_writes = static_cast<std::uint32_t>(pending_.size());
    return out;
}

void ConversationCache::schedule_write(EntryId id,
                                       std::shared_ptr<const ConversationSnapshot> snapshot) {
    std::filesystem::path path;
    {
        std::lock_guard lock(mutex_);
        Entry* entry = find(id);
        if (entry == nullptr) { return; }
        if (entry->file.empty()) {
            entry->file = options_.disk_dir / snapshot_file_name(next_file_serial_++);
        }
        path = entry->file;
    }
    {
        std::lock_guard lock(writer_mutex_);
        // Coalesce: only the newest snapshot of a conversation is worth writing, and a queued
        // older one is already superseded.
        const bool queued = pending_.find(id) != pending_.end();
        pending_[id]      = std::move(snapshot);
        write_paths_[id]  = std::move(path);
        if (!queued) { write_order_.push_back(id); }
    }
    writer_cv_.notify_one();
}

ConversationFileCatalog ConversationCache::catalog_of(const std::filesystem::path& path,
                                                      const ConversationSnapshot& snapshot,
                                                      std::uint64_t file_bytes) const {
    ConversationFileCatalog catalog;
    catalog.path       = path;
    catalog.geometry   = snapshot.geometry;
    catalog.ledger     = snapshot.ledger;
    catalog.identity   = snapshot.identity;
    catalog.file_bytes = file_bytes;
    catalog.checkpoints.reserve(snapshot.checkpoints.size());
    for (const ConversationCheckpoint& checkpoint : snapshot.checkpoints) {
        ConversationCheckpoint record = checkpoint;
        record.state.clear();
        record.state.shrink_to_fit();
        catalog.checkpoints.push_back(std::move(record));
    }
    return catalog;
}

void ConversationCache::writer_loop() {
    for (;;) {
        EntryId id = 0;
        std::shared_ptr<const ConversationSnapshot> snapshot;
        std::filesystem::path path;
        {
            std::unique_lock lock(writer_mutex_);
            writer_cv_.wait(lock, [this] { return writer_stopping_ || !write_order_.empty(); });
            if (write_order_.empty()) { return; }
            id = write_order_.front();
            write_order_.pop_front();
            const auto pending = pending_.find(id);
            if (pending == pending_.end()) { continue; }
            snapshot = pending->second;
            path     = write_paths_[id];
            pending_.erase(pending);
        }

        std::uint64_t previous_bytes = 0;
        {
            std::lock_guard lock(mutex_);
            const Entry* entry = find(id);
            if (entry == nullptr || entry->file != path) { continue; }
            previous_bytes = entry->file_bytes;
            enforce_disk_budget(0);
            if (find(id) == nullptr) { continue; }
        }

        std::uint64_t written = 0;
        try {
            written = write_conversation_file(path, *snapshot);
        } catch (const std::exception&) {
            // The previous valid file is untouched; the next completed turn tries again.
            continue;
        }

        std::lock_guard lock(mutex_);
        Entry* entry = find(id);
        if (entry == nullptr || entry->file != path) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            continue;
        }
        disk_bytes_ -= std::min(disk_bytes_, previous_bytes);
        disk_bytes_ += written;
        entry->file_bytes = written;
        // Keep a catalog for the file just published, so this conversation stays selectable and
        // lazily readable after the RAM budget releases its payload.
        entry->disk = catalog_of(path, *snapshot, written);
        enforce_disk_budget(0);
    }
}

} // namespace ninfer::runtime
