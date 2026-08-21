#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

} // namespace

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    return true;
}

namespace {

// Length-prefixed vectors, little-endian host order. The file never leaves the machine that wrote
// it — a restore already refuses on model and KV-format mismatch — so no byte-order conversion.
template <typename T>
void append_vector(std::vector<std::byte>& out, const std::vector<T>& values) {
    const std::uint64_t count = values.size();
    const auto* count_bytes   = reinterpret_cast<const std::byte*>(&count);
    out.insert(out.end(), count_bytes, count_bytes + sizeof(count));
    if (values.empty()) { return; }
    const auto* payload = reinterpret_cast<const std::byte*>(values.data());
    out.insert(out.end(), payload, payload + values.size() * sizeof(T));
}

template <typename T>
void read_vector(std::span<const std::byte>& cursor, std::vector<T>& values) {
    std::uint64_t count = 0;
    if (cursor.size() < sizeof(count)) {
        throw std::invalid_argument("prefix identity payload is truncated");
    }
    std::memcpy(&count, cursor.data(), sizeof(count));
    cursor = cursor.subspan(sizeof(count));
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
    if (count > cursor.size() / sizeof(T)) {
        throw std::invalid_argument("prefix identity payload is truncated");
    }
    values.resize(static_cast<std::size_t>(count));
    if (count != 0) { std::memcpy(values.data(), cursor.data(), bytes); }
    cursor = cursor.subspan(bytes);
}

} // namespace

// Fixed-layout head of one VisionItem. The two vectors that follow it are length-prefixed like
// every other section, so a video item with per-frame timestamps round-trips as well as an image.
struct SerializedVisionItem {
    std::uint8_t modality;
    std::uint8_t reserved[7];
    std::int32_t grid_temporal;
    std::int32_t grid_height;
    std::int32_t grid_width;
    std::int32_t padding;
    std::uint64_t patch_begin;
    std::uint64_t patch_count;
    std::array<std::uint8_t, 32> content_digest;
};

static_assert(sizeof(SerializedVisionItem) == 72, "vision item head layout is part of the format");
static_assert(sizeof(TokenSpan) == 16, "token span layout is part of the format");

std::vector<std::byte> ResidentPrefixIdentity::serialize() const {
    std::vector<std::byte> out;
    append_vector(out, token_types_);
    for (const std::vector<std::int32_t>& axis : positions_) { append_vector(out, axis); }

    // Vision items participate in matches(), so persisting an identity without them would let a
    // restored prefix compare equal to a prompt carrying different media.
    const std::uint64_t item_count = vision_items_.size();
    const auto* count_bytes        = reinterpret_cast<const std::byte*>(&item_count);
    out.insert(out.end(), count_bytes, count_bytes + sizeof(item_count));
    for (const VisionItem& item : vision_items_) {
        SerializedVisionItem head{};
        head.modality       = static_cast<std::uint8_t>(item.modality);
        head.grid_temporal  = item.grid.temporal;
        head.grid_height    = item.grid.height;
        head.grid_width     = item.grid.width;
        head.patch_begin    = item.patch_begin;
        head.patch_count    = item.patch_count;
        head.content_digest = item.content_digest;
        const auto* head_bytes = reinterpret_cast<const std::byte*>(&head);
        out.insert(out.end(), head_bytes, head_bytes + sizeof(head));
        append_vector(out, item.timestamps);
        append_vector(out, item.token_spans);
    }
    return out;
}

void ResidentPrefixIdentity::deserialize(std::span<const std::byte> bytes) {
    clear();
    try {
        std::span<const std::byte> cursor = bytes;
        read_vector(cursor, token_types_);
        for (std::vector<std::int32_t>& axis : positions_) { read_vector(cursor, axis); }

        std::uint64_t item_count = 0;
        if (cursor.size() < sizeof(item_count)) {
            throw std::invalid_argument("prefix identity payload is truncated");
        }
        std::memcpy(&item_count, cursor.data(), sizeof(item_count));
        cursor = cursor.subspan(sizeof(item_count));
        if (item_count > cursor.size() / sizeof(SerializedVisionItem)) {
            throw std::invalid_argument("prefix identity vision item count is implausible");
        }
        vision_items_.resize(static_cast<std::size_t>(item_count));
        for (VisionItem& item : vision_items_) {
            SerializedVisionItem head{};
            if (cursor.size() < sizeof(head)) {
                throw std::invalid_argument("prefix identity payload is truncated");
            }
            std::memcpy(&head, cursor.data(), sizeof(head));
            cursor = cursor.subspan(sizeof(head));
            if (head.modality != static_cast<std::uint8_t>(PromptModality::Image) &&
                head.modality != static_cast<std::uint8_t>(PromptModality::Video)) {
                throw std::invalid_argument("prefix identity vision modality is unknown");
            }
            item.modality       = static_cast<PromptModality>(head.modality);
            item.grid.temporal  = head.grid_temporal;
            item.grid.height    = head.grid_height;
            item.grid.width     = head.grid_width;
            item.patch_begin    = static_cast<std::size_t>(head.patch_begin);
            item.patch_count    = static_cast<std::size_t>(head.patch_count);
            item.content_digest = head.content_digest;
            read_vector(cursor, item.timestamps);
            read_vector(cursor, item.token_spans);
        }
        if (!cursor.empty()) {
            throw std::invalid_argument("prefix identity payload has trailing bytes");
        }
        for (const std::vector<std::int32_t>& axis : positions_) {
            if (axis.size() != token_types_.size()) {
                throw std::invalid_argument("prefix identity axes disagree with token count");
            }
        }
    } catch (...) {
        clear();
        throw;
    }
}

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      resident_tokens.begin()) &&
           resident_identity.matches(prompt, count);
}

} // namespace ninfer::targets::qwen3_6::detail
