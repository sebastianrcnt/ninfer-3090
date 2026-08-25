#include "serve/stream_writer.h"

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

void StreamWriter::write(const std::string& item) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!write_locked(item)) {
        cancelled_.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void StreamWriter::write_all(const std::vector<std::string>& items) {
    for (const std::string& item : items) { write(item); }
}

bool StreamWriter::keepalive(const std::string& item,
                             const std::chrono::steady_clock::duration idle_for,
                             std::chrono::steady_clock::time_point& next_check) {
    if (cancelled_.load(std::memory_order_acquire)) { return false; }

    // try_to_lock, not a wait: holding this mutex means the request thread is writing a real
    // event, which resets the client's timer by itself. Waiting would only queue a redundant
    // keep-alive behind it, and would extend how long this thread can be found holding the
    // mutex the request thread needs.
    const std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    const auto now = std::chrono::steady_clock::now();
    if (!lock.owns_lock()) {
        next_check = now + idle_for;
        return true;
    }
    if (now - last_write_ < idle_for) {
        next_check = last_write_ + idle_for;
        return true;
    }
    if (!write_locked(item)) {
        cancelled_.store(true, std::memory_order_release);
        return false;
    }
    next_check = last_write_ + idle_for;
    return true;
}

bool StreamWriter::write_locked(const std::string& item) {
    if (cancelled_.load(std::memory_order_acquire) ||
        (sink_.is_writable && !sink_.is_writable()) || !sink_.write(item.data(), item.size())) {
        return false;
    }
    last_write_ = std::chrono::steady_clock::now();
    return true;
}

void StreamKeepAlive::stop() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void StreamKeepAlive::run(StreamWriter& writer, const std::string& payload) {
    auto deadline = std::chrono::steady_clock::now() + interval_;
    std::unique_lock<std::mutex> lock(mutex_);
    while (!cv_.wait_until(lock, deadline, [this] { return stopped_; })) {
        // The sink write happens with this mutex released, so stop() never waits behind it.
        lock.unlock();
        const bool open = writer.keepalive(payload, interval_, deadline);
        lock.lock();
        if (!open) { return; }
    }
}

}  // namespace ninfer::serve
