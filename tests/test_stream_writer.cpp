#include "serve/stream_writer.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using ninfer::serve::ClientDisconnected;
using ninfer::serve::StreamKeepAlive;
using ninfer::serve::StreamWriter;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

// A sink that records what a client would have received.
class RecordingSink {
public:
    RecordingSink() {
        sink.is_writable = [this] { return writable_.load(); };
        sink.write       = [this](const char* data, std::size_t length) {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!writable_.load()) { return false; }
            items_.emplace_back(data, length);
            return true;
        };
    }

    void disconnect() { writable_.store(false); }

    [[nodiscard]] std::size_t count_of(const std::string& item) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        std::size_t seen = 0;
        for (const std::string& written : items_) {
            if (written == item) { ++seen; }
        }
        return seen;
    }

    [[nodiscard]] std::size_t size() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    httplib::DataSink sink;

private:
    mutable std::mutex mutex_;
    std::vector<std::string> items_;
    std::atomic<bool> writable_{true};
};

// A stream that writes nothing -- a cold prefill, or a tool call being buffered by the
// content filter -- must still put bytes on the wire, or a client closes the idle body.
int test_idle_stream_is_kept_alive() {
    RecordingSink recording;
    std::atomic<bool> cancelled{false};
    StreamWriter writer(recording.sink, cancelled);

    {
        StreamKeepAlive keepalive(writer, ": ka\n\n", 20ms);
        std::this_thread::sleep_for(300ms);
    }

    const std::size_t sent = recording.count_of(": ka\n\n");
    return check(sent >= 3, "an idle stream was not kept alive (" + std::to_string(sent) +
                                " keep-alives in 300 ms at a 20 ms interval)");
}

// The condition is silence, not the absence of a first token: a stream that keeps writing
// must never spend bytes on keep-alives.
int test_busy_stream_is_not_kept_alive() {
    RecordingSink recording;
    std::atomic<bool> cancelled{false};
    StreamWriter writer(recording.sink, cancelled);

    {
        StreamKeepAlive keepalive(writer, ": ka\n\n", 200ms);
        for (int i = 0; i < 40; ++i) {
            writer.write("delta\n\n");
            std::this_thread::sleep_for(10ms);
        }
    }

    int failures = 0;
    failures += check(recording.count_of(": ka\n\n") == 0,
                      "a stream writing every 10 ms emitted a 200 ms keep-alive");
    failures += check(recording.count_of("delta\n\n") == 40, "content deltas were lost");
    return failures;
}

// Stopping must be immediate and final: the handler completes the body right after, and a
// keep-alive written past the terminating chunk would corrupt the response.
int test_stop_ends_the_timer() {
    RecordingSink recording;
    std::atomic<bool> cancelled{false};
    StreamWriter writer(recording.sink, cancelled);

    StreamKeepAlive keepalive(writer, ": ka\n\n", 20ms);
    std::this_thread::sleep_for(100ms);
    keepalive.stop();
    const std::size_t at_stop = recording.size();
    std::this_thread::sleep_for(100ms);

    int failures = 0;
    failures += check(at_stop > 0, "the timer never ran");
    failures += check(recording.size() == at_stop, "the stream was written to after stop()");
    keepalive.stop();  // idempotent
    return failures;
}

// A keep-alive is how a departed client is noticed while the engine is still working, so a
// failed one must cancel the request instead of throwing on the timer thread.
int test_failed_keepalive_cancels_the_request() {
    RecordingSink recording;
    std::atomic<bool> cancelled{false};
    StreamWriter writer(recording.sink, cancelled);

    StreamKeepAlive keepalive(writer, ": ka\n\n", 20ms);
    recording.disconnect();
    for (int i = 0; i < 100 && !cancelled.load(); ++i) { std::this_thread::sleep_for(10ms); }
    keepalive.stop();

    int failures = 0;
    failures += check(cancelled.load(), "a keep-alive to a gone client did not cancel the request");

    bool threw = false;
    try {
        writer.write("delta\n\n");
    } catch (const ClientDisconnected&) { threw = true; }
    failures += check(threw, "a write on a cancelled stream did not report the disconnect");
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_idle_stream_is_kept_alive();
    failures += test_busy_stream_is_not_kept_alive();
    failures += test_stop_ends_the_timer();
    failures += test_failed_keepalive_cancels_the_request();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
