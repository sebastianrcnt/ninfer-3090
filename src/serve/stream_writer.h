#pragma once

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

// A streamed body writes nothing while the engine is working, and there is no bound on how
// long that lasts: a cold prefill is silent until the first token, and the tool-call filter
// withholds every content delta from the moment it sees a marker until generation ends.
// Clients close an idle body long before either window can close -- undici, which most Node
// clients use, defaults to 300 s -- so the request dies before it can produce the bytes that
// would have kept it alive. An ignorable event on this interval resets that timer, and a
// failed write detects a client that has already gone.
inline constexpr auto kStreamKeepAliveInterval = std::chrono::seconds(15);

// Thrown when the client is gone. Handlers unwind on it and report the request as
// disconnected rather than failed.
class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

// The sole writer of one streamed response body. Two threads write to the sink -- the request
// thread carrying generated events, and the keep-alive timer -- so writes are serialized here,
// along with the time of the last one, which is what "this stream is idle" is measured
// against. Lives inside the content provider that owns the sink.
class StreamWriter {
public:
    // `cancelled` outlives the sink: httplib's completion callback sets it when the
    // connection ends, and the generation loop polls it.
    StreamWriter(httplib::DataSink& sink, std::atomic<bool>& cancelled)
        : sink_(sink), cancelled_(cancelled) {}

    StreamWriter(const StreamWriter&)            = delete;
    StreamWriter& operator=(const StreamWriter&) = delete;

    // Writes one already-encoded event, throwing ClientDisconnected if the client is gone.
    void write(const std::string& item);
    void write_all(const std::vector<std::string>& items);

    // Writes `item` only if the stream has been idle for `idle_for`, and reports through
    // `next_check` when it should be examined again. Returns false once the stream is over.
    // Never throws -- the request thread owns the failure path -- and never waits for a write
    // in flight, because such a write is itself the activity a keep-alive stands in for.
    bool keepalive(const std::string& item, std::chrono::steady_clock::duration idle_for,
                   std::chrono::steady_clock::time_point& next_check);

private:
    bool write_locked(const std::string& item);

    httplib::DataSink& sink_;
    std::atomic<bool>& cancelled_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point last_write_ = std::chrono::steady_clock::now();
};

// Writes `payload` whenever its stream has been quiet for kStreamKeepAliveInterval, for as
// long as it exists. Stopping is idempotent, and must happen before the body is completed so
// that nothing is written after the terminating chunk.
class StreamKeepAlive {
public:
    // The interval is a constructor argument so that a test can observe the idle condition
    // without waiting a quarter of a minute for each transition.
    explicit StreamKeepAlive(StreamWriter& writer, std::string payload,
                             std::chrono::steady_clock::duration interval =
                                 kStreamKeepAliveInterval)
        : interval_(interval),
          worker_([this, &writer, payload = std::move(payload)] { run(writer, payload); }) {}

    StreamKeepAlive(const StreamKeepAlive&)            = delete;
    StreamKeepAlive& operator=(const StreamKeepAlive&) = delete;

    ~StreamKeepAlive() { stop(); }

    void stop();

private:
    void run(StreamWriter& writer, const std::string& payload);

    std::chrono::steady_clock::duration interval_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
    std::thread worker_;
};

}  // namespace ninfer::serve
