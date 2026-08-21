#include "product/load_progress/load_progress.h"
#include "serve/console_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::size_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (static_cast<double>(bytes) >= kGiB) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ninfer::serve::ServeOptions options = ninfer::serve::parse_serve_options(argc, argv);
        if (options.help_requested) {
            std::cout << ninfer::serve::serve_usage_text(argv[0]);
            return 0;
        }

        using Clock = std::chrono::steady_clock;
        ninfer::serve::HttpServer server(options);
        if (!server.bind()) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "loading model...");
        auto load_progress_options        = ninfer::product::stderr_load_progress_options();
        load_progress_options.line_prefix = [] {
            return ninfer::serve::current_console_log_prefix(ninfer::serve::ConsoleLogLevel::Info);
        };
        ninfer::product::LoadProgressRenderer load_progress(std::cerr,
                                                            std::move(load_progress_options));
        const auto load_start = Clock::now();
        ninfer::serve::GenerationService service(options, load_progress.callback());
        server.attach(service);
        std::ostringstream loaded;
        loaded << "model loaded in "
               << std::chrono::duration<double>(Clock::now() - load_start).count() << " s";
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, loaded.str());

        const ninfer::MemorySummary memory = service.memory_summary();
        std::ostringstream capacity;
        capacity << "KV capacity "
                 << (memory.kv_capacity_mode == ninfer::KvCapacityMode::Automatic ? "auto"
                                                                                  : "explicit")
                 << " resolved=" << memory.kv_capacity
                 << " tokens pages=" << memory.kv_capacity_page_groups << '/'
                 << memory.kv_capacity_max_page_groups
                 << " runtime=" << format_bytes(memory.runtime_reservation_bytes)
                 << " free-after-weights=" << format_bytes(memory.available_after_weights_bytes)
                 << " free-after-startup=" << format_bytes(memory.available_after_startup_bytes)
                 << " headroom=" << format_bytes(memory.kv_capacity_headroom_bytes)
                 << " slack=" << format_bytes(memory.planned_slack_bytes)
                 << " graphs=" << format_bytes(memory.cuda_graph_observed_bytes) << '/'
                 << format_bytes(memory.cuda_graph_allowance_bytes);
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, capacity.str());

        if (options.conversation_cache_ram_bytes != 0) {
            const ninfer::ConversationCacheSummary cache = service.conversation_cache_summary();
            std::ostringstream conversations;
            conversations << "conversation cache ram="
                          << format_bytes(static_cast<std::size_t>(options.conversation_cache_ram_bytes))
                          << " checkpoints/conversation=" << options.context_checkpoints
                          << " min-step=" << options.checkpoint_min_step << " tokens";
            if (options.conversation_cache_dir.empty()) {
                conversations << " disk=disabled";
            } else {
                conversations << " disk=" << options.conversation_cache_dir << '@'
                              << format_bytes(static_cast<std::size_t>(options.conversation_cache_disk_bytes))
                              << " restored=" << cache.adopted_at_startup << " conversation(s)";
            }
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info,
                                             conversations.str());
        }

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "warming up...");
        service.warmup();

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::ostringstream listening;
        listening << "listening on http://" << options.host << ':' << options.port
                  << " (model id: " << server.public_model_id()
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ')';
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, listening.str());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }
        return 0;
    } catch (const std::exception& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    }
}
