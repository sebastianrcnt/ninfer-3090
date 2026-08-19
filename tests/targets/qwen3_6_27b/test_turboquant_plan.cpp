#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdint>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::targets::qwen3_6_27b;
    using namespace ninfer::targets::qwen3_6_27b::detail;
    try {
        DeviceContext device(0);
        for (const std::uint32_t chunk : {2048U, 1024U, 512U, 256U, 128U}) {
            for (const bool graph : {true, false}) {
                EngineOptions options;
                options.max_context = 262144;
                options.kv_capacity = KvCapacityPolicy::explicit_capacity(262144);
                options.max_concurrency = 1;
                options.prefill_chunk = chunk;
                options.kv_cache = KvCacheStorage::TurboQuant;
                options.speculative.backend = SpeculativeBackend::DFlash;
                options.speculative.draft_tokens = 7;
                options.use_cuda_graph = graph;
                auto planner = Package::make_sequence_planner(device, options,
                                                               WeightsProfile::GroupwiseInt);
                const auto curve = planner.capacity_curve();
                auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);
                std::cout << "chunk=" << chunk << " graph=" << graph
                          << " reservation=" << plan.device_reservation_bytes()
                          << " workspace=" << plan.workspace_capacity_bytes()
                          << " request=" << plan.request_transient_capacity_bytes() << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
