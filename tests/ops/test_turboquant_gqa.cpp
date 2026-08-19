#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/turboquant.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD = 256;
constexpr int kQH = 24;
constexpr int kKVH = 4;
constexpr int kT = 8;

std::size_t q_index(int d, int h, int t) {
    return static_cast<std::size_t>(d) + kD * static_cast<std::size_t>(h + kQH * t);
}

std::size_t kv_index(int d, int h, int t) {
    return static_cast<std::size_t>(d) + kD * static_cast<std::size_t>(h + kKVH * t);
}

std::vector<double> reference(const std::vector<float>& q, const std::vector<float>& k,
                              const std::vector<float>& v, int tokens) {
    std::vector<double> out(static_cast<std::size_t>(kD) * kQH * tokens);
    std::vector<double> scores(tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int qh = 0; qh < kQH; ++qh) {
            const int kvh = qh / 6;
            double maximum = -1.0e300;
            for (int key = 0; key <= t; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    dot += static_cast<double>(q[q_index(d, qh, t)]) *
                           static_cast<double>(k[kv_index(d, kvh, key)]);
                }
                scores[key] = dot * 0.0625;
                maximum = std::max(maximum, scores[key]);
            }
            double sum = 0.0;
            for (int key = 0; key <= t; ++key) {
                scores[key] = std::exp(scores[key] - maximum);
                sum += scores[key];
            }
            for (int d = 0; d < kD; ++d) {
                double value = 0.0;
                for (int key = 0; key <= t; ++key) {
                    value += scores[key] / sum * static_cast<double>(v[kv_index(d, kvh, key)]);
                }
                out[q_index(d, qh, t)] = value;
            }
        }
    }
    return out;
}

double relative_l2(const std::vector<double>& got, const std::vector<double>& expected) {
    double error = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double delta = got[i] - expected[i];
        error += delta * delta;
        norm += expected[i] * expected[i];
    }
    return std::sqrt(error / std::max(norm, 1.0e-30));
}

struct Fixture {
    static constexpr int kPages = 2;
    static constexpr std::size_t kKeyPlaneBytes =
        static_cast<std::size_t>(ops::turboquant::kKeyBytes) * kPagedKVPageSize * kKVH * kPages;
    static constexpr std::size_t kValuePlaneBytes =
        static_cast<std::size_t>(ops::turboquant::kValueBytes) * kPagedKVPageSize * kKVH * kPages;

    DeviceBuffer key{kKeyPlaneBytes};
    DeviceBuffer value{kValuePlaneBytes};
    DeviceBuffer table{kPages * sizeof(std::int32_t)};

    Fixture() {
        cuda_check(cudaMemset(key.p, 0, key.bytes), "zero TurboQuant keys");
        cuda_check(cudaMemset(value.p, 0, value.bytes), "zero TurboQuant values");
        const std::int32_t pages[kPages] = {0, 1};
        table.copy_from_host(pages, sizeof(pages));
    }

    PagedKVLayerView view() {
        return {
            .k_pages = Tensor(key.p, DType::U8,
                              {ops::turboquant::kKeyBytes, kPagedKVPageSize, kKVH, kPages}),
            .v_pages = Tensor(value.p, DType::U8,
                              {ops::turboquant::kValueBytes, kPagedKVPageSize, kKVH, kPages}),
            .block_table = Tensor(table.p, DType::I32, {kPages}),
            .head_dim = kD,
            .num_kv_heads = kKVH,
            .dtype = DType::U8,
            .storage = KvCacheStorage::TurboQuant,
        };
    }

    PagedKVBatchLayerView batch_view() {
        return {
            .k_pages = Tensor(key.p, DType::U8,
                              {ops::turboquant::kKeyBytes, kPagedKVPageSize, kKVH, kPages}),
            .v_pages = Tensor(value.p, DType::U8,
                              {ops::turboquant::kValueBytes, kPagedKVPageSize, kKVH, kPages}),
            .block_tables = Tensor(table.p, DType::I32, {kPages, 1}),
            .head_dim = kD,
            .num_kv_heads = kKVH,
            .dtype = DType::U8,
            .storage = KvCacheStorage::TurboQuant,
        };
    }
};

int run() {
    std::vector<float> q(static_cast<std::size_t>(kD) * kQH * kT);
    std::vector<float> k(static_cast<std::size_t>(kD) * kKVH * kT);
    std::vector<float> v(static_cast<std::size_t>(kD) * kKVH * kT);
    fill_uniform(q, 7101, -0.3f, 0.3f);
    fill_uniform(k, 7102, -0.3f, 0.3f);
    fill_uniform(v, 7103, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<std::int32_t> positions(kT);
    std::iota(positions.begin(), positions.end(), 0);
    const std::int32_t table_row = 0;

    DeviceBuffer dq = to_device_bf16(q);
    DeviceBuffer dk = to_device_bf16(k);
    DeviceBuffer dv = to_device_bf16(v);
    DeviceBuffer dp = to_device(positions);
    DeviceBuffer drow = to_device(std::vector<std::int32_t>{table_row});
    DeviceBuffer eager(static_cast<std::size_t>(kD) * kQH * kT * sizeof(std::uint16_t));
    DeviceBuffer cached(eager.bytes);
    DeviceBuffer sequential(eager.bytes);
    DeviceBuffer graphed(eager.bytes);
    Fixture cache;

    Tensor tq(dq.p, DType::BF16, {kD, kQH, kT});
    Tensor tk(dk.p, DType::BF16, {kD, kKVH, kT});
    Tensor tv(dv.p, DType::BF16, {kD, kKVH, kT});
    Tensor tp(dp.p, DType::I32, {kT});
    Tensor tr(drow.p, DType::I32, {1});
    Tensor te(eager.p, DType::BF16, {kD, kQH, kT});
    Tensor tc(cached.p, DType::BF16, {kD, kQH, kT});
    Tensor ts(sequential.p, DType::BF16, {kD, kQH, kT});
    Tensor tg(graphed.p, DType::BF16, {kD, kQH, kT});
    const ops::GqaExecutionEnvelope envelope{1, kT};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQH, DType::U8, envelope, 1, kT, kT);
    DeviceBuffer workspace_storage(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, 0.0625f, cache.batch_view(), envelope,
                       workspace, te, nullptr);
    cuda_synchronize();
    const std::vector<double> eager_host = from_device_bf16(eager, q.size());
    const double rel = relative_l2(eager_host, reference(q, k, v, kT));
    std::cout << "TurboQuant A1 relative L2=" << rel << '\n';
    int failures = 0;
    if (!std::isfinite(rel) || rel > 0.25) {
        std::cerr << "TurboQuant A1 relative L2 " << rel << " exceeds 0.25\n";
        ++failures;
    }

    const std::vector<std::uint8_t> key_once = from_device<std::uint8_t>(cache.key, cache.key.bytes);
    const std::vector<std::uint8_t> value_once =
        from_device<std::uint8_t>(cache.value, cache.value.bytes);
    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();
    if (from_device<std::uint8_t>(cache.key, cache.key.bytes) != key_once ||
        from_device<std::uint8_t>(cache.value, cache.value.bytes) != value_once) {
        std::cerr << "TurboQuant append is not bit deterministic\n";
        ++failures;
    }

    workspace.reset();
    ops::gqa_attention_cached(tq, tp, 0.0625f, cache.view(), envelope, workspace, tc, nullptr);
    cuda_synchronize();
    if (from_device<std::uint16_t>(eager, q.size()) !=
        from_device<std::uint16_t>(cached, q.size())) {
        std::cerr << "TurboQuant A1/A3 packed-cache parity failed\n";
        ++failures;
    }

    // Speculative target verification evaluates T=K+1 positions together, while ordinary
    // greedy decode evaluates one position at a time.  Those two paths must be bit-identical at
    // the BF16 layer boundary; otherwise a tiny width-dependent rounding difference can change a
    // later argmax and violate speculative decoding losslessness.
    for (int token = 0; token < kT; ++token) {
        Tensor q_one = tq.slice(2, token, 1);
        Tensor p_one = tp.slice(0, token, 1);
        Tensor out_one = ts.slice(2, token, 1);
        workspace.reset();
        ops::gqa_attention_cached(q_one, p_one, 0.0625f, cache.view(), envelope, workspace,
                                  out_one, nullptr);
    }
    cuda_synchronize();
    const std::vector<std::uint16_t> batched_bits =
        from_device<std::uint16_t>(cached, q.size());
    const std::vector<std::uint16_t> sequential_bits =
        from_device<std::uint16_t>(sequential, q.size());
    if (batched_bits != sequential_bits) {
        std::size_t mismatches = 0;
        std::size_t first = 0;
        for (std::size_t i = 0; i < batched_bits.size(); ++i) {
            if (batched_bits[i] != sequential_bits[i]) {
                if (mismatches == 0) { first = i; }
                ++mismatches;
            }
        }
        std::cerr << "TurboQuant batched/sequential small-T parity failed: mismatches="
                  << mismatches << " first=" << first << " batched=0x" << std::hex
                  << batched_bits[first] << " sequential=0x" << sequential_bits[first]
                  << std::dec << '\n';
        ++failures;
    }

    DeviceBuffer sequential_live(eager.bytes);
    Tensor sequential_live_out(sequential_live.p, DType::BF16, {kD, kQH, kT});
    Fixture sequential_cache;
    for (int token = 0; token < kT; ++token) {
        workspace.reset();
        Tensor q_one = tq.slice(2, token, 1);
        Tensor k_one = tk.slice(2, token, 1);
        Tensor v_one = tv.slice(2, token, 1);
        Tensor p_one = tp.slice(0, token, 1);
        Tensor out_one = sequential_live_out.slice(2, token, 1);
        ops::gqa_attention(q_one, k_one, v_one, p_one, Tensor{}, tr, 0.0625f,
                           sequential_cache.batch_view(), envelope, workspace, out_one, nullptr);
    }
    cuda_synchronize();
    if (from_device<std::uint16_t>(eager, q.size()) !=
        from_device<std::uint16_t>(sequential_live, q.size())) {
        std::cerr << "TurboQuant batched/sequential append+attention parity failed\n";
        ++failures;
    }
    if (from_device<std::uint8_t>(cache.key, cache.key.bytes) !=
            from_device<std::uint8_t>(sequential_cache.key, sequential_cache.key.bytes) ||
        from_device<std::uint8_t>(cache.value, cache.value.bytes) !=
            from_device<std::uint8_t>(sequential_cache.value, sequential_cache.value.bytes)) {
        std::cerr << "TurboQuant batched/sequential packed append parity failed\n";
        ++failures;
    }

    // Adaptive DFlash keeps a fixed T=8 graph and masks unused proposal columns.  Verify every
    // valid prefix against the same sequential T=1 reference; varying the launch width is not an
    // equivalent test because it instantiates a different kernel.
    for (int valid = 1; valid <= kT; ++valid) {
        Fixture masked_cache;
        DeviceBuffer masked_output(eager.bytes);
        DeviceBuffer dvalid = to_device(std::vector<std::int32_t>{valid});
        Tensor valid_columns(dvalid.p, DType::I32, {1});
        Tensor masked_out(masked_output.p, DType::BF16, {kD, kQH, kT});
        workspace.reset();
        ops::gqa_attention(tq, tk, tv, tp, valid_columns, tr, 0.0625f,
                           masked_cache.batch_view(), envelope, workspace, masked_out, nullptr);
        cuda_synchronize();
        const std::vector<std::uint16_t> masked_bits =
            from_device<std::uint16_t>(masked_output, q.size());
        const std::size_t valid_elements = static_cast<std::size_t>(kD) * kQH * valid;
        if (!std::equal(masked_bits.begin(), masked_bits.begin() + valid_elements,
                        sequential_bits.begin())) {
            std::cerr << "TurboQuant masked T=8/sequential parity failed valid=" << valid
                      << '\n';
            ++failures;
        }
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create stream");
    workspace.reset();
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    ops::gqa_attention_cached(tq, tp, 0.0625f, cache.view(), envelope, workspace, tg, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
    cuda_check(cudaGraphLaunch(executable, stream), "launch graph");
    cuda_synchronize(stream);
    if (from_device<std::uint16_t>(cached, q.size()) !=
        from_device<std::uint16_t>(graphed, q.size())) {
        std::cerr << "TurboQuant CUDA Graph replay differs from eager\n";
        ++failures;
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    // Every fixed small-T instantiation is captured independently by production warmup. Exercise
    // all eight shapes here so alignment or topology defects cannot hide behind the T=8 case.
    for (int width = 1; width <= kT; ++width) {
        DeviceBuffer eager_width(static_cast<std::size_t>(kD) * kQH * width * sizeof(std::uint16_t));
        DeviceBuffer graph_width(eager_width.bytes);
        Tensor q_width = tq.slice(2, 0, width);
        Tensor p_width = tp.slice(0, 0, width);
        Tensor eager_out(eager_width.p, DType::BF16, {kD, kQH, width});
        Tensor graph_out(graph_width.p, DType::BF16, {kD, kQH, width});
        const ops::GqaExecutionEnvelope width_envelope{1, static_cast<std::uint32_t>(width)};

        workspace.reset();
        ops::gqa_attention_cached(q_width, p_width, 0.0625f, cache.view(), width_envelope,
                                  workspace, eager_out, nullptr);
        cuda_synchronize();
        const std::vector<std::uint16_t> width_bits = from_device<std::uint16_t>(
            eager_width, static_cast<std::size_t>(kD) * kQH * width);
        std::size_t width_mismatches = 0;
        for (std::size_t i = 0; i < width_bits.size(); ++i) {
            width_mismatches += width_bits[i] != sequential_bits[i];
        }
        if (width_mismatches != 0) {
            std::cerr << "TurboQuant T=" << width
                      << " differs from sequential T=1: mismatches=" << width_mismatches
                      << '\n';
        }

        cuda_check(cudaStreamCreate(&stream), "create width stream");
        workspace.reset();
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "begin width capture");
        ops::gqa_attention_cached(q_width, p_width, 0.0625f, cache.view(), width_envelope,
                                  workspace, graph_out, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end width capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "instantiate width graph");
        cuda_check(cudaGraphLaunch(executable, stream), "launch width graph");
        cuda_synchronize(stream);
        if (from_device<std::uint16_t>(eager_width,
                                       static_cast<std::size_t>(kD) * kQH * width) !=
            from_device<std::uint16_t>(graph_width,
                                       static_cast<std::size_t>(kD) * kQH * width)) {
            std::cerr << "TurboQuant CUDA Graph replay differs at T=" << width << '\n';
            ++failures;
        }
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    return failures;
}

int run_prompt() {
    constexpr int kPromptT = 128;
    std::vector<float> q(static_cast<std::size_t>(kD) * kQH * kPromptT);
    std::vector<float> k(static_cast<std::size_t>(kD) * kKVH * kPromptT);
    std::vector<float> v(static_cast<std::size_t>(kD) * kKVH * kPromptT);
    fill_uniform(q, 7201, -0.3f, 0.3f);
    fill_uniform(k, 7202, -0.3f, 0.3f);
    fill_uniform(v, 7203, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<std::int32_t> positions(kPromptT);
    std::iota(positions.begin(), positions.end(), 0);

    DeviceBuffer dq = to_device_bf16(q);
    DeviceBuffer dk = to_device_bf16(k);
    DeviceBuffer dv = to_device_bf16(v);
    DeviceBuffer dp = to_device(positions);
    DeviceBuffer drow = to_device(std::vector<std::int32_t>{0});
    DeviceBuffer output(static_cast<std::size_t>(kD) * kQH * kPromptT * sizeof(std::uint16_t));
    DeviceBuffer workspace_storage(256);
    WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});
    Fixture cache;

    Tensor tq(dq.p, DType::BF16, {kD, kQH, kPromptT});
    Tensor tk(dk.p, DType::BF16, {kD, kKVH, kPromptT});
    Tensor tv(dv.p, DType::BF16, {kD, kKVH, kPromptT});
    Tensor tp(dp.p, DType::I32, {kPromptT});
    Tensor tr(drow.p, DType::I32, {1});
    Tensor out(output.p, DType::BF16, {kD, kQH, kPromptT});
    const ops::GqaExecutionEnvelope envelope{1, kPromptT};
    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, 0.0625f, cache.batch_view(), envelope,
                       workspace, out, nullptr);
    cuda_synchronize();
    const double rel = relative_l2(from_device_bf16(output, q.size()),
                                   reference(q, k, v, kPromptT));
    std::cout << "TurboQuant prompt relative L2=" << rel << '\n';
    if (!std::isfinite(rel) || rel > 0.25) {
        std::cerr << "TurboQuant prompt relative L2 " << rel << " exceeds 0.25\n";
        return 1;
    }
    return 0;
}

int run_prompt_benchmark(int tokens, int iterations) {
    if (tokens <= 0 || tokens > 16384 || iterations <= 0) {
        throw std::invalid_argument("TurboQuant benchmark arguments out of range");
    }
    const int pages = (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
    std::vector<float> q(static_cast<std::size_t>(kD) * kQH * tokens);
    std::vector<float> k(static_cast<std::size_t>(kD) * kKVH * tokens);
    std::vector<float> v(static_cast<std::size_t>(kD) * kKVH * tokens);
    fill_uniform(q, 7401, -0.3f, 0.3f);
    fill_uniform(k, 7402, -0.3f, 0.3f);
    fill_uniform(v, 7403, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<std::int32_t> positions(tokens);
    std::iota(positions.begin(), positions.end(), 0);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);

    DeviceBuffer dq = to_device_bf16(q);
    DeviceBuffer dk = to_device_bf16(k);
    DeviceBuffer dv = to_device_bf16(v);
    DeviceBuffer dp = to_device(positions);
    DeviceBuffer dt = to_device(table);
    DeviceBuffer key(static_cast<std::size_t>(ops::turboquant::kKeyBytes) *
                     kPagedKVPageSize * kKVH * pages);
    DeviceBuffer value(static_cast<std::size_t>(ops::turboquant::kValueBytes) *
                       kPagedKVPageSize * kKVH * pages);
    DeviceBuffer output(static_cast<std::size_t>(kD) * kQH * tokens * sizeof(std::uint16_t));
    PagedKVLayerView cache{
        .k_pages = Tensor(key.p, DType::U8,
                          {ops::turboquant::kKeyBytes, kPagedKVPageSize, kKVH, pages}),
        .v_pages = Tensor(value.p, DType::U8,
                          {ops::turboquant::kValueBytes, kPagedKVPageSize, kKVH, pages}),
        .block_table = Tensor(dt.p, DType::I32, {pages}),
        .head_dim = kD,
        .num_kv_heads = kKVH,
        .dtype = DType::U8,
        .storage = KvCacheStorage::TurboQuant,
    };
    Tensor tq(dq.p, DType::BF16, {kD, kQH, tokens});
    Tensor tk(dk.p, DType::BF16, {kD, kKVH, tokens});
    Tensor tv(dv.p, DType::BF16, {kD, kKVH, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor out(output.p, DType::BF16, {kD, kQH, tokens});
    const ops::GqaExecutionEnvelope envelope{1, static_cast<std::uint32_t>(tokens)};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQH, DType::U8, envelope, 1, tokens, tokens);
    DeviceBuffer workspace_storage(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});

    ops::gqa_kv_append(tk, tv, tp, cache, nullptr);
    workspace.reset();
    ops::gqa_attention_cached(tq, tp, 0.0625f, cache, envelope, workspace, out, nullptr);
    cuda_synchronize();

    cudaEvent_t begin = nullptr, end = nullptr;
    cuda_check(cudaEventCreate(&begin), "create benchmark begin event");
    cuda_check(cudaEventCreate(&end), "create benchmark end event");
    cuda_check(cudaEventRecord(begin), "record benchmark begin");
    for (int i = 0; i < iterations; ++i) {
        workspace.reset();
        ops::gqa_attention_cached(tq, tp, 0.0625f, cache, envelope, workspace, out, nullptr);
    }
    cuda_check(cudaEventRecord(end), "record benchmark end");
    cuda_check(cudaEventSynchronize(end), "synchronize benchmark end");
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "measure benchmark events");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    std::cout << "TurboQuant prompt benchmark T=" << tokens << " iterations=" << iterations
              << " mean_ms=" << milliseconds / iterations << '\n';
    return 0;
}

int run_decode_benchmark(int context, int query_tokens, int iterations) {
    if (context <= 0 || context > 16384 || query_tokens <= 0 || query_tokens > 8 ||
        query_tokens > context || iterations <= 0) {
        throw std::invalid_argument("TurboQuant decode benchmark arguments out of range");
    }
    const int pages = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
    std::vector<float> q(static_cast<std::size_t>(kD) * kQH * query_tokens);
    std::vector<float> k(static_cast<std::size_t>(kD) * kKVH * context);
    std::vector<float> v(static_cast<std::size_t>(kD) * kKVH * context);
    fill_uniform(q, 7501, -0.3f, 0.3f);
    fill_uniform(k, 7502, -0.3f, 0.3f);
    fill_uniform(v, 7503, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<std::int32_t> cache_positions(context);
    std::iota(cache_positions.begin(), cache_positions.end(), 0);
    std::vector<std::int32_t> query_positions(query_tokens);
    std::iota(query_positions.begin(), query_positions.end(), context - query_tokens);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);

    DeviceBuffer dq = to_device_bf16(q);
    DeviceBuffer dk = to_device_bf16(k);
    DeviceBuffer dv = to_device_bf16(v);
    DeviceBuffer dcp = to_device(cache_positions);
    DeviceBuffer dqp = to_device(query_positions);
    DeviceBuffer dt = to_device(table);
    DeviceBuffer key(static_cast<std::size_t>(ops::turboquant::kKeyBytes) *
                     kPagedKVPageSize * kKVH * pages);
    DeviceBuffer value(static_cast<std::size_t>(ops::turboquant::kValueBytes) *
                       kPagedKVPageSize * kKVH * pages);
    DeviceBuffer output(static_cast<std::size_t>(kD) * kQH * query_tokens *
                        sizeof(std::uint16_t));
    DeviceBuffer sequential_output(output.bytes);
    PagedKVLayerView cache{
        .k_pages = Tensor(key.p, DType::U8,
                          {ops::turboquant::kKeyBytes, kPagedKVPageSize, kKVH, pages}),
        .v_pages = Tensor(value.p, DType::U8,
                          {ops::turboquant::kValueBytes, kPagedKVPageSize, kKVH, pages}),
        .block_table = Tensor(dt.p, DType::I32, {pages}),
        .head_dim = kD,
        .num_kv_heads = kKVH,
        .dtype = DType::U8,
        .storage = KvCacheStorage::TurboQuant,
    };
    Tensor tq(dq.p, DType::BF16, {kD, kQH, query_tokens});
    Tensor tk(dk.p, DType::BF16, {kD, kKVH, context});
    Tensor tv(dv.p, DType::BF16, {kD, kKVH, context});
    Tensor tcp(dcp.p, DType::I32, {context});
    Tensor tqp(dqp.p, DType::I32, {query_tokens});
    Tensor out(output.p, DType::BF16, {kD, kQH, query_tokens});
    Tensor sequential_out(sequential_output.p, DType::BF16, {kD, kQH, query_tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(context),
                                              static_cast<std::uint32_t>(context)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kQH, DType::U8, envelope, 1, query_tokens, query_tokens);
    DeviceBuffer workspace_storage(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});

    ops::gqa_kv_append(tk, tv, tcp, cache, nullptr);
    workspace.reset();
    ops::gqa_attention_cached(tq, tqp, 0.0625f, cache, envelope, workspace, out, nullptr);
    cuda_synchronize();

    for (int token = 0; token < query_tokens; ++token) {
        workspace.reset();
        Tensor q_one = tq.slice(2, token, 1);
        Tensor p_one = tqp.slice(0, token, 1);
        Tensor out_one = sequential_out.slice(2, token, 1);
        ops::gqa_attention_cached(q_one, p_one, 0.0625f, cache, envelope, workspace, out_one,
                                  nullptr);
    }
    cuda_synchronize();
    const std::size_t elements = static_cast<std::size_t>(kD) * kQH * query_tokens;
    const std::vector<std::uint16_t> batched_bits =
        from_device<std::uint16_t>(output, elements);
    const std::vector<std::uint16_t> sequential_bits =
        from_device<std::uint16_t>(sequential_output, elements);
    if (batched_bits != sequential_bits) {
        std::size_t mismatches = 0;
        std::size_t first = 0;
        for (std::size_t i = 0; i < elements; ++i) {
            if (batched_bits[i] != sequential_bits[i]) {
                if (mismatches == 0) { first = i; }
                ++mismatches;
            }
        }
        std::cerr << "TurboQuant decode batched/sequential parity failed context=" << context
                  << " T=" << query_tokens << " mismatches=" << mismatches
                  << " first=" << first << '\n';
        return 1;
    }

    cudaEvent_t begin = nullptr, end = nullptr;
    cuda_check(cudaEventCreate(&begin), "create decode benchmark begin event");
    cuda_check(cudaEventCreate(&end), "create decode benchmark end event");
    cuda_check(cudaEventRecord(begin), "record decode benchmark begin");
    for (int i = 0; i < iterations; ++i) {
        workspace.reset();
        ops::gqa_attention_cached(tq, tqp, 0.0625f, cache, envelope, workspace, out, nullptr);
    }
    cuda_check(cudaEventRecord(end), "record decode benchmark end");
    cuda_check(cudaEventSynchronize(end), "synchronize decode benchmark end");
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "measure decode benchmark events");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    std::cout << "TurboQuant decode benchmark context=" << context
              << " T=" << query_tokens << " iterations=" << iterations
              << " mean_ms=" << milliseconds / iterations << '\n';
    return 0;
}

int run_max_logical_address() {
    constexpr int kLogicalPages = 262144 / kPagedKVPageSize;
    constexpr int kPosition = 262144 - 1;
    std::vector<float> q(static_cast<std::size_t>(kD) * kQH);
    std::vector<float> k(static_cast<std::size_t>(kD) * kKVH);
    std::vector<float> v(static_cast<std::size_t>(kD) * kKVH);
    fill_uniform(q, 7301, -0.3f, 0.3f);
    fill_uniform(k, 7302, -0.3f, 0.3f);
    fill_uniform(v, 7303, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    DeviceBuffer dq = to_device_bf16(q);
    DeviceBuffer dk = to_device_bf16(k);
    DeviceBuffer dv = to_device_bf16(v);
    DeviceBuffer dp = to_device(std::vector<std::int32_t>{kPosition});
    std::vector<std::int32_t> table(kLogicalPages, 0);
    table.back() = 1;
    DeviceBuffer dtable = to_device(table);
    DeviceBuffer output(static_cast<std::size_t>(kD) * kQH * sizeof(std::uint16_t));
    Fixture storage;
    PagedKVLayerView cache = storage.view();
    cache.block_table = Tensor(dtable.p, DType::I32, {kLogicalPages});

    Tensor tq(dq.p, DType::BF16, {kD, kQH, 1});
    Tensor tk(dk.p, DType::BF16, {kD, kKVH, 1});
    Tensor tv(dv.p, DType::BF16, {kD, kKVH, 1});
    Tensor tp(dp.p, DType::I32, {1});
    Tensor out(output.p, DType::BF16, {kD, kQH, 1});
    const ops::GqaExecutionEnvelope envelope{262144, 262144};
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(kQH, DType::U8, envelope, 1, 1, 1);
    DeviceBuffer workspace_storage(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});

    ops::gqa_kv_append(tk, tv, tp, cache, nullptr);
    ops::gqa_attention_cached(tq, tp, 0.0625f, cache, envelope, workspace, out, nullptr);
    cuda_synchronize();
    for (const double value : from_device_bf16(output, q.size())) {
        if (!std::isfinite(value)) {
            std::cerr << "TurboQuant max logical address produced a non-finite value\n";
            return 1;
        }
    }
    std::cout << "TurboQuant max logical address=262143 PASS\n";
    return 0;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "turboquant_gqa: SKIP (CUDA unavailable)\n";
            return 77;
        }
        if (const char* value = std::getenv("NINFER_TQ_BENCH_DECODE_CONTEXT")) {
            const int context = std::atoi(value);
            const char* token_value = std::getenv("NINFER_TQ_BENCH_DECODE_T");
            const char* iteration_value = std::getenv("NINFER_TQ_BENCH_ITERS");
            return run_decode_benchmark(context,
                                        token_value == nullptr ? 1 : std::atoi(token_value),
                                        iteration_value == nullptr ? 20
                                                                   : std::atoi(iteration_value));
        }
        if (const char* value = std::getenv("NINFER_TQ_BENCH_T")) {
            const int tokens = std::atoi(value);
            const char* iteration_value = std::getenv("NINFER_TQ_BENCH_ITERS");
            return run_prompt_benchmark(tokens,
                                        iteration_value == nullptr ? 5
                                                                   : std::atoi(iteration_value));
        }
        const int failures = run() + run_prompt() + run_max_logical_address();
        if (failures != 0) { return 1; }
        std::cout << "turboquant_gqa: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "turboquant_gqa: " << error.what() << '\n';
        return 1;
    }
}
